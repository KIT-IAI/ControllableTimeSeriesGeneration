import functools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib
import xarray as xr
from pywatts.callbacks import CSVCallback, LinePlotCallback
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.modules import CalendarExtraction, CalendarFeature, Sampler, FunctionModule, SKLearnWrapper, \
    DriftInformation, StatisticExtraction, StatisticFeature
from pywatts.modules.postprocessing.merger import Merger
from sklearn.preprocessing import StandardScaler

from ablation_study_pipeline import get_reshaping
from generative_models.INN import INNWrapper, subnet, INN

COND_FEATURES = 4

CONDITIONS = ["CalendarExtraction", "statistics"]

AGGREGATORS = [
    ("first", 0),
    ("mean", "mean"),
    ("median", "median")
]

# TODO change position to time stamp
DRIFTS = [
    ("Linear", DriftInformation(lambda x: (np.linspace(0, 500, x)), 1000, 16064)),
    ("Sine", DriftInformation(lambda x: (np.sin(np.linspace(0, 50, x)) * 500 + 500).reshape(-1, 1, 1), 5000, 10000))
]

DATASETS = [
    ("cnt", "data/bikesharing_hourly.csv", "2012-12-31", "dteday", 24, "1h"),
]


def get_train_pipeline(HORIZON, column, gan, inns, vae, scaler, calendar, stats, path="results"):
    pipeline = Pipeline(path)
    scaled = scaler(x=pipeline[column])
    in_data = FunctionModule(get_reshaping(column))(x=scaled)

    extractor = calendar(x=in_data)
    target = Sampler(sample_size=HORIZON)(x=in_data)
    stats = stats(x=target)
    # Mean of each sample is taken...

    sampled_cal = Sampler(sample_size=HORIZON)(x=extractor)
    sampled_stats = Sampler(sample_size=HORIZON)(x=stats)

    if vae is not None:
        # VAE should only run with scaling
        sampler_vae = Sampler(sample_size=HORIZON)(x=extractor)
        vae(input_calendar=sampler_vae, input_stats=stats, input_data=target, target=target)
    if gan is not None:
        gan(input_cal=sampled_cal, input_stats=sampled_stats, target=target)
        gan.is_fitted = True
    inn_steps = [
        (inn.name,
         inn(cal_input=sampled_cal, stats_input=sampled_stats, input_data=target,
             computation_mode=ComputationMode.Train))
        for
        inn in inns]
    return inn_steps, pipeline


# Aim of this pipeline is to generate data with statistics

def create_run_pipelines(column, split_date, data, HORIZON, freq, inns, drift, name=""):
    inn_wrappers = [INNWrapper(name + f"{500}", inn(5e-4), epochs=500, horizon=HORIZON,
                               stats_loss=stats_loss)
                    for name, inn, stats_loss in inns]
    scaler = SKLearnWrapper(StandardScaler())
    calendar_module = CalendarExtraction(country="BadenWurttemberg",
                                         features=[CalendarFeature.hour_sine, CalendarFeature.weekend,
                                                   CalendarFeature.month_sine])
    stats_module = StatisticExtraction(features=[StatisticFeature.mean], dim="horizon")
    inn_steps, pipeline = get_train_pipeline(HORIZON, column, None, inn_wrappers, None, scaler,
                                             calendar_module, stats_module)

    ##### Run pipeline
    train_data = data[:split_date]
    pipeline.train(train_data)

    ################# TEST Pipeine #########################
    data_vars = {
        f"Random_{name}": (
            ["time", "dims_rand"], np.random.multivariate_normal(
                [0 for _ in range(len(np.cov(list(inn_step.step.buffer.values())[0].values, rowvar=False)))],
                np.cov(list(inn_step.step.buffer.values())[0].values, rowvar=False), len(data[column])))
        for name, inn_step in inn_steps
    }
    data_vars.update({
        # TODO use here the drift parameters
        "statistics": np.cos(np.linspace(-np.pi, 5 * np.pi, 365 * 24 * 3)) * 50 + np.linspace(150, 250, 365 * 24 * 3),
        column: (["time"], data[column].values),

    })
    ################# TEST Pipeine #########################
    random_init = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": pd.date_range("2011-01-01 00:00:00", freq=freq, periods=len(data[column])),
        })

    pipeline2 = Pipeline("result_controllable_decoder")
    calendar = calendar_module(x=pipeline2[column])
    sampled_cal = Sampler(sample_size=HORIZON)(x=calendar)

    stats = pipeline2["statistics"]
    sampled_stats = Sampler(sample_size=HORIZON)(x=stats)
    generator_inns = [(inn.name, inn(cal_input=sampled_cal, stats_input=sampled_stats,
                                     input_data=pipeline2[f"Random_{inn.name}"],
                                     computation_mode=ComputationMode.Transform, use_inverse_transform=True))
                      for inn in inn_wrappers]

    generator_inns = list(
        map(lambda d: (d[0], FunctionModule(get_reshaping(horizon=HORIZON, name="reverse_" + d[0]))(
            x=scaler(x=d[1], use_inverse_transform=True, computation_mode=ComputationMode.Transform))),
            generator_inns))

    agg_data = []
    for aggregator_name, method in AGGREGATORS:
        agg_data.extend(
            [Merger(method=method, name=f"{aggregator_name}_{name}")(
                x=generator_inn, callbacks=[CSVCallback(method), LinePlotCallback(method)]) for name, generator_inn
                in generator_inns])

    pipeline2.test(random_init)

    for agg in agg_data:
        agg.step.buffer[agg.step.name].plot()
        inverse_stats = scaler.module.inverse_transform(stats.step.buffer["statistics"])
        xr.DataArray(inverse_stats.reshape((-1,)),
                     coords=[pd.date_range("2011-01-01 00:00:00", freq=freq, periods=len(data[column]))],
                     dims=["time"]).plot()
        tikzplotlib.save(f"result_controllable_decoder/statsAndGenerated_{agg.step.name}_Drift{name}.tex")
        plt.savefig(f"result_controllable_decoder/statsAndGenerated_{agg.step.name}_Drift{name}.png")
        plt.close()

    print("Finished")


if __name__ == "__main__":
    for column, path, split_date, date_col, HORIZON, freq in DATASETS:
        inns = [("inn_bottleneck16_15_0_single_value",
                 functools.partial(INN, horizon=HORIZON, cond_features=COND_FEATURES, n_layers_cond=15,
                                   subnet=functools.partial(subnet, bottleneck_size=16)), False)]

        data = pd.read_csv(path, index_col=date_col, parse_dates=[date_col],
                           infer_datetime_format=True)

        if column == "cnt":
            data.index = data.index + pd.Series(map(lambda v: pd.Timedelta(f"{v}h"), data['hr'].values))
            data = data.resample("1h").interpolate()
        for name, drift in DRIFTS:
            create_run_pipelines(column, split_date, data, HORIZON, freq, inns, drift=drift, name=name)
        print("finished")
