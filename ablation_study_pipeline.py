import functools

import numpy as np
import pandas as pd
import tensorflow.keras as keras
import torch
import xarray as xr
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.modules import CalendarExtraction, CalendarFeature, StatisticExtraction, StatisticFeature, Slicer
from pywatts.modules import ClockShift, Sampler, FunctionModule, SKLearnWrapper
from pywatts.summaries.tsne_visualisation import TSNESummary
from pywatts.summaries.discriminative_score import DiscriminativeScore
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray
from sklearn.preprocessing import StandardScaler
from torch import nn

from generative_models.INN import INN, INNWrapper
from pywatts.modules.postprocessing.merger import Merger

from pywatts.summaries.train_synthetic_test_real import TrainSyntheticTestReal


def _get_model(horizon):
    input = keras.layers.Input(((horizon)))
    states = keras.layers.Dense(5, activation="tanh")(input)
    out = keras.layers.Dense(1, activation="sigmoid")(states)
    model = keras.Model(input, out)
    model.compile(optimizer="Adam",
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


COND_FEATURES = 3

CONDITIONS = ["CalendarExtraction", "statistics"]

AGGREGATORS = [
    ("first", 0),
    ("mean", "mean"),
    ("median", "median")
]
SCALING = False


def get_reshaping(name="StandardScaler", horizon=None):
    def reshaping(x):
        if horizon is None:
            data = numpy_to_xarray(x.values.reshape((-1)), x)
        else:
            data = numpy_to_xarray(x.values.reshape((-1, horizon)), x)
        return data

    return reshaping


def subnet(ch_in, ch_out, bottleneck_size=32, activation=torch.nn.Tanh):
    return nn.Sequential(
        nn.Linear(ch_in, bottleneck_size),
        activation(),
        nn.Linear(bottleneck_size, ch_out))


def get_lstm_model(pred_horizon=1):
    input = keras.layers.Input((HORIZON))
    state = keras.layers.Dense(10, activation="relu")(input)
    output = keras.layers.Dense(pred_horizon, activation="linear")(state)

    model = keras.Model(input, output)
    model.compile(loss="mse")
    return model


def get_repeat(horizon):
    def repeat(x):
        data = numpy_to_xarray(x.values.repeat(horizon, axis=-1).reshape((-1, horizon, 1)), x)
        return data

    return repeat


DATASETS = [

    ("cnt", "data/bikesharing_hourly.csv", "2012-01-01", "dteday", 24, "1h", True, "2011-01-01", 168),
    # ("The_Hitchhiker's_Guide_to_the_Galaxy_(video_game)_en.wikipedia.org_all-access_all-agents",
    # "data/HitchhikersWiki.csv",
    # "2016-07-01", "Date", 7, "1d", True,
    # "2015-07-01", 7),
    #   ("MT_158", "data/uci_electricity_hourly.csv", "2014-01-01", "date", 24, "1h", True, "2012-12-31", 168),
]


def create_run_pipelines(column, split_date, data, HORIZON, freq, SCALING, start_date, inns=[], trajectory=True,
                         mase_lag=168):
    for i in range(3):
        INN_EPOCHS = 200
        inn_wrappers_none = [INNWrapper(name + "None", inn(5e-4, cond_features=0), epochs=INN_EPOCHS, horizon=HORIZON,
                                        stats_loss=stats_loss) for name, inn, stats_loss in inns]
        #
        inn_wrappers_stats = [INNWrapper(name + "Stats", inn(5e-4, cond_features=1), epochs=INN_EPOCHS, horizon=HORIZON,
                                         stats_loss=stats_loss) for name, inn, stats_loss in inns]

        inn_wrappers_cal = [INNWrapper(name + "Cal", inn(5e-4, cond_features=3), epochs=INN_EPOCHS, horizon=HORIZON,
                                       stats_loss=stats_loss) for name, inn, stats_loss in inns]

        inn_wrappers = [INNWrapper(name + "Condition", inn(5e-4, cond_features=4), epochs=INN_EPOCHS, horizon=HORIZON,
                                   stats_loss=stats_loss) for name, inn, stats_loss in inns]
        calendar_module = CalendarExtraction(country="BadenWurttemberg",
                                             features=[CalendarFeature.hour_sine, CalendarFeature.weekend,
                                                       CalendarFeature.month_sine
                                                       ])
        stats_module = StatisticExtraction(features=[StatisticFeature.mean], dim="horizon")

        scaler = SKLearnWrapper(StandardScaler())

        pipeline = Pipeline("results_new/train_ablation_new")
        scaled = scaler(x=pipeline[column])
        in_data = FunctionModule(get_reshaping(column))(x=scaled)
        target = Sampler(sample_size=HORIZON)(x=in_data)
        stats = stats_module(x=target)
        cal = calendar_module(x=in_data)

        inn_steps = []
        sampled_stats = Sampler(sample_size=HORIZON)(x=stats)
        sampled_cal = Sampler(sample_size=HORIZON)(x=cal)

        sampled_cal = Slicer(HORIZON)(x=sampled_cal)
        target = Slicer(HORIZON)(x=target)
        sampled_stats = Slicer(HORIZON)(x=sampled_stats)

        inn_steps.extend([(inn.name, inn(input_data=target, computation_mode=ComputationMode.Train, )) for inn in inn_wrappers_none])
        inn_steps.extend([(inn.name,
                           inn(stats_input=sampled_stats, input_data=target, computation_mode=ComputationMode.Train)) for inn
                          in inn_wrappers_stats])
        inn_steps.extend([(inn.name, inn(cal_input=sampled_cal, stats_input=sampled_stats, input_data=target,
                                         computation_mode=ComputationMode.Train))
                          for inn in inn_wrappers])
        inn_steps.extend(
            [(inn.name, inn(cal_input=sampled_cal, input_data=target, computation_mode=ComputationMode.Train)) for inn
             in inn_wrappers_cal])

        ##### Run pipeline
        train_data = data[:split_date]

        pipeline.train(train_data)
        data_vars = {
            f"Random_{name}": (
                ["time", "dims_rand"], np.random.multivariate_normal(
                    [0 for _ in range(len(np.cov(list(inn_step.step.buffer.values())[0].values, rowvar=False))
                                      )],
                    np.cov(list(inn_step.step.buffer.values())[0].values, rowvar=False), len(data[column]) - 24))
            for name, inn_step in inn_steps
        }
        data_vars.update({
            "statistics": (
                ["time", "dims_stats"],
                np.random.normal(0 if SCALING else data[column].median(), 0, (len(data[column]) - 24, 1))),
            column: (["time2"], data[column].values),
            "Random": (
                ["time", "dims_rand"], np.random.normal(0, 1, size=(len(data[column]) - 24, HORIZON))),
        }),
        ################# TEST Pipeine #########################
        random_init = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": pd.date_range(start_date, freq=freq, periods=len(data[column]) - 24),
                "time2": pd.date_range(start_date, freq=freq, periods=len(data[column])),
            })

        tstr = TrainSyntheticTestReal(repetitions=5, fit_kwargs={"epochs": 100, "validation_split": 0.2,
                                                                 "callbacks": [keras.callbacks.EarlyStopping(
                                                                     monitor='val_loss', min_delta=0, patience=5,
                                                                     verbose=0,
                                                                     mode='auto', baseline=None,
                                                                     restore_best_weights=True)]})

        tstr_raw = TrainSyntheticTestReal(repetitions=5, name="TSTRRaw",
                                          fit_kwargs={"epochs": 100, "validation_split": 0.2,
                                                      "callbacks": [keras.callbacks.EarlyStopping(
                                                          monitor='val_loss', min_delta=0, patience=5, verbose=0,
                                                          mode='auto', baseline=None, restore_best_weights=True)]})

        pipeline_decoder = Pipeline("results_new/decoder_ablation_wiki")
        scaled = scaler(x=pipeline_decoder[column], computation_mode=ComputationMode.Transform)
        in_data = FunctionModule(get_reshaping(column))(x=scaled)
        target = Sampler(sample_size=HORIZON)(x=in_data)
        stats = stats_module(x=target)

        generator_inns = []
        cal = calendar_module(x=in_data)
        if trajectory:
            sampled_stats = Sampler(sample_size=HORIZON)(x=stats)
        else:
            sampled_stats = FunctionModule(get_repeat(HORIZON))(x=stats)

        sampled_stats = Slicer(HORIZON)(x=sampled_stats)

        sampled_cal = Sampler(sample_size=HORIZON)(x=cal)
        sampled_cal = Slicer(HORIZON)(x=sampled_cal)

        generator_inns.extend([(inn.name, inn(input_data=pipeline_decoder[f"Random_{inn.name}"],
                                              computation_mode=ComputationMode.Transform, use_inverse_transform=True,
                                              )) for inn in
                               inn_wrappers_none])
        generator_inns.extend(
            [(inn.name, inn(stats_input=sampled_stats, input_data=pipeline_decoder[f"Random_{inn.name}"],
                            computation_mode=ComputationMode.Transform, use_inverse_transform=True)) for inn
             in inn_wrappers_stats])
        generator_inns.extend([(inn.name, inn(cal_input=sampled_cal, stats_input=sampled_stats,
                                              input_data=pipeline_decoder[f"Random_{inn.name}"],
                                              computation_mode=ComputationMode.Transform, use_inverse_transform=True,
                                              ))
                               for inn in inn_wrappers])
        generator_inns.extend(
            [(inn.name, inn(cal_input=sampled_cal, input_data=pipeline_decoder[f"Random_{inn.name}"],
                            computation_mode=ComputationMode.Transform, use_inverse_transform=True,
                            )) for inn
             in inn_wrappers_cal])

        generator_inns = list(
            map(lambda d: (d[0], FunctionModule(get_reshaping(horizon=HORIZON, name="reverse_" + d[0]))(
                x=scaler(x=d[1], use_inverse_transform=True, computation_mode=ComputationMode.Transform))),
                generator_inns))
        agg_data = []
        for aggregator_name, method in AGGREGATORS:
            agg_data.extend(
                [Merger(method=method, name=f"{aggregator_name}_{name}")(x=generator_inn) for name, generator_inn
                 in generator_inns])

        input_component = {}
        for generated_data in agg_data:
            generated_data = Slicer(HORIZON)(x=generated_data)
            input_data = Sampler(HORIZON)(x=generated_data)
            input_data = Slicer(HORIZON)(x=input_data)
            input_component[generated_data.step.name] = input_data
        gt = ClockShift(lag=-HORIZON)(x=pipeline_decoder[column])
        gt = Sampler(HORIZON)(x=gt)
        gt = Slicer(HORIZON)(x=gt)

        TSNESummary(name=column, max_points=1000, tsne_params={"n_components": 2, "perplexity": 40, "n_iter": 300})(
            gt=gt, **input_component)
        TSNESummary(name=column, max_points=1000, all_in_one_plot=True,
                    tsne_params={"n_components": 2, "perplexity": 40, "n_iter": 300})(gt=gt, **input_component)

        disc_input = {name: generator_inn for name, generator_inn in generator_inns}

        DiscriminativeScore(repetitions=5, test_size=0.3,
                            fit_kwargs={"epochs": 10, "validation_split": 0.2,
                                        "callbacks": [keras.callbacks.EarlyStopping(
                                            monitor='val_loss', min_delta=0, patience=5, verbose=0,
                                            mode='auto', baseline=None, restore_best_weights=True)]},
                            get_model=_get_model,
                            name="DiscRaw")(gt=gt, **disc_input)
        DiscriminativeScore(repetitions=5, test_size=0.3,
                            fit_kwargs={"epochs": 10, "validation_split": 0.2,
                                        "callbacks": [keras.callbacks.EarlyStopping(
                                            monitor='val_loss', min_delta=0, patience=5, verbose=0,
                                            mode='auto', baseline=None, restore_best_weights=True)]},
                            get_model=_get_model,
                            name="DiscAGG")(gt=gt, **input_component)

        tstr(real=gt, **input_component)
        tstr_raw(real=gt, **disc_input)

        result, summary = pipeline_decoder.test(random_init)

        print("Finished")


if __name__ == "__main__":
    for column, path, split_date, date_col, HORIZON, freq, SCALING, start_date, mase_lag in DATASETS:
        inns = [("inn_bottleneck16_15_0_single_value",
                 functools.partial(INN, horizon=HORIZON, cond_features=COND_FEATURES, n_layers_cond=15,
                                   subnet=functools.partial(subnet, bottleneck_size=16)), False)]
        data = pd.read_csv(path, index_col=date_col, parse_dates=[date_col],
                           infer_datetime_format=True)

        if column == "cnt":
            data.index = data.index + pd.Series(map(lambda v: pd.Timedelta(f"{v}h"), data['hr'].values))
            data = data.resample("1h").interpolate()
        create_run_pipelines(column, split_date, data, HORIZON, freq, SCALING, start_date, inns=inns, trajectory=False,
                             mase_lag=mase_lag)
        print("finished")
