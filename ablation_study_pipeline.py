import argparse
import functools

import numpy as np
import pandas as pd
import tensorflow.keras as keras
import xarray as xr
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.modules import CalendarExtraction, CalendarFeature, StatisticExtraction, StatisticFeature, Slicer
from pywatts.modules import ClockShift, Sampler, FunctionModule, SKLearnWrapper
from pywatts.modules.postprocessing.merger import Merger
from pywatts.summaries.discriminative_score import DiscriminativeScore
from pywatts.summaries.train_synthetic_test_real import TrainSyntheticTestReal
from pywatts.summaries.tsne_visualisation import TSNESummary
from sklearn.preprocessing import StandardScaler

from generative_models.INN import INN, INNWrapper, subnet
from utils import get_reshaping, get_repeat


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

DATASETS = [

    ("cnt", "data/bikesharing_hourly.csv", "2012-01-01", "dteday", 24, "1h", True, "2011-01-01", 168),
    # ("The_Hitchhiker's_Guide_to_the_Galaxy_(video_game)_en.wikipedia.org_all-access_all-agents",
    # "data/HitchhikersWiki.csv",
    # "2016-07-01", "Date", 7, "1d", True,
    # "2015-07-01", 7),
    #   ("MT_158", "data/uci_electricity_hourly.csv", "2014-01-01", "date", 24, "1h", True, "2012-12-31", 168),
]


def create_run_pipelines(column, split_date, data, HORIZON, freq, start_date, inns=[]):
    INN_EPOCHS = 200
    # Create modules that are used multiple times
    inn_wrappers_none = [INNWrapper(name + "None", inn(5e-4, cond_features=0), epochs=INN_EPOCHS, horizon=HORIZON,
                                    stats_loss=stats_loss) for name, inn, stats_loss in inns]
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

    ### Training Pipeline
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

    # Add the INNs for training
    inn_steps.extend(
        [(inn.name, inn(input_data=target, computation_mode=ComputationMode.Train, )) for inn in inn_wrappers_none])
    inn_steps.extend([(inn.name,
                       inn(stats_input=sampled_stats, input_data=target, computation_mode=ComputationMode.Train))
                      for inn
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

    ### Evaluate Pipeline

    # Get evaluation data: Random Noise, original data for gt.
    data_vars = {
        f"Random_{name}": (
            ["time", "dims_rand"], np.random.multivariate_normal(
                [0 for _ in range(len(np.cov(list(inn_step.step.buffer.values())[0].values, rowvar=False))
                                  )],
                np.cov(list(inn_step.step.buffer.values())[0].values, rowvar=False), len(data[column]) - 24))
        for name, inn_step in inn_steps
    }
    data_vars.update({
        column: (["time2"], data[column].values),
    }),
    ################# TEST Pipeine #########################
    random_init = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": pd.date_range(start_date, freq=freq, periods=len(data[column]) - 24),
            "time2": pd.date_range(start_date, freq=freq, periods=len(data[column])),
        })

    pipeline_decoder = Pipeline(f"results_new/decoder_ablation{column}")
    scaled = scaler(x=pipeline_decoder[column], computation_mode=ComputationMode.Transform)
    in_data = FunctionModule(get_reshaping(column))(x=scaled)
    target = Sampler(sample_size=HORIZON)(x=in_data)

    # Extract condition features
    stats = stats_module(x=target)
    cal = calendar_module(x=in_data)
    sampled_stats = FunctionModule(get_repeat(HORIZON))(x=stats)
    sampled_stats = Slicer(HORIZON)(x=sampled_stats)
    sampled_cal = Sampler(sample_size=HORIZON)(x=cal)
    sampled_cal = Slicer(HORIZON)(x=sampled_cal)

    # Add the different inn versions
    generator_inns = []
    generator_inns.extend([(inn.name, inn(input_data=pipeline_decoder[f"Random_{inn.name}"],
                                          computation_mode=ComputationMode.Transform, use_inverse_transform=True,
                                          )) for inn in inn_wrappers_none])
    generator_inns.extend(
        [(inn.name, inn(stats_input=sampled_stats, input_data=pipeline_decoder[f"Random_{inn.name}"],
                        computation_mode=ComputationMode.Transform, use_inverse_transform=True)) for inn
         in inn_wrappers_stats])
    generator_inns.extend([(inn.name, inn(cal_input=sampled_cal, stats_input=sampled_stats,
                                          input_data=pipeline_decoder[f"Random_{inn.name}"],
                                          computation_mode=ComputationMode.Transform, use_inverse_transform=True,
                                          )) for inn in inn_wrappers])
    generator_inns.extend(
        [(inn.name, inn(cal_input=sampled_cal, input_data=pipeline_decoder[f"Random_{inn.name}"],
                        computation_mode=ComputationMode.Transform, use_inverse_transform=True,
                        )) for inn
         in inn_wrappers_cal])

    # Reshape the data
    generator_inns = list(
        map(lambda d: (d[0], FunctionModule(get_reshaping(horizon=HORIZON, name="reverse_" + d[0]))(
            x=scaler(x=d[1], use_inverse_transform=True, computation_mode=ComputationMode.Transform))),
            generator_inns))

    # Apply aggregation mechanism
    agg_data = []
    for aggregator_name, method in AGGREGATORS:
        agg_data.extend(
            [Merger(method=method, name=f"{aggregator_name}_{name}")(x=generator_inn) for name, generator_inn
             in generator_inns])

    # Get gt data
    gt = ClockShift(lag=-HORIZON)(x=pipeline_decoder[column])
    gt = Sampler(HORIZON)(x=gt)
    gt = Slicer(HORIZON)(x=gt)

    # Collect Predecessors for the input
    input_component = {}
    for generated_data in agg_data:
        sliced = Slicer(HORIZON)(x=generated_data)
        input_data = Sampler(HORIZON)(x=sliced)
        input_data = Slicer(HORIZON)(x=input_data)
        input_component[generated_data.step.name] = input_data
    raw_input = {name: generator_inn for name, generator_inn in generator_inns}

    # Evaluation Modules
    TSNESummary(name=column, max_points=1000, tsne_params={"n_components": 2, "perplexity": 40, "n_iter": 300})(
        gt=gt, **input_component)

    DiscriminativeScore(repetitions=5, test_size=0.3,
                        fit_kwargs={"epochs": 10, "validation_split": 0.2,
                                    "callbacks": [keras.callbacks.EarlyStopping(
                                        monitor='val_loss', min_delta=0, patience=5, verbose=0,
                                        mode='auto', baseline=None, restore_best_weights=True)]},
                        get_model=_get_model,
                        name="DiscRaw")(gt=gt, **raw_input)
    DiscriminativeScore(repetitions=5, test_size=0.3,
                        fit_kwargs={"epochs": 10, "validation_split": 0.2,
                                    "callbacks": [keras.callbacks.EarlyStopping(
                                        monitor='val_loss', min_delta=0, patience=5, verbose=0,
                                        mode='auto', baseline=None, restore_best_weights=True)]},
                        get_model=_get_model,
                        name="DiscAGG")(gt=gt, **input_component)
    TrainSyntheticTestReal(repetitions=5,
                           fit_kwargs={"epochs": 100, "validation_split": 0.2,
                                       "callbacks": [keras.callbacks.EarlyStopping(
                                           monitor='val_loss', min_delta=0, patience=5, verbose=0,
                                           mode='auto', baseline=None, restore_best_weights=True)]}
                           )(real=gt, **input_component)

    TrainSyntheticTestReal(repetitions=5, name="TSTRRaw",
                           fit_kwargs={"epochs": 100, "validation_split": 0.2,
                                       "callbacks": [keras.callbacks.EarlyStopping(
                                           monitor='val_loss', min_delta=0, patience=5, verbose=0,
                                           mode='auto', baseline=None, restore_best_weights=True)]}
                           )(real=gt, **raw_input)

    pipeline_decoder.test(random_init)

    print("Finished")


parser = argparse.ArgumentParser()
parser.add_argument("--data", help="The dataset that should be used", type=str, default="data/bike.csv")
parser.add_argument("--column", help="The target column in the dataset", type=str, default="cnt")
parser.add_argument("--index", help="The index column in the dataset", type=str, default="time")
parser.add_argument("--freq", help="The frequency of the dataset", type=str, default="1h")
parser.add_argument("--statistic", help="Periodic or increasing change", choices=["increase", "periodic"],
                    default="periodic")
parser.add_argument("--horizon", help="The horizon or sample size for the generative model", type=int, default=24)
parser.add_argument("--split_date", help="The date at which the train and test data are spliited",
                    type=str, default="2012-12-31")
parser.add_argument("--start_date", help="The time stamp of the first entry in the training data.",
                    type=str, default="2011-01-01")

if __name__ == "__main__":
    args = parser.parse_args()
    inns = [("inn_bottleneck16_15_0_single_value",
             functools.partial(INN, horizon=args.horizon, cond_features=COND_FEATURES, n_layers_cond=15,
                               subnet=subnet), False)]
    data = pd.read_csv(args.data, index_col=args.index, parse_dates=[args.index], infer_datetime_format=True)
    create_run_pipelines(args.column, args.split_date, data, args.horizon, args.freq, args.start_date, inns=inns)
    print("finished")
