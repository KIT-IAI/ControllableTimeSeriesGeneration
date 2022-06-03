from pywatts.utils._xarray_time_series_utils import numpy_to_xarray


def get_reshaping(name="StandardScaler", horizon=None):
    def reshaping(x):
        if horizon is None:
            data = numpy_to_xarray(x.values.reshape((-1)), x)
        else:
            data = numpy_to_xarray(x.values.reshape((-1, horizon)), x)
        return data

    return reshaping


def get_repeat(horizon):
    def repeat(x):
        data = numpy_to_xarray(x.values.repeat(horizon, axis=-1).reshape((-1, horizon, 1)), x)
        return data

    return repeat
