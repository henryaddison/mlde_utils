import logging
import pickle

import numpy as np

logger = logging.getLogger()


def save_transform(xfm, path):
    with open(path, "wb") as f:
        logger.info(f"Storing transform: {path}")
        pickle.dump(xfm, f, pickle.HIGHEST_PROTOCOL)


def load_transform(path):
    with open(path, "rb") as f:
        logger.info(f"Using stored transform: {path}")
        xfm = pickle.load(f)

    return xfm


_XFMS = {}


def register_transform(cls=None, *, name=None):
    """A decorator for registering transform classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _XFMS:
            raise ValueError(f"Already registered transform with name: {local_name}")
        _XFMS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_transform(name):
    return _XFMS[name]


class CropT:
    def __init__(self, size):
        self.size = size

    def fit(self, target_ds, model_src_ds):
        return self

    def transform(self, ds):
        return ds.isel(
            grid_longitude=slice(0, self.size), grid_latitude=slice(0, self.size)
        )


@register_transform(name="stan")
class Standardize:
    def __init__(self, variables):
        self.variables = variables

    def fit(self, target_ds, model_src_ds):
        self.means = {var: target_ds[var].mean().values for var in self.variables}
        self.stds = {var: target_ds[var].std().values for var in self.variables}

        return self

    def transform(self, ds):
        for var in self.variables:
            ds[var] = (ds[var] - self.means[var]) / self.stds[var]

        return ds

    def invert(self, ds):
        for var in self.variables:
            ds[var] = (ds[var] * self.stds[var]) + self.means[var]

        return ds


@register_transform(name="pixelstan")
class PixelStandardize:
    def __init__(self, variables):
        self.variables = variables

    def fit(self, target_ds, model_src_ds):
        self.means = {
            var: target_ds[var].mean(dim=["time"]).values for var in self.variables
        }
        self.stds = {
            var: target_ds[var].std(dim=["time"]).values for var in self.variables
        }

        return self

    def transform(self, ds):
        for var in self.variables:
            ds[var] = (ds[var] - self.means[var]) / self.stds[var]

        return ds


@register_transform(name="noop")
class NoopT:
    def fit(self, target_ds, model_src_ds):
        return self

    def transform(self, ds):
        return ds

    def invert(self, ds):
        return ds


@register_transform(name="pixelmmsstan")
class PixelMatchModelSrcStandardize:
    def __init__(self, variables):
        self.variables = variables

    def fit(self, target_ds, model_src_ds):
        self.pixel_target_means = {
            variable: target_ds[variable].mean(dim=["time", "ensemble_member"])
            for variable in self.variables
        }
        self.pixel_target_stds = {
            variable: target_ds[variable].std(dim=["time", "ensemble_member"])
            for variable in self.variables
        }

        self.pixel_model_src_means = {
            variable: model_src_ds[variable].mean(dim=["time", "ensemble_member"])
            for variable in self.variables
        }
        self.pixel_model_src_stds = {
            variable: model_src_ds[variable].std(dim=["time", "ensemble_member"])
            for variable in self.variables
        }

        self.global_model_src_means = {
            var: model_src_ds[var].mean().values for var in self.variables
        }
        self.global_model_src_stds = {
            var: model_src_ds[var].std().values for var in self.variables
        }

        return self

    def transform(self, ds):
        for variable in self.variables:
            # first standardize each pixel
            da_pixel_stan = (
                ds[variable] - self.pixel_target_means[variable]
            ) / self.pixel_target_stds[variable]
            # then match mean and variance of each pixel to model source distribution
            da_pixel_like_model_src = (
                da_pixel_stan * self.pixel_model_src_stds[variable]
            ) + self.pixel_model_src_means[variable]
            # finally standardize globally (assuming a model source distribution)
            da_global_stan_like_model_src = (
                da_pixel_like_model_src - self.global_model_src_means[variable]
            ) / self.global_model_src_stds[variable]
            ds[variable] = da_global_stan_like_model_src
        return ds


@register_transform(name="mm")
class MinMax:
    def __init__(self, variables):
        self.variables = variables

    def fit(self, target_ds, model_src_ds):
        self.maxs = {var: target_ds[var].max().values for var in self.variables}
        self.mins = {var: target_ds[var].min().values for var in self.variables}

        return self

    def transform(self, ds):
        for var in self.variables:
            ds[var] = (ds[var] - self.mins[var]) / (self.maxs[var] - self.mins[var])

        return ds

    def invert(self, ds):
        for var in self.variables:
            ds[var] = ds[var] * (self.maxs[var] - self.mins[var]) + self.mins[var]

        return ds


@register_transform(name="ur")
class UnitRangeT:
    """WARNING: This transform assumes all values are positive"""

    def __init__(self, variables):
        self.variables = variables

    def fit(self, target_ds, model_src_ds):
        self.maxs = {var: target_ds[var].max().values for var in self.variables}

        return self

    def transform(self, ds):
        for var in self.variables:
            ds[var] = ds[var] / self.maxs[var]

        return ds

    def invert(self, ds):
        for var in self.variables:
            ds[var] = ds[var] * self.maxs[var]

        return ds


@register_transform(name="clip")
class ClipT:
    def __init__(self, variables):
        self.variables = variables

    def fit(self, target_ds, model_src_ds):
        return self

    def transform(self, ds):
        # target pr should be all non-negative so transform is no-op
        return ds

    def invert(self, ds):
        for var in self.variables:
            min = 0.0
            nclipped = (ds[var] < min).sum().item()
            logger.debug(f"Clipping {var} to {min}: {nclipped}")
            ds[var] = ds[var].clip(min=min)

        return ds


@register_transform(name="pc")
class PercentToPropT:
    def __init__(self, variables):
        self.variables = variables

    def fit(self, target_ds, _model_src_ds):
        return self

    def transform(self, ds):
        for var in self.variables:
            ds[var] = ds[var] / 100.0

        return ds

    def invert(self, ds):
        for var in self.variables:
            ds[var] = ds[var] * 100.0

        return ds


@register_transform(name="recen")
class RecentreT:
    def __init__(self, variables):
        self.variables = variables

    def fit(self, target_ds, model_src_ds):
        return self

    def transform(self, ds):
        for var in self.variables:
            ds[var] = ds[var] * 2.0 - 1.0

        return ds

    def invert(self, ds):
        for var in self.variables:
            ds[var] = (ds[var] + 1.0) / 2.0

        return ds


@register_transform(name="sqrt")
class SqrtT:
    def __init__(self, variables):
        self.variables = variables

    def fit(self, target_ds, model_src_ds):
        return self

    def transform(self, ds):
        for var in self.variables:
            ds[var] = ds[var] ** (0.5)

        return ds

    def invert(self, ds):
        for var in self.variables:
            ds[var] = ds[var] ** 2

        return ds


@register_transform(name="root")
class RootT:
    def __init__(self, variables, root_base):
        self.variables = variables
        self.root_base = root_base

    def fit(self, target_ds, model_src_ds):
        return self

    def transform(self, ds):
        for var in self.variables:
            ds[var] = np.power(ds[var], 1 / self.root_base)

        return ds

    def invert(self, ds):
        for var in self.variables:
            ds[var] = np.power(ds[var], self.root_base)

        return ds


@register_transform(name="rm")
class RawMomentT:
    def __init__(self, variables, root_base):
        self.variables = variables
        self.root_base = root_base

    def fit(self, target_ds, model_src_ds):
        self.raw_moments = {
            var: np.power(
                np.mean(np.power(target_ds[var], self.root_base)), 1 / self.root_base
            )
            for var in self.variables
        }

        return self

    def transform(self, ds):
        for var in self.variables:
            ds[var] = ds[var] / self.raw_moments[var]

        return ds

    def invert(self, ds):
        for var in self.variables:
            ds[var] = ds[var] * self.raw_moments[var]

        return ds


@register_transform(name="log")
class LogT:
    def __init__(self, variables):
        self.variables = variables

    def fit(self, target_ds, model_src_ds):
        return self

    def transform(self, ds):
        for var in self.variables:
            ds[var] = np.log1p(ds[var])

        return ds

    def invert(self, ds):
        for var in self.variables:
            ds[var] = np.expm1(ds[var])

        return ds


@register_transform(name="compose")
class ComposeT:
    def __init__(self, transforms):
        self.transforms = transforms

    def fit(self, target_ds, model_src_ds=None):
        for t in self.transforms:
            target_ds = t.fit(target_ds, model_src_ds).transform(target_ds)

        return self

    def transform(self, ds):
        for t in self.transforms:
            ds = t.transform(ds)

        return ds

    def invert(self, ds):
        for t in reversed(self.transforms):
            ds = t.invert(ds)

        return ds


def build_input_transform(variables, key="v1"):
    if key == "v1":
        return ComposeT([Standardize(variables), UnitRangeT(variables)])

    if key == "none":
        return NoopT()

    if key in ["standardize", "stan"]:
        return ComposeT([Standardize(variables)])

    if key == "stanur":
        return ComposeT(
            [
                Standardize(variables),
                UnitRangeT(variables),
            ]
        )

    if key == "stanurrecen":
        return ComposeT(
            [
                Standardize(variables),
                UnitRangeT(variables),
                RecentreT(variables),
            ]
        )

    if key == "pixelstan":
        return ComposeT([PixelStandardize(variables)])

    if key == "pixelmmsstan":
        return ComposeT(
            [
                PixelMatchModelSrcStandardize(variables),
            ]
        )

    if key == "pixelmmsstanur":
        return ComposeT(
            [
                PixelMatchModelSrcStandardize(variables),
                UnitRangeT(variables),
            ]
        )

    # otherwise assume the key is ; separated list of transform names
    xfms = map(lambda name: get_transform(name)(variables), key.split(";"))
    return ComposeT(list(xfms))


def build_target_transform(target_variables, keys):
    return ComposeT(
        [_build_target_transform(tvar, keys[tvar]) for tvar in target_variables]
    )


def _build_target_transform(target_variable, key):
    if key == "v1":
        return ComposeT(
            [
                SqrtT([target_variable]),
                ClipT([target_variable]),
                UnitRangeT([target_variable]),
            ]
        )

    if key == "none":
        return NoopT()

    if key == "sqrt":
        return ComposeT(
            [
                RootT([target_variable], 2),
                ClipT([target_variable]),
            ]
        )

    if key == "sqrtur":
        return ComposeT(
            [
                RootT([target_variable], 2),
                ClipT([target_variable]),
                UnitRangeT([target_variable]),
            ]
        )

    if key == "sqrturrecen":
        return ComposeT(
            [
                RootT([target_variable], 2),
                ClipT([target_variable]),
                UnitRangeT([target_variable]),
                RecentreT([target_variable]),
            ]
        )

    if key == "sqrtrm":
        return ComposeT(
            [
                RootT([target_variable], 2),
                RawMomentT([target_variable], 2),
                ClipT([target_variable]),
            ]
        )

    if key == "cbrt":
        return ComposeT(
            [
                RootT([target_variable], 3),
                ClipT([target_variable]),
            ]
        )

    if key == "cbrtur":
        return ComposeT(
            [
                RootT([target_variable], 3),
                ClipT([target_variable]),
                UnitRangeT([target_variable]),
            ]
        )

    if key == "qdrt":
        return ComposeT(
            [
                RootT([target_variable], 4),
                ClipT([target_variable]),
            ]
        )

    if key == "log":
        return ComposeT(
            [
                LogT([target_variable]),
                ClipT([target_variable]),
            ]
        )

    if key == "logurrecen":
        return ComposeT(
            [
                ClipT([target_variable]),
                LogT([target_variable]),
                UnitRangeT([target_variable]),
                RecentreT([target_variable]),
            ]
        )

    if key == "stanurrecen":
        return ComposeT(
            [
                Standardize([target_variable]),
                UnitRangeT([target_variable]),
                RecentreT([target_variable]),
            ]
        )

    if key == "stanmmrecen":
        return ComposeT(
            [
                Standardize([target_variable]),
                MinMax([target_variable]),
                RecentreT([target_variable]),
            ]
        )

    if key == "urrecen":
        return ComposeT(
            [
                UnitRangeT([target_variable]),
                RecentreT([target_variable]),
            ]
        )

    if key == "mmrecen":
        return ComposeT(
            [
                MinMax([target_variable]),
                RecentreT([target_variable]),
            ]
        )

    if key == "pcrecen":
        return ComposeT(
            [
                PercentToPropT([target_variable]),
                RecentreT([target_variable]),
            ]
        )

    if key == "recen":
        return ComposeT(
            [
                RecentreT([target_variable]),
            ]
        )

    # otherwise assume the key is ; separated list of transform names
    xfms = map(lambda name: get_transform(name)([target_variable]), key.split(";"))
    return ComposeT(list(xfms))
