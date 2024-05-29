import cftime
import numpy as np
import pytest
from typing import Callable
import xarray as xr

from mlde_utils.transforms import PercentToPropT


def test_transform(dataset: xr.Dataset):
    xfm = PercentToPropT(["target_relhum"])
    xfmed_ds = xfm.transform(dataset.copy())

    assert np.all(xfmed_ds["target_relhum"] == dataset["target_relhum"] / 100)


def test_invert(dataset: xr.Dataset):
    xfm = PercentToPropT(["target_relhum"])
    xfmed_ds = xfm.invert(xfm.transform(dataset.copy()))

    assert np.all(xfmed_ds["target_relhum"] == dataset["target_relhum"])


def build_time_dim(start_year: int = 1980, time_len: int = 10):
    time = xr.Variable(
        ["time"],
        xr.cftime_range(
            cftime.Datetime360Day(start_year, 12, 1, 12, 0, 0, 0, has_year_zero=True),
            periods=time_len,
            freq="D",
        ),
    )
    time_bnds_values = xr.cftime_range(
        cftime.Datetime360Day(start_year, 12, 1, 0, 0, 0, 0, has_year_zero=True),
        periods=len(time) + 1,
        freq="D",
    ).values
    time_bnds_pairs = np.concatenate(
        [time_bnds_values[:-1, np.newaxis], time_bnds_values[1:, np.newaxis]], axis=1
    )

    time_bnds = xr.Variable(["time", "bnds"], time_bnds_pairs, attrs={})

    return time, time_bnds


@pytest.fixture
def grid_latitude():
    return xr.Variable(["grid_latitude"], np.linspace(-3, 3, 13), attrs={})


@pytest.fixture
def grid_longitude():
    return xr.Variable(["grid_longitude"], np.linspace(-4, 4, 17), attrs={})


@pytest.fixture
def dataset_factory(grid_latitude, grid_longitude) -> Callable[[int, int], xr.Dataset]:
    """Create a factory function for creating dummy xarray Datasets that look like the training data."""

    def _dataset_factory(start_year: int = 1980, time_len: int = 10) -> xr.Dataset:
        ensemble_member = xr.Variable(
            ["ensemble_member"], np.array([f"{i:02}" for i in range(3)])
        )

        time, time_bnds = build_time_dim(start_year=start_year, time_len=time_len)

        coords = {
            "ensemble_member": ensemble_member,
            "time": time,
            "grid_latitude": grid_latitude,
            "grid_longitude": grid_longitude,
        }

        data_vars = {
            "linpr": xr.Variable(
                ["ensemble_member", "time", "grid_latitude", "grid_longitude"],
                np.random.rand(
                    len(ensemble_member),
                    len(time),
                    len(grid_latitude),
                    len(grid_longitude),
                ),
            ),
            "target_pr": xr.Variable(
                ["ensemble_member", "time", "grid_latitude", "grid_longitude"],
                np.random.rand(
                    len(ensemble_member),
                    len(time),
                    len(grid_latitude),
                    len(grid_longitude),
                ),
            ),
            "target_relhum": xr.Variable(
                ["ensemble_member", "time", "grid_latitude", "grid_longitude"],
                np.random.rand(
                    len(ensemble_member),
                    len(time),
                    len(grid_latitude),
                    len(grid_longitude),
                )
                * 100,
            ),
            "time_bnds": time_bnds,
        }

        ds = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
        )

        return ds

    return _dataset_factory


@pytest.fixture
def dataset(dataset_factory) -> xr.Dataset:
    """Create a dummy xarray Dataset representing a split of a set of data for training and sampling."""
    return dataset_factory()
