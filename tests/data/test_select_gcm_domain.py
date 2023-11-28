from importlib.resources import files
import pytest
import xarray as xr

from mlde_utils.data.select_gcm_domain import SelectGCMDomain
from mlde_utils.data.shift_lon_break import ShiftLonBreak


def test_select_bham64_domain(global_dataset):
    bham_gcm_ds = SelectGCMDomain(subdomain="2.2km-coarsened-4x_bham-64").run(
        global_dataset
    )

    assert bham_gcm_ds["precipitation_flux"].size == 9 * 9
    assert bham_gcm_ds["longitude"].size == 9
    assert bham_gcm_ds["latitude"].size == 9


@pytest.fixture
def global_dataset():
    filepath = files("mlde_utils.data").joinpath(
        f"target_grids/60km/global/pr/moose_grid.nc"
    )
    return ShiftLonBreak().run(xr.open_dataset(filepath))
