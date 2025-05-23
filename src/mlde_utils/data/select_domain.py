import logging
import math
import cf_xarray  # noqa: F401

import numpy as np

from mlde_utils import cp_model_rotated_pole, platecarree

logger = logging.getLogger(__name__)


class SelectDomain:

    DOMAIN_CENTRES_LON_LAT = {
        "london": (-0.118092, 51.509865),
        "birmingham": (-1.898575, 52.489471),
        "glasgow": (-4.25763000, 55.86515000),
        "aberdeen": (-2.09814000, 57.14369000),
        "scotland": (-4.20264580, 56.49067120),
        "dublin": (-6.267494, 53.344105),
    }

    DOMAIN_CENTRES_RP_LONG_LAT = {
        domain_name: cp_model_rotated_pole.transform_point(
            lon + 360, lat, src_crs=platecarree
        )
        for domain_name, (lon, lat) in DOMAIN_CENTRES_LON_LAT.items()
    }

    def __init__(self, subdomain, size=64) -> None:
        self.subdomain = subdomain
        self.size = size

    def run(self, ds):
        logger.info(f"Selecting subdomain {self.subdomain}")
        if "rotated_latitude_longitude" in ds.cf.grid_mapping_names:
            centre_xy = self.DOMAIN_CENTRES_RP_LONG_LAT[self.subdomain]
            query = dict(
                X=centre_xy[0],
                Y=centre_xy[1],
            )
        elif "latitude_longitude" in ds.cf.grid_mapping_names:
            centre_xy = self.DOMAIN_CENTRES_LON_LAT[self.subdomain]
            query = dict(
                X=centre_xy[0],
                Y=centre_xy[1],
            )
        else:
            raise ValueError(f"Unknown grid type: {self.grid}")

        centre_ds = ds.cf.sel(query, method="nearest")
        centre_long_idx = np.where(ds.cf.X.values == centre_ds.cf.X.values)[0].item()
        centre_lat_idx = np.where(ds.cf.Y.values == centre_ds.cf.Y.values)[0].item()

        radius = self.size - 1
        left_length = math.floor(radius / 2.0)
        right_length = math.ceil(radius / 2.0)
        down_length = math.floor(radius / 2.0)
        up_length = math.ceil(radius / 2.0)

        ds = ds.cf.sel(
            X=slice(
                ds.cf.X[centre_long_idx - left_length].values,
                ds.cf.X[centre_long_idx + right_length].values,
            ),
            Y=slice(
                ds.cf.Y[centre_lat_idx - down_length].values,
                ds.cf.Y[centre_lat_idx + up_length].values,
            ),
        )

        return ds
