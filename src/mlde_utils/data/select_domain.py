import logging
import math

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
            *lon_lat, src_crs=platecarree
        )
        for domain_name, lon_lat in DOMAIN_CENTRES_LON_LAT.items()
    }

    def __init__(self, subdomain, grid="cpm", size=64) -> None:
        self.subdomain = subdomain
        self.grid = grid
        self.size = size

    def run(self, ds):
        logger.info(f"Selecting subdomain {self.subdomain}")
        if self.grid == "cpm":
            centre_xy = self.DOMAIN_CENTRES_RP_LONG_LAT[self.subdomain]
            query = dict(
                grid_longitude=360.0 + centre_xy[0],
                grid_latitude=centre_xy[1],
            )
        elif self.grid == "gcm":
            centre_xy = self.DOMAIN_CENTRES_LON_LAT[self.subdomain]
            query = dict(
                longitude=centre_xy[0],
                latitude=centre_xy[1],
            )
        else:
            raise ValueError(f"Unknown grid type: {self.grid}")

        centre_ds = ds.sel(query, method="nearest")
        if self.grid == "cpm":
            centre_long_idx = np.where(
                ds.grid_longitude.values == centre_ds.grid_longitude.values
            )[0].item()
            centre_lat_idx = np.where(
                ds.grid_latitude.values == centre_ds.grid_latitude.values
            )[0].item()
        elif self.grid == "gcm":
            centre_long_idx = np.where(
                ds.longitude.values == centre_ds.longitude.values
            )[0].item()
            centre_lat_idx = np.where(ds.latitude.values == centre_ds.latitude.values)[
                0
            ].item()
        else:
            raise ValueError(f"Unknown grid type: {self.grid}")

        radius = self.size - 1
        left_length = math.floor(radius / 2.0)
        right_length = math.ceil(radius / 2.0)
        down_length = math.floor(radius / 2.0)
        up_length = math.ceil(radius / 2.0)

        if self.grid == "cpm":
            ds = ds.sel(
                grid_longitude=slice(
                    ds.grid_longitude[centre_long_idx - left_length].values,
                    ds.grid_longitude[centre_long_idx + right_length].values,
                ),
                grid_latitude=slice(
                    ds.grid_latitude[centre_lat_idx - down_length].values,
                    ds.grid_latitude[centre_lat_idx + up_length].values,
                ),
            )
        elif self.grid == "gcm":
            ds = ds.sel(
                longitude=slice(
                    ds.longitude[centre_long_idx - left_length].values,
                    ds.longitude[centre_long_idx + right_length].values,
                ),
                latitude=slice(
                    ds.latitude[centre_lat_idx - down_length].values,
                    ds.latitude[centre_lat_idx + up_length].values,
                ),
            )
        else:
            raise ValueError(f"Unknown grid type: {self.grid}")

        return ds
