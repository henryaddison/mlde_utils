import glob
import os
from pathlib import Path
from typing import List
import yaml

import cartopy.crs as ccrs
import cftime

cp_model_rotated_pole = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
platecarree = ccrs.PlateCarree()

DEFAULT_ENSEMBLE_MEMBER = "01"

TIME_PERIODS = {
    "historic": (
        cftime.Datetime360Day(1980, 12, 1, 12, 0, 0, 0, has_year_zero=True),
        cftime.Datetime360Day(2000, 11, 30, 12, 0, 0, 0, has_year_zero=True),
    ),
    "present": (
        cftime.Datetime360Day(2020, 12, 1, 12, 0, 0, 0, has_year_zero=True),
        cftime.Datetime360Day(2040, 11, 30, 12, 0, 0, 0, has_year_zero=True),
    ),
    "future": (
        cftime.Datetime360Day(2060, 12, 1, 12, 0, 0, 0, has_year_zero=True),
        cftime.Datetime360Day(2080, 11, 30, 12, 0, 0, 0, has_year_zero=True),
    ),
}


class VariableMetadata:
    def __init__(
        self,
        base_dir,
        variable,
        frequency,
        domain,
        resolution,
        ensemble_member,
        scenario="rcp85",
    ):
        self.base_dir = base_dir
        self.variable = variable
        self.frequency = frequency
        self.resolution = resolution
        self.domain = domain
        self.scenario = scenario
        self.ensemble_member = ensemble_member

        if self.resolution.startswith("2.2km"):
            self.collection = "land-cpm"
        elif self.resolution.startswith("60km"):
            self.collection = "land-gcm"

    def __str__(self):
        return "VariableMetadata: " + str(self.__dict__)

    def filename_prefix(self):
        return "_".join(
            [
                self.variable,
                self.scenario,
                self.collection,
                self.domain,
                self.resolution,
                self.ensemble_member,
                self.frequency,
            ]
        )

    def filename(self, year):
        return f"{self.filename_prefix()}_{year-1}1201-{year}1130.nc"

    def subdir(self):
        return os.path.join(
            self.domain,
            self.resolution,
            self.scenario,
            self.ensemble_member,
            self.variable,
            self.frequency,
        )

    def dirpath(self):
        return os.path.join(self.base_dir, self.subdir())

    def filepath(self, year):
        return os.path.join(self.dirpath(), self.filename(year))

    def filepath_prefix(self):
        return os.path.join(self.dirpath(), self.filename_prefix())

    def existing_filepaths(self):
        return glob.glob(f"{self.filepath_prefix()}_*.nc")

    def years(self):
        filenames = [
            os.path.basename(filepath) for filepath in self.existing_filepaths()
        ]
        return list([int(filename[-20:-16]) for filename in filenames])


class DatasetMetadata:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"DatasetMetadata({self.path()})"

    def path(self):
        return Path(os.getenv("DERIVED_DATA"), "moose", "nc-datasets", self.name)

    def splits(self):
        return map(
            lambda f: os.path.splitext(f)[0],
            glob.glob("*.nc", root_dir=str(self.path())),
        )

    def split_path(self, split):
        return self.path() / f"{split}.nc"

    def config_path(self) -> Path:
        return self.path() / "ds-config.yml"

    def config(self) -> dict:
        with open(self.config_path(), "r") as f:
            return yaml.safe_load(f)

    def ensemble_members(self) -> List[str]:
        return self.config()["ensemble_members"]


def workdir_path(fq_run_id: str) -> Path:
    return Path(os.getenv("DERIVED_DATA"), "workdirs", fq_run_id)


def samples_path(
    workdir: str,
    checkpoint: str,
    input_xfm: str,
    dataset: str,
    split: str,
    ensemble_member: str,
) -> Path:
    return Path(
        workdir, "samples", checkpoint, dataset, input_xfm, split, ensemble_member
    )


def samples_glob(samples_path: Path) -> list[Path]:
    return samples_path.glob("predictions-*.nc")


def dataset_path(dataset: str, base_dir: str = None) -> Path:
    if base_dir is None:
        base_dir = os.getenv("DERIVED_DATA")
    return Path(base_dir, "moose", "nc-datasets", dataset)


def dataset_split_path(dataset: str, split: str, base_dir: str = None) -> Path:
    return dataset_path(dataset, base_dir=base_dir) / f"{split}.nc"


def dataset_config_path(dataset: str, base_dir: str = None) -> Path:
    return dataset_path(dataset, base_dir=base_dir) / "ds-config.yml"


def dataset_config(dataset: str, base_dir: str = None) -> dict:
    with open(dataset_config_path(dataset, base_dir=base_dir), "r") as f:
        return yaml.safe_load(f)
