import glob
import os
from pathlib import Path
from typing import List
import yaml

import cartopy.crs as ccrs
import cftime

WORKDIRS_PATH = Path(os.getenv("WORKDIRS_PATH"))
DERIVED_DATA = Path(os.getenv("DERIVED_DATA"))

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
        scenario,
        collection,
    ):
        self.base_dir = base_dir
        self.variable = variable
        self.frequency = frequency
        self.resolution = resolution
        self.domain = domain
        self.scenario = scenario
        self.ensemble_member = ensemble_member
        self.collection = collection

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
            self.collection,
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
    def __init__(self, name, base_dir=DERIVED_DATA):
        self.name = name
        self.base_dir = base_dir

    def __str__(self):
        return f"DatasetMetadata({self.path()})"

    def path(self):
        return Path(self.base_dir, "moose", "nc-datasets", self.name)

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


class EmulatorOutputMetadata:
    def __init__(self, fq_run_id: str, base_dir: Path = WORKDIRS_PATH):
        self.base_dir = base_dir
        self.fq_run_id = fq_run_id

    def workdir_path(self) -> Path:
        """
        Returns the path to the emulator output for the given run ID.
        """
        return Path(self.base_dir, self.fq_run_id)

    def __str__(self) -> str:
        return f"EmulatorOutputMetadata(path={self.workdir_path()})"

    def samples_path(
        self,
        checkpoint: str,
        input_xfm: str,
        dataset: str,
        split: str,
        ensemble_member: str,
    ) -> Path:
        """
        Returns the path to the samples for the given parameters.
        """
        return (
            self.workdir_path()
            / "samples"
            / checkpoint
            / dataset
            / input_xfm
            / split
            / ensemble_member
        )

    def samples_glob(self, *args, **kwargs) -> list[Path]:
        """
        Returns a list of prediction files for the given parameters
        """
        return self.samples_path(*args, **kwargs).glob("predictions-*.nc")
