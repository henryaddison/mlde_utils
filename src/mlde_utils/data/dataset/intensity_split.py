import datetime
import logging

import numpy as np

from ml_downscaling_emulator.data.dataset.random_split import RandomSplit

logger = logging.getLogger(__name__)


class IntensitySplit:
    def __init__(self, time_encoding, val_prop=0.2, test_prop=0.1) -> None:
        self.val_prop = val_prop
        self.test_prop = test_prop
        self.time_encoding = time_encoding

    def run(self, combined_dataset):
        sorted_time = np.flipud(
            combined_dataset.isel(ensemble_member=0)
            .sum(dim=["grid_latitude", "grid_longitude"])
            .sortby("target_pr")
            .time.values.copy()
        )

        test_size = int(len(sorted_time) * self.test_prop)
        val_size = int(len(sorted_time) * self.val_prop)

        test_times = set()
        val_times = set()

        working_set = test_times

        for timestamp in sorted_time:
            lag_duration = 5
            event = set(
                np.arange(
                    timestamp - datetime.timedelta(days=lag_duration),
                    timestamp + datetime.timedelta(days=lag_duration + 1),
                    datetime.timedelta(days=1),
                    dtype=type(timestamp),
                )
            )
            unseen_event_days = event - test_times - val_times
            working_set.update(unseen_event_days)
            if len(test_times) >= test_size:
                working_set = val_times
            if len(val_times) >= val_size:
                break

        train_times = set(sorted_time) - test_times - val_times

        print(f"train size: {len(train_times)}")
        print(f"val size: {len(val_times)}")
        print(f"test size: {len(test_times)}")
        print(f"all times: {len(sorted_time)}")

        extreme_test_set = combined_dataset.where(
            combined_dataset.time.isin(list(test_times)) == True,
            drop=True,  # noqa: E712
        )
        extreme_val_set = combined_dataset.where(
            combined_dataset.time.isin(list(val_times)) == True, drop=True  # noqa: E712
        )
        extreme_train_set = combined_dataset.where(
            combined_dataset.time.isin(list(train_times)) == True,
            drop=True,  # noqa: E712
        )

        splits = RandomSplit(
            time_encoding=self.time_encoding,
            val_prop=self.val_prop,
            test_prop=self.test_prop,
        ).run(extreme_train_set)
        splits.update(
            {"extreme_val": extreme_val_set, "extreme_test": extreme_test_set}
        )

        for ds in splits.values():
            # https://github.com/pydata/xarray/issues/2436 - time dim encoding lost when opened using open_mfdataset
            ds.time.encoding.update(self.time_encoding)

        return splits
