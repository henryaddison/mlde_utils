import pandas as pd
import xarray as xr

from .. import dataset_split_path, workdir_path, samples_path, samples_glob


def si_to_mmday(ds, varname):
    print("hello world")
    # convert from kg m-2 s-1 (i.e. mm s-1) to mm day-1
    return (ds[varname] * 3600 * 24).assign_attrs({"units": "mm day-1"})


def open_samples_ds(
    run_name,
    checkpoint_id,
    dataset_name,
    input_xfm_key,
    split,
    num_samples,
    deterministic,
):
    samples_dir = samples_path(
        workdir=workdir_path(run_name),
        checkpoint=checkpoint_id,
        input_xfm=input_xfm_key,
        dataset=dataset_name,
        split=split,
    )
    samples_filepath_glob = samples_glob(samples_dir)

    sample_ds_list = [
        xr.open_dataset(sample_filepath)
        for sample_filepath in samples_filepath_glob[:num_samples]
    ]
    if len(sample_ds_list) == 0:
        raise RuntimeError(f"{samples_dir} has no sample files")
    if not deterministic:
        if len(sample_ds_list) < num_samples:
            raise RuntimeError(
                f"{samples_dir} does not have {num_samples} sample files"
            )

        ds = xr.concat(sample_ds_list, dim="sample_id")
        ds = ds.isel(sample_id=range(num_samples))
    else:
        ds = sample_ds_list[0]

    ds["pred_pr"] = si_to_mmday(ds, "pred_pr")

    return ds


def open_split_ds(dataset_name, split):
    ds = xr.open_dataset(dataset_split_path(dataset_name, split))
    ds["target_pr"] = si_to_mmday(ds, "target_pr")

    return ds


def open_merged_split_datasets(sample_runs, split):
    return xr.merge(
        [
            open_split_ds(dataset_name, split)
            for dataset_name in set(
                [sample_run["dataset"] for sample_run in sample_runs]
            )
        ],
        compat="override",
    )


def open_concat_sample_datasets(sample_runs, split, samples_per_run):
    samples_das = [
        open_samples_ds(
            run_name=sample_run["fq_model_id"],
            checkpoint_id=sample_run["checkpoint"],
            dataset_name=sample_run["dataset"],
            input_xfm_key=sample_run["input_xfm"],
            split=split,
            num_samples=samples_per_run,
            deterministic=sample_run["deterministic"],
        )["pred_pr"]
        for sample_run in sample_runs
    ]

    samples_ds = xr.concat(
        samples_das, pd.Index([sr["label"] for sr in sample_runs], name="model")
    )

    return samples_ds


def prep_eval_data(sample_runs, split, samples_per_run=3):
    samples_ds = open_concat_sample_datasets(sample_runs, split, samples_per_run)

    eval_ds = open_merged_split_datasets(sample_runs, split)

    return xr.merge([samples_ds, eval_ds], join="inner", compat="override")
