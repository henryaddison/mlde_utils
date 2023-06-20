import pandas as pd
import xarray as xr

from .. import dataset_split_path, workdir_path, samples_path, samples_glob


def si_to_mmday(ds, varname):
    # convert from kg m-2 s-1 (i.e. mm s-1) to mm day-1
    return (ds[varname] * 3600 * 24).assign_attrs({"units": "mm day-1"})


def open_samples_ds(
    run_name,
    checkpoint_id,
    dataset_name,
    input_xfm_key,
    split,
    ensemble_members,
    num_samples,
    deterministic,
):

    per_em_datasets = []
    for ensemble_member in ensemble_members:
        samples_dir = samples_path(
            workdir=workdir_path(run_name),
            checkpoint=checkpoint_id,
            input_xfm=input_xfm_key,
            dataset=dataset_name,
            split=split,
            ensemble_member=ensemble_member,
        )
        sample_files_list = samples_glob(samples_dir)
        if len(sample_files_list) == 0:
            raise RuntimeError(f"{samples_dir} has no sample files")

        if deterministic:
            em_ds = xr.open_dataset(sample_files_list[0])
        else:
            sample_files_list = sample_files_list[:num_samples]
            if len(sample_files_list) < num_samples:
                raise RuntimeError(
                    f"{samples_dir} does not have {num_samples} sample files"
                )
            em_ds = xr.concat(
                [
                    xr.open_dataset(sample_filepath)
                    for sample_filepath in sample_files_list
                ],
                dim="sample_id",
            ).isel(sample_id=range(num_samples))

        per_em_datasets.append(em_ds)

    ds = xr.concat(per_em_datasets, dim="ensemble_member")
    ds["pred_pr"] = si_to_mmday(ds, "pred_pr")

    return ds


def open_split_ds(dataset_name, split, ensemble_members):
    ds = xr.open_dataset(dataset_split_path(dataset_name, split)).sel(
        ensemble_member=ensemble_members
    )
    ds["target_pr"] = si_to_mmday(ds, "target_pr")

    return ds


def open_merged_split_datasets(sample_runs, split, ensemble_members):
    return xr.merge(
        [
            open_split_ds(dataset_name, split, ensemble_members)
            for dataset_name in set(
                [sample_run["dataset"] for sample_run in sample_runs]
            )
        ],
        compat="override",
    )


def open_concat_sample_datasets(sample_runs, split, ensemble_members, samples_per_run):
    samples_das = [
        open_samples_ds(
            run_name=sample_run["fq_model_id"],
            checkpoint_id=sample_run["checkpoint"],
            dataset_name=sample_run["dataset"],
            input_xfm_key=sample_run["input_xfm"],
            split=split,
            ensemble_members=ensemble_members,
            num_samples=samples_per_run,
            deterministic=sample_run["deterministic"],
        )["pred_pr"]
        for sample_run in sample_runs
    ]

    samples_ds = xr.concat(
        samples_das, pd.Index([sr["label"] for sr in sample_runs], name="model")
    )

    return samples_ds


def prep_eval_data(sample_runs, split, ensemble_members, samples_per_run=3):
    samples_ds = open_concat_sample_datasets(
        sample_runs, split, ensemble_members, samples_per_run
    )

    eval_ds = open_merged_split_datasets(sample_runs, split, ensemble_members)

    return xr.merge([samples_ds, eval_ds], join="inner", compat="override")
