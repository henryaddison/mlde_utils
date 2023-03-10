from datetime import timedelta
import logging
import os
import yaml

import xarray as xr

from ..transforms import (
    build_input_transform,
    build_target_transform,
    save_transform,
    load_transform,
)


def get_dataset(
    active_dataset_name,
    model_src_dataset_name,
    input_transform_key,
    target_transform_key,
    transform_dir,
    split,
    evaluation=False,
):
    """Create data loaders for given split.

    Args:
      active_dataset_name: Name of dataset from which to load data splits
      model_src_dataset_name: Name of dataset used to train the diffusion model (may be the same)
      transform_dir: Path to where transforms should be stored
      input_transform_key: Name of input transform pipeline to use
      target_transform_key: Name of target transform pipeline to use
      split: Split of the active dataset to load
      evaluation: If `True`, fix number of epochs to 1.

    Returns:
      data_loader, transform, target_transform.
    """

    transform, target_transform = _find_or_create_transforms(
        active_dataset_name,
        model_src_dataset_name,
        transform_dir,
        input_transform_key,
        target_transform_key,
        evaluation,
    )

    xr_data = load_raw_dataset_split(active_dataset_name, split)

    xr_data = transform.transform(xr_data)
    xr_data = target_transform.transform(xr_data)

    return xr_data, transform, target_transform


def load_raw_dataset_split(dataset_name, split):
    data_dirpath = os.path.join(
        os.getenv("DERIVED_DATA"), "moose", "nc-datasets", dataset_name
    )
    return xr.load_dataset(os.path.join(data_dirpath, f"{split}.nc"))


def get_variables(dataset_name):
    data_dirpath = os.path.join(
        os.getenv("DERIVED_DATA"), "moose", "nc-datasets", dataset_name
    )
    with open(os.path.join(data_dirpath, "ds-config.yml"), "r") as f:
        ds_config = yaml.safe_load(f)

    variables = [pred_meta["variable"] for pred_meta in ds_config["predictors"]]
    target_variables = ["target_pr"]

    return variables, target_variables


def _build_transform(
    variables,
    active_dataset_name,
    model_src_dataset_name,
    transform_key,
    builder,
):
    logging.info(f"Fitting transform")
    model_src_training_split = load_raw_dataset_split(model_src_dataset_name, "train")

    active_dataset_training_split = load_raw_dataset_split(active_dataset_name, "train")

    xfm = builder(variables, key=transform_key)

    xfm.fit(active_dataset_training_split, model_src_training_split)

    return xfm


def _find_or_create_transforms(
    active_dataset_name,
    model_src_dataset_name,
    transform_dir,
    input_transform_key,
    target_transform_key,
    evaluation,
):
    variables, target_variables = get_variables(model_src_dataset_name)

    if transform_dir is None:
        input_transform = _build_transform(
            variables,
            active_dataset_name,
            model_src_dataset_name,
            input_transform_key,
            build_input_transform,
        )

        if evaluation:
            raise RuntimeError("Target transform should only be fitted during training")
        target_transform = _build_transform(
            target_variables,
            active_dataset_name,
            model_src_dataset_name,
            target_transform_key,
            build_target_transform,
        )
    else:
        from flufl.lock import Lock

        dataset_transform_dir = os.path.join(
            transform_dir, active_dataset_name, input_transform_key
        )
        os.makedirs(dataset_transform_dir, exist_ok=True)
        input_transform_path = os.path.join(dataset_transform_dir, "input.pickle")
        target_transform_path = os.path.join(transform_dir, "target.pickle")

        lock_path = os.path.join(transform_dir, ".lock")
        lock = Lock(lock_path, lifetime=timedelta(hours=1))
        with lock:
            if os.path.exists(input_transform_path):
                input_transform = load_transform(input_transform_path)
            else:
                input_transform = _build_transform(
                    variables,
                    active_dataset_name,
                    model_src_dataset_name,
                    input_transform_key,
                    build_input_transform,
                )
                save_transform(input_transform, input_transform_path)

            if os.path.exists(target_transform_path):
                target_transform = load_transform(target_transform_path)
            else:
                if evaluation:
                    raise RuntimeError(
                        "Target transform should only be fitted during training"
                    )
                target_transform = _build_transform(
                    target_variables,
                    active_dataset_name,
                    model_src_dataset_name,
                    target_transform_key,
                    build_target_transform,
                )
                save_transform(target_transform, target_transform_path)

    return input_transform, target_transform
