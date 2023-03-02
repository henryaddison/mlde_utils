import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .training.dataset import get_dataset, get_variables


class XRDataset(Dataset):
    def __init__(self, ds, variables, target_variables):
        self.ds = ds
        self.variables = variables
        self.target_variables = target_variables

    def __len__(self):
        return len(self.ds.time)

    def __getitem__(self, idx):
        subds = self.ds.isel(time=idx)

        cond = torch.tensor(
            np.stack([subds[var].values for var in self.variables], axis=0)
        ).float()

        x = torch.tensor(
            np.stack([subds[var].values for var in self.target_variables], axis=0)
        ).float()

        return cond, x


def get_dataloader(
    active_dataset_name,
    model_src_dataset_name,
    input_transform_key,
    target_transform_key,
    transform_dir,
    batch_size,
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
      batch_size: Size of batch to use for DataLoaders
      evaluation: If `True`, fix number of epochs to 1.
      split: Split of the active dataset to load

    Returns:
      data_loader, transform, target_transform.
    """
    xr_data, transform, target_transform = get_dataset(
        active_dataset_name,
        model_src_dataset_name,
        input_transform_key,
        target_transform_key,
        transform_dir,
        batch_size,
        split,
        evaluation,
    )

    variables, target_variables = get_variables(model_src_dataset_name)

    xr_dataset = XRDataset(xr_data, variables, target_variables)
    data_loader = DataLoader(xr_dataset, batch_size=batch_size, shuffle=True)

    return data_loader, transform, target_transform
