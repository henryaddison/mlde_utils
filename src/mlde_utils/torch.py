import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .training.dataset import get_dataset, get_variables


class XRDataset(Dataset):
    def __init__(self, ds, variables, target_variables):
        self.ds = ds
        self.variables = variables
        self.target_variables = target_variables

    @classmethod
    def to_tensor(cls, ds, variables):
        return torch.tensor(
            # stack features before lat-lon (HW)
            np.stack([ds[var].values for var in variables], axis=-3)
        ).float()

    def __len__(self):
        return len(self.ds.time)

    def __getitem__(self, idx):
        subds = self.ds.isel(time=idx)
        cond = self.to_tensor(subds, self.variables)
        x = self.to_tensor(subds, self.target_variables)
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
      split: Split of the active dataset to load
      evaluation: If `True`, fix number of epochs to 1.

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
