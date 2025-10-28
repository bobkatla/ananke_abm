from torch.utils.data import Dataset
import numpy as np
import torch


class GridDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.Y = d["Y"].astype(np.int64)
    def __len__(self):
        return self.Y.shape[0]
    def __getitem__(self, i):
        y = self.Y[i]
        return torch.from_numpy(y)


def read_n_split_data(val_frac, data_npz_path, seed):
    full_dataset = GridDataset(data_npz_path)
    num_total = len(full_dataset)
    num_val = max(1, int(num_total * val_frac))
    num_train = num_total - num_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(seed),
    )
    return train_dataset, val_dataset