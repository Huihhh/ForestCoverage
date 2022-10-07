from typing import List, Callable, Iterable, Optional
import torch
import numpy as np
from PIL import Image

from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms as T


def train_test_val_split(data_dir: str, test_ratio: float = 0.2, val_ratio: float = 0.1, seed: int = 0) -> List[str]:
    '''
    Split the data paths with given test_ratio and val_ratio (to all samples)

    Parameters
    -----------
    * data_dir: the folder containing all samples
    * test_ratio: 0-1, default 0.2
    * val_ratio: 0-1, default 0.1
    * seed: default 0, random train and val data
    '''
    all_samples = np.array(sorted(glob(f"{data_dir}/*.pt")))  # sorted for reproducibility
    y0 = []
    y1 = []
    for s in all_samples:
        y = torch.load(s)['y']
        if y > 0:
            y1.append(y)
        else:
            y0.append(y)
    train_paths, val_paths, test_paths = [], [], []
    for y in [y0, y1]:
        n = np.array(y).size
        n_test = int(test_ratio * n)
        n_val = int(val_ratio * n)
        np.random.seed(100)
        indices = np.random.permutation(n)
        test_paths.append(all_samples[indices[:n_test]])
        train_val_paths = all_samples[indices[n_test:]]
        np.random.seed(seed)
        indices = np.random.permutation(train_val_paths.size)
        val_paths.append(all_samples[indices[:n_val]])
        train_paths.append(all_samples[indices[n_val:]])
    return np.concatenate(train_paths), np.concatenate(val_paths), np.concatenate(test_paths)


class ForestDataset(Dataset):
    MEAN = (0.3444, 0.3803, 0.4078)  #(0.4914, 0.4822, 0.4465),  #
    STD = (0.2037, 0.1366, 0.1148)  #}, # (0.2471, 0.2435, 0.2616),  #

    def __init__(
        self,
        path: List[str]=[],
        get_raw: bool=False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        '''
        Forest Torch Dataset

        Parameter
        ----------
        * path: path of training/val/test data (folder, abs path if using hydra)

        Return (getitem)
        -------
        X: PIL Image from numpy ndarray, (3,100,100)
        y: numpy float32, (1,)
        '''
        self.path = path
        # default transform
        if transform is None:
            transform = T.Compose([T.ToTensor(), T.Normalize(mean=self.MEAN, std=self.STD)])
        self.transform = transform
        self.target_transform = target_transform
        self.get_raw = get_raw

    def __getitem__(self, idx):
        X = torch.load(self.path[idx])

        # cast input and target to float
        x = np.moveaxis(X["x"], 0, -1)
        y_raw = X['y'] if X.get('y') else np.array(0)
        y_raw = y_raw.astype(np.float)
        y = np.log10(y_raw) / 4 if y_raw > 0 else y_raw
        y = np.array([y])

        img = Image.fromarray(x)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            y = self.y_transform(y)
        if self.get_raw:
            return img, y, y_raw, self.path[idx].split('/')[-1]
        else:
            return img, y

    def __len__(self):
        return len(self.path)
