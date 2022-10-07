import logging
from typing import Type, Any, Callable, Union, List, Optional
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torchvision import transforms as T

from dataset.forest import train_test_val_split, ForestDataset

logger = logging.getLogger(__name__)

class PlDataset(LightningDataModule):

    def __init__(self, 
                data_dir: str,
                name: Optional[str] = None,
                n_val: float = 0.1,
                n_test: float = 0.2,
                batch_size: int = 32,
                num_workers: int = 4,
                seed: int=0,
                **kwargs):
        '''
        Lightning Data Module

        Parameter:
        ----------
        * name: optional, str
        * n_val: float, 0-1
        * n_test: float, 0-1
        * batch_size: int, default 32,
        * num_workers: int, default 4
        '''
        super().__init__()
        self.name = name
        self.n_val = n_val
        self.n_test = n_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.paths = train_test_val_split(data_dir, n_test, n_val, seed)
        self.get_dataset(**kwargs)


    def get_dataset(self, use_aug: bool=False, aug_times: int=0, **kwargs):
                # *** stack augmented data ***
        if use_aug:
            self.aug_times = aug_times
            logger.info('********* apply data augmentation ***********')
            transform = T.Compose([
                T.RandomHorizontalFlip(p=kwargs['flip_p']),
                T.RandomApply([
                    T.RandomCrop(size=64, padding=int(64 * 0.125), padding_mode='reflect'),
                ], p=0.5),
                T.RandomApply([
                    T.RandomRotation(90),
                ], p=0.5),
                T.RandomAdjustSharpness(0.5),
                T.ToTensor(),
                T.Normalize(mean=ForestDataset.MEAN, std=ForestDataset.STD)
            ])
            self.trainset_aug = ForestDataset(self.paths[0], transform)
        
        self.trainset = ForestDataset(self.paths[0])
        self.valset = ForestDataset(self.paths[1])
        self.testset = ForestDataset(self.paths[2])

    def train_dataloader(self):
        kwargs = dict(num_workers=self.num_workers, pin_memory=False)
        if hasattr(self, 'aug_times'):
            replacement = self.aug_times > 1
            num_samples = self.aug_times * len(self.trainset_aug) if replacement else None
            data_sampler = RandomSampler(self.trainset_aug, replacement=replacement, num_samples=num_samples)
            aug_loader = DataLoader(self.trainset_aug,
                            batch_sampler=BatchSampler(data_sampler, self.batch_size * self.aug_times, drop_last=False),
                            **kwargs)
            train_loader = CombinedLoader([
                DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, **kwargs),
                aug_loader
            ])
        else:
            train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, **kwargs)

        return train_loader

    def val_dataloader(self):
        return DataLoader(self.valset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=False,
                          drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.testset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=False,
                          drop_last=False)

    
        



    
