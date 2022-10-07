from glob import glob
import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path

import wandb
import os, sys
import torch
from dataset.forest import train_test_val_split

from trainer.summary import Summary
from utils import flat_omegadict, set_random_seed


logger = logging.getLogger(__name__)
api = wandb.Api()

def isdebugging():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        print('No sys.gettrace')
    elif gettrace():
        print('Hmm, Big Debugger is watching me')
        return True
    return False

@hydra.main(config_path='./config', config_name='config')
def main(CFG: DictConfig) -> None:
    # initial logging file
    logger.info(OmegaConf.to_yaml(CFG))
    config = flat_omegadict(CFG)
    set_random_seed(0)
    if (device_n := CFG.CUDA_VISIBLE_DEVICES) is not None:
        # import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_n)
    if isdebugging():
        os.environ['CUDA_VISIBLE_DEVICES'] = '8'
        print('CUDA_VISIBLE_DEVICES: ', os.environ['CUDA_VISIBLE_DEVICES'])
    device = torch.device('cuda' if torch.cuda.is_available() and CFG.use_gpu else 'cpu')


    # if needs predictions
    data_dir = Path(hydra.utils.get_original_cwd()) / CFG.DATASET.data_dir
    print(CFG.DATASET.data_dir)
    if 'train' in CFG.DATASET.data_dir:      
        CFG.DATASET.data_dir = str(data_dir)
        train_path, val_path, test_path = train_test_val_split(CFG.DATASET.data_dir)
        summary = Summary(test_path, CFG.wandb_project, CFG.run_name, CFG.seed, True)
        summary.plot_pred_true_values(CFG.DATASET)
    else:
        print('test')
        test_path = sorted(glob(f"{str(data_dir)}/*.pt"))
        summary = Summary(test_path, CFG.wandb_project, CFG.run_name, CFG.seed, True)
        summary.save_inference(CFG.DATASET)


if __name__ == '__main__':
    main()
