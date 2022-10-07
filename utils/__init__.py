import os
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf


def accuracy(output, target):
    correct_n = (output == target).sum()
    batch_size = target.size(0)
    return correct_n / batch_size

def acc_topk(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262 """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True) # pred is the indices
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def flat_omegadict(odict, parentKey=None, connector='_'):
    flat = {}
    for key, value in odict.items():
        if isinstance(value, DictConfig) or isinstance(value, dict):
            flat |= flat_omegadict(value, key)
        else:
            newKey = f'{parentKey}{connector}{key}' if parentKey else key
            flat |= {newKey: value}
    return flat


def set_random_seed(seed):
    if seed == 'None':
        seed = random.randint(1, 10000)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False