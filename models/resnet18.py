import re
import os
import sys
from typing import List
from torch import nn
from torch.nn.modules import Module
import torchvision.models as models
import torch
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

ACT_METHOD = {'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU()}


class ResNet(Module):
    def __init__(self,
                 h_nodes: List[int],
                 out_dim: int=1,
                 activation: str='relu',
                 dropout: float=0,
                 use_bn: bool=False,
                 seed: int=0,
                 *args,
                 **kwargs) -> None:
        super().__init__()
        resnet = models.resnet18(pretrained=True)

        # *** FC layers ***
        h_nodes = [resnet.fc.in_features] + list(h_nodes)
        self.use_bn = use_bn
        self.n_neurons = sum(h_nodes)
        self.layers = []
        for i in range(len(h_nodes) - 1):
            s = nn.Sequential()
            torch.random.manual_seed(i + seed)
            s.add_module('fc', nn.Linear(h_nodes[i], h_nodes[i + 1]))
            s.add_module('ac', ACT_METHOD[activation])
            if use_bn:
                bn = nn.BatchNorm1d(h_nodes[i + 1])
                s.add_module('bn', bn)
            if dropout != 0:
                s.add_module('dropout', nn.Dropout(p=dropout))
            self.layers.append(s)

        # self.layers.append(nn.Linear(h_nodes[-1], out_dim))
        self.fcs = torch.nn.Sequential(*self.layers)
        self.fcs.add_module('reg_head', nn.Linear(h_nodes[-1], 1))
        self.fcs.add_module('cf_head', nn.Linear(h_nodes[-1], out_dim))

        # *** ResNet ***
        self.resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        # freeze the pretrained model
        # for params in self.resnet[:kwargs['freeze_layers']].parameters():
        #     params.requires_grad = False
        
        
    def reset_parameters(self, winit: dict, binit: dict, seed: int=0) -> None:
        '''
        reinit the model's weights and bias (batchnorm weights excluded)

        Parameter
        ---------
        * winit: dict, init method for weights
        * binit: dict, init method for bias
        * seed: random seed
        '''
        p1 = re.compile(r'^((?!bn).)*weight') #find conv weight
        # p2 = re.compile(r'^((?!bn).)*bias') # excluding bn bias
        for i, (name, param) in enumerate(self.named_parameters()):
            torch.random.manual_seed(i + seed)
            if p1.search(name):
                eval(winit.func)(param, **winit.params)
                continue

    def forward(self, x):
        x = self.resnet(x)
        x= x.reshape(x.shape[0], -1)
        reg_out = self.fcs.reg_head(x)
        cf_out = self.fcs.cf_head(x)
        return reg_out, torch.sigmoid(cf_out)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf
    import os
    import sys
    sys.path.append(os.getcwd())
    print(os.getcwd())
    import os
    import sys
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)

    @hydra.main(config_name='config', config_path='../config')
    def main(CFG: DictConfig):
        print('==> CONFIG is \n', OmegaConf.to_yaml(CFG.MODEL), '\n')
        net = ResNet(**CFG.MODEL)
        print(net)
        print()

    main()
