import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self, device, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = torch.tensor(eps, device=device)
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss