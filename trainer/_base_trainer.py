import logging
from typing import Any, Callable
from pytorch_lightning import LightningModule, LightningDataModule
import torch
import torch.nn.functional as F
from torchmetrics import R2Score
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.nn import BCELoss

from utils import acc_topk, accuracy
from utils.ema import EMA
from utils.losses import RMSELoss

logger = logging.getLogger(__name__)

class PLModel(LightningModule):
    def __init__(self, model: Any, dataset: LightningDataModule,
        n_epoch: int=100,
        lr: float=0.01,
        use_scheduler: bool=True,
        warmup: int=0,
        wdecay: float=0.01,
        batch_size: int=32,
        ema_used: bool=False, 
        ema_decay: float=0.9,
        loss_lambda: float=1,
        **kwargs
    ) -> None:
        '''
        Pytorch lightning trainer, Training process.

        Parameter
        ----------
        * model: NN model
        * dataset: pytorch-lightning LightningDataModule
        * n_epoch: number of training epochs
        * th: threshold on sigmoid for prediction
        * lr: learning rate
        * warmup: warmup epochs
        * wdecay: weight decay
        * batch_size: default 32
        * ema_used: if use ema
        * ema_decay: ema decay rate
        '''
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.n_epoch = n_epoch
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.wdecay = wdecay
        self.batch_size=batch_size
        self.warmup = warmup

        self.init_criterion(loss_lambda)
        # used EWA or not
        self.EMA = ema_used
        if self.EMA:
            self.EMA_DECAY = ema_decay,
            self.ema_model = EMA(self.model, ema_decay)
            logger.info("[EMA] initial ")
    
    def init_criterion(self, loss_lambda) -> Callable:
        '''
        Generate loss function
        '''
        rmse_loss_fn = RMSELoss(device=self.device)
        r2score_fn = R2Score()
        bce_loss_fn = BCELoss()

        def compute_loss(out_reg, out_cf, y, y_cf):
            y_pred = (out_cf > 0.5).type(torch.float)
            rmse_loss =  rmse_loss_fn(out_reg, y)
            bce_loss =  bce_loss_fn(out_cf, y_cf) if loss_lambda>0 else torch.tensor(0)     
            r2score =  r2score_fn((out_reg * y_pred).cpu(), y.cpu())
            r2score_reg =  r2score_fn(out_reg.cpu(), y.cpu())
            
            return {
                'total_loss': rmse_loss + loss_lambda * bce_loss,
                'rmse_loss': rmse_loss, 
                'bce_loss': bce_loss,
                'r2score': r2score, 
                'r2score_reg': r2score_reg, 
                }
        self.criterion = compute_loss

    def configure_optimizers(self):
        # optimizer
        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.wdecay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        
        if self.use_scheduler:
            scheduler = {
                'scheduler': StepLR(optimizer, step_size=10000, gamma=0.8),
                'interval': 'step',
                # 'strict': True,
            }
            return [optimizer], [scheduler]
        return optimizer

    def forward(self, x):
        return self.model.forward(x)

    def on_post_move_to_device(self):
        super().on_post_move_to_device()
        # used EWA or not
        # init ema model after model moved to device
        if self.EMA:
            self.ema_model = EMA(self.model, self.EMA_DECAY)
            logger.info("[EMA] initial ")


    def training_step(self, batch, batch_idx=None):
        if hasattr(self.dataset, 'aug_times'):
            batch0, batch1 = batch
            x = torch.cat([batch0[0], batch1[0]])
            y = torch.cat([batch0[1], batch1[1]])
        else:
            x, y = batch
        
        y_cf = (y > 0).type(torch.float)
        out = self.model.forward(x.float())
        losses = self.criterion(*out, y.float(), y_cf)
        acc = accuracy((out[1] > 0.5).type(torch.float), y_cf)
        for name, metric in losses.items():
            self.log(f'train.{name}', metric.item(), on_step=False, on_epoch=True, batch_size=self.batch_size, prog_bar=True, logger=True)
        self.log('train.acc', acc.item(), on_step=False, on_epoch=True, batch_size=self.batch_size)
        return losses['total_loss']

    def training_step_end(self, loss, *args, **kwargs):
        if self.EMA:
            self.ema_model.update_params()
        return loss

    def on_train_epoch_end(self) -> None:
        if self.EMA:
            self.ema_model.update_buffer()
            self.ema_model.apply_shadow()

    def validation_step(self, batch, batch_idx=None):
        x, y, = batch
        out = self.model.forward(x)
        y_cf = (y > 0).type(torch.float)
        losses = self.criterion(*out, y, y_cf)
        acc = accuracy((out[1] > 0.5).type(torch.float), y_cf)
        for name, metric in losses.items():
            self.log(f'val.{name}', metric.item(), on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log('val.acc', acc.item(), on_step=False, on_epoch=True, batch_size=self.batch_size)
        return losses['r2score']

    def validation_epoch_end(self, *args, **kwargs):
        if self.EMA:
            self.ema_model.restore()

    def test_step(self, batch, batch_idx=None):
        x, y = batch
        out = self.model.forward(x)
        y_cf = (y > 0).type(torch.float)
        losses = self.criterion(*out, y, y_cf)
        acc = accuracy((out[1] > 0.5).type(torch.float), y_cf)
        for name, metric in losses.items():
            self.log(f'test.{name}', metric.item(), on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log('test.acc', acc.item(), on_step=False, on_epoch=True, batch_size=self.batch_size)
        return losses['r2score']