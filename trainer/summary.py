import enum
import io
import logging
from typing import Any
from unicodedata import decimal
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning.trainer.trainer import Trainer
import wandb
import hydra
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from models import *
from dataset import PlDataset
from dataset.forest import ForestDataset

api = wandb.Api()
logger = logging.getLogger(__name__)
label_encoder = LabelEncoder()

def fig2html(fig) -> Any:
    '''
    write plotly figure to buffer to get html

    Parameter
    ----------
    fig: plotly figure

    Return
    ---------
    html that can be passed to wandb.Html(html)
    '''
    buffer = io.StringIO()
    fig.write_html(buffer)
    return buffer.getvalue()

class Summary:
    '''
    Use wandb.api to revisit trained models or wandb run summaries.
    '''

    def __init__(self, data_path, wandb_project: str, run_name: str, seed: int = 0, use_gpu: bool = True) -> None:
        '''
        Parameters
        ---------------
        * wandb_project: wandb project name, used to init wandb run
        * run_name: current run name, used to init wandb run
        * seed: random seed
        * use_gpu: bool, used to init device
        '''
        self.root = Path(hydra.utils.get_original_cwd())
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.seed = seed
        self.wandb_project = wandb_project
        self.run_name = run_name
        self.path = data_path

    def plot_pred_true_values(self, data_cfg):
        wandb.init(project=self.wandb_project, name=self.run_name)
        
        root_dir = Path(hydra.utils.get_original_cwd())
        # *********************************** prapare data *********************************
        # new Dataset for circles, moons, spirals
        dataset = ForestDataset(self.path, get_raw=True)
        for n, loss_lambda in enumerate([0, 1]):
            test_res = inference(dataset, data_cfg, self.seed, loss_lambda, self.device)
            print(test_res)
            fig1 = px.scatter(test_res, x='regression_output', y='regression_label', trendline='ols')
            fig2 = px.scatter(test_res, x='regression_classfication_output', y='regression_label', trendline='ols')
            html1 = fig2html(fig1)
            html2 = fig2html(fig2)
            wandb.log({f'regression output v.s ground truth (classification factor: {loss_lambda})': wandb.Html(html1)})
            wandb.log({f'regression output * classification output v.s ground truth (classification factor: {loss_lambda})': wandb.Html(html2)})
            filepath = root_dir / 'csv'/f'test_inference_lambda{loss_lambda}_seed{self.seed}.csv'
            filepath.parent.mkdir(parents=True, exist_ok=True)
            print('filepath', filepath)
            test_res.to_csv(filepath)    
        wandb.finish()


    def save_inference(self, data_cfg):
        wandb.init(project=self.wandb_project, name=self.run_name)
        
        # *********************************** prapare data *********************************
        # new Dataset for circles, moons, spirals
        dataset = ForestDataset(self.path, get_raw=True)
        root_dir = Path(hydra.utils.get_original_cwd()) / 'results'
        root_dir.mkdir(parents=True, exist_ok=True)
        for n, loss_lambda in enumerate([0, 1]):
            logger.info(f'root_dir{str(root_dir)}')
            test_res = inference(dataset, data_cfg, self.seed, loss_lambda, self.device)
            # out_reg_cf = torch.where(out_reg_cf>0, 10**(out_reg*4), 0)
            # out_reg = torch.where(out_reg>0, 10**(out_reg*4), 0)
            filepath = root_dir / f'inference_lambda{loss_lambda}_seed{self.seed}.csv'
            print('filepath', filepath)
            test_res.to_csv(filepath)    
        wandb.finish()


def load_model_artifact(project: str, filters: dict) -> nn.Module:
    '''
    load torch.nn.Module from wandb artifact with given filters

    Parameters
    ----------
    * project: wandb project name
    * filters: dict, used to filter runs

    Return
    ----------
    * model: nn model
    '''
    runs = api.runs(project, filters=filters)
    print('len(runs)', len(runs))
    artifacts = runs[0].logged_artifacts()
    print(len(artifacts), artifacts)
    filtered = [atf for atf in artifacts if atf._sequence_name=='model_epoch199']
    model_path = filtered[0].download() #* use the first run
    model = torch.load(Path(model_path) / 'model.h5')
    return model

def inference(dataset, data_cfg,
                       seed: int,
                       loss_lambda: int,
                       device: torch.device):
    '''
    Inference on trained model. A pandas dataframe is returned, containing following columns:
    regression output, classification output (binarized), 
    regression output corrected by the classification output,
    regression label (log transformed), true count, regression count (transformed back), 
    regression*classification count (transformed back), input file name

    Parameters
    ------------
    * dataset: torch.Dataset object,
    * data_cfg: data config, attributes must have: batch_size, num_workers
    * loss_lambda: controls the contribution of the classification head in loss
    '''
    test_res = pd.DataFrame(columns=['regression_output', 'classification_output', 'regression_classfication_output', 'regression_label', 'file_name'])
    dataloader = DataLoader(dataset,
                    batch_size=data_cfg.batch_size,
                    num_workers=data_cfg.num_workers,
                    pin_memory=False,
                    drop_last=False)
    # *********************************** build model *********************************
    model = load_model_artifact('forest-coverage', 
                    filters={
                        'config.run_name': 'reg_only_test_loss_corrected' if loss_lambda==0 else 'reg_cf_test_loss_corrected',
                        'config.seed': seed,
                        'config.TRAIN_loss_lambda': loss_lambda,
                        'config.log_artifact_every': [180, 190, 195, 198, 199]
                    })
    logger.info("[Model] Loading model...")

    model = model.to(device=device)
    model.eval()
    for x, y, y_raw, path in dataloader:
        x = x.to(device)
        out_reg, out_cf = model(x)
        out_reg = out_reg.detach().squeeze()
        out_cf = torch.where(out_cf>0.5, 1, 0).squeeze()
        out_reg_cf = (out_reg * out_cf).detach().squeeze()
        reg_cf_count = (10**(out_reg_cf * 4)*out_cf).round().type(torch.int16)#torch.where(out_reg_cf>0, 10**(out_reg_cf*4), 0)#.type(torch.int16)
        reg_count = (10**(out_reg * 4)).round().squeeze().type(torch.int16)
        reg_count[reg_count==1] = 0
        df = pd.DataFrame({
            'regression_output': out_reg.detach().cpu(),
            'classification_output': out_cf.detach().cpu(),
            'regression_classfication_output': out_reg_cf.detach().cpu(),
            'regression_label': y.squeeze(),
            'true_count': y_raw,
            'reg_count': reg_count.cpu(),
            'reg_cf_count': reg_cf_count.cpu(),
            'file_name': path
        })
        test_res = pd.concat([test_res, df], ignore_index=True)
    return test_res


def get_total_error(df, lmbd: float=0, bin_width: int=50):
    '''
    Get total error for each prediction:
    * reg_count: predicted count from regression model (only regression head is trained) if lmbd=0
                 predicted count from regression head (both regression & classification heads are trained) if lmbd=1
    * reg_cf_count: predicted count from regression head multiplied by classification predictions (both regression & classification heads are trained), lmbd=1
                    not used if lmbd=0

    Parameters
    ------------
    * bin_width: devides true counts, to determine which bin each prediction
    '''
    df.loc[df['reg_count'] > 10000, 'reg_count'] = 10000
    if lmbd == 1:
        df.loc[df['reg_cf_count'] > 10000, 'reg_cf_count'] = 10000
    df['delta'] = df['true_count'] - df['reg_count']
    df['bins'] = df['true_count'] // bin_width 
    df['log_bins'] = np.log10(df['true_count']) // 0.1

    true_sum = df['true_count'].sum()

    if lmbd == 1:
        return true_sum - df['reg_count'].sum(), true_sum - df['reg_cf_count'].sum()
    
    return true_sum - df['reg_count'].sum()






