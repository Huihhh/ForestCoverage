# Forest Coverage

This project aims to estimate the amount of pixels in an image that belongs to forests given a satellite image.



## Framework
![image](assets/framework.png)

## Requirements
- [Hydra](https://hydra.cc/)
- [Weights & Biases](https://wandb.ai/site)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)

## Run
```shell
python train.py --config-name config \
         wandb_project=forest-coverage \
         run_name=reg_only_test_loss_corrected \
         TRAIN.n_epoch=200 \
         TRAIN.lr=0.0002 \
         TRAIN.wdecay=0.008 \
         TRAIN.loss_lambda=0 \
         log_artifact_every=[180,190,195,198,199] \
         seed=0
```

## Result Visualization
[slides](https://docs.google.com/presentation/d/1IK6USMBe9VH8_b3_GdN1k45Tb2VM6SzP/edit?usp=sharing&ouid=109297129024520349522&rtpof=true&sd=true)

[code](https://drive.google.com/file/d/1rPRh13gtYRcn-fhtLMxgKzr34zrLZw04/view?usp=sharing)




