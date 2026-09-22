# Physics-Informed Super-Resolution Framework for Climate Data Downscaling

This repository contains preprocessing, training, and evaluation code for the downscaling experiments in the manuscript.

Implemented models:

- DDPM
- Physics-DDPM
- FNO
- PINO
- SR-GAN
- PINNSR

ERA5 and IBTrACS are not redistributed in this repository. Download them from the original providers before running the experiments.

## Repository structure

```text
.
├── preprocess.py
├── train.py
├── evaluate.py
├── DATA.md
├── requirements.txt
├── .gitignore
└── data/
    └── .gitkeep
```

## Environment

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is recommended.

## Data

See [`DATA.md`](DATA.md). The default local layout is:

```text
./data/era5_2014.nc
./data/ibtracs.WP.list.v04r00.csv
```

## Preprocessing

```bash
python preprocess.py \
  --nc_path ./data/era5_2014.nc \
  --output ./outputs/preprocessed_boxmean_wind_torch.npz
```

## Training

Train all models:

```bash
python train.py \
  --output_dir ./outputs \
  --preprocessed_npz ./outputs/preprocessed_boxmean_wind_torch.npz \
  --models all
```

Train selected models:

```bash
python train.py \
  --output_dir ./outputs \
  --preprocessed_npz ./outputs/preprocessed_boxmean_wind_torch.npz \
  --models pino,pinnsr \
  --overwrite
```

The default configuration uses CUDA devices `0,1`. For one GPU:

```bash
python train.py \
  --output_dir ./outputs \
  --preprocessed_npz ./outputs/preprocessed_boxmean_wind_torch.npz \
  --models all \
  --single_gpu \
  --device cuda:0
```

## Evaluation

```bash
python evaluate.py \
  --output_dir ./outputs \
  --preprocessed_npz ./outputs/preprocessed_boxmean_wind_torch.npz \
  --ibtracs ./data/ibtracs.WP.list.v04r00.csv \
  --device cuda:0
```

The evaluator reports PSNR, SSIM, LPIPS, one-step PDE RMSE, and typhoon-event spectral RMSE.

## Generated files

```text
outputs/
├── preprocessed_boxmean_wind_torch.npz
├── run_config.json
└── checkpoints/
    ├── ddpm/final.pt
    ├── physics_ddpm/final.pt
    ├── fno/final.pt
    ├── pino/final.pt
    ├── srgan/final.pt
    └── pinnsr/final.pt
```
