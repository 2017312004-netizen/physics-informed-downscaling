#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for the six downscaling models.
"""

# ============================================================
# Imports
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import threading
import queue
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

@dataclass
class TrainConfig:
    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    output_dir: str = "./outputs"
    preprocessed_npz: str = "./outputs/preprocessed_boxmean_wind_torch.npz"

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    seed: int = 42
    batch_size: int = 8
    epochs: int = 60
    learning_rate: float = 1e-4
    num_workers: int = 0
    grad_clip_norm: float = 1.0

    save_every_epochs: int = 10

    skip_existing_models: bool = True
    force_retrain: bool = False

    # --------------------------------------------------------
    # GAN / PINNSR
    # --------------------------------------------------------
    adv_weight: float = 0.001
    gp_weight: float = 10.0

    pinnsr_rec_weight: float = 0.649
    pinnsr_phys_weight: float = 0.35

    srgan_rec_weight: float = 0.999

    # --------------------------------------------------------
    # DDPM / Physics-DDPM
    # --------------------------------------------------------
    diffusion_steps: int = 100
    diffusion_base_channels: int = 64
    diffusion_beta_start: float = 1e-4
    diffusion_beta_end: float = 0.015

    ddpm_rec_weight: float = 2.0
    ddpm_eps_weight: float = 0.05
    physics_ddpm_phys_weight: float = 0.15

    diffusion_train_x0_clamp: bool = True

    diffusion_init_noise_std: float = 0.15
    diffusion_auto_noise_std: bool = True
    diffusion_noise_max_samples: int = 512

    # --------------------------------------------------------
    # FNO / PINO
    # --------------------------------------------------------
    fno_modes_h: int = 16
    fno_modes_w: int = 16
    fno_width: int = 64
    fno_layers: int = 4

    pino_rec_weight: float = 0.85
    pino_phys_weight: float = 0.15

    # --------------------------------------------------------
    # Physics loss
    # --------------------------------------------------------
    physics_clip_scale: float = 0.10

    physics_common_raw_k0: float = 1.0
    physics_common_raw_k1: float = 0.1

    physics_grad_balance_ema_decay: float = 0.90
    physics_grad_balance_eps: float = 1e-12
    physics_grad_balance_min_scale: float = 1e-8
    physics_grad_balance_max_scale: float = 1e4
    physics_grad_match_ratio: float = 1.0

    physics_warmup_epochs: int = 5
    physics_ramp_epochs: int = 15

    dx_scaled: float = 0.359
    dy_scaled: float = 0.3588
    cfl: float = 0.5
    diffusion_cfl: float = 0.20
    max_delta_t: float = 1.0


    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------
    device: str = "cuda"

    models: str = "all"

    overwrite_selected_models: bool = False

    gpu_ids: str = "0,1"
    parallel_gpu_training: bool = True


MODEL_NAMES = [
    "ddpm",
    "physics_ddpm",
    "fno",
    "pino",
    "srgan",
    "pinnsr",
]


def parse_model_selection(value):
    """
    Parse "all" or a comma-separated model subset.

    Examples
    --------
    "all"                   -> all six models
    "pino,pinnsr"           -> PINO + PINNSR only
    "ddpm,physics_ddpm"     -> the diffusion pair only
    """
    if isinstance(
        value,
        (list, tuple),
    ):
        tokens = [
            str(x).strip().lower()
            for x in value
            if str(x).strip()
        ]
    else:
        tokens = [
            token.strip().lower()
            for token in str(value).split(",")
            if token.strip()
        ]

    if not tokens:
        raise ValueError(
            "No models were selected."
        )

    if "all" in tokens:
        if len(tokens) != 1:
            raise ValueError(
                '"all" cannot be combined with individual model names.'
            )
        return list(
            MODEL_NAMES
        )

    unknown = [
        token
        for token in tokens
        if token not in MODEL_NAMES
    ]

    if unknown:
        raise ValueError(
            "Unknown model selection: "
            + ", ".join(
                unknown
            )
            + ". Valid names are: "
            + ", ".join(
                MODEL_NAMES
            )
        )

    # Deduplicate while preserving the user's order.
    result = []
    seen = set()

    for token in tokens:
        if token not in seen:
            seen.add(
                token
            )
            result.append(
                token
            )

    return result


# ============================================================
# Runtime helpers
# ============================================================

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False


def get_device(requested: str):
    requested = str(requested).strip().lower()

    if requested == "auto":
        requested = "cuda"

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False. "
                "This training script is intended to run on an NVIDIA GPU."
            )

        device = torch.device(requested)
        if device.index is not None:
            torch.cuda.set_device(device.index)
        else:
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")

        return device

    return torch.device(requested)


def clip_grad(parameters, max_norm: float):
    if max_norm is not None and float(max_norm) > 0:
        torch.nn.utils.clip_grad_norm_(parameters, float(max_norm))


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)


def save_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def mean_rows(rows):
    return np.mean(np.asarray(rows, dtype=np.float64), axis=0)


# ============================================================
# Data loading
# ============================================================

def default_preprocessed_path(
    cfg: TrainConfig,
) -> Path:
    return Path(cfg.preprocessed_npz).expanduser()


def load_preprocessed_dataset(
    path: Path,
):
    z = np.load(path, allow_pickle=False)

    required = [
        "x_lr", "y_hr", "u_hr", "v_hr",
        "time_iso", "lat", "lon",
        "train_idx", "val_idx", "test_idx",
        "ws_scaler", "uv_scaler",
    ]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")

    def scalar_json(name):
        value = z[name]
        if isinstance(value, np.ndarray):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(str(value))

    data = {
        "x_lr": z["x_lr"],
        "y_hr": z["y_hr"],
        "u_hr": z["u_hr"],
        "v_hr": z["v_hr"],
        "time_iso": z["time_iso"],
        "lat": z["lat"],
        "lon": z["lon"],
        "train_idx": z["train_idx"],
        "val_idx": z["val_idx"],
        "test_idx": z["test_idx"],
        "ws_scaler": scalar_json("ws_scaler"),
        "uv_scaler": scalar_json("uv_scaler"),
    }

    print("[LOAD]", path)
    print("[DATA] x_lr:", data["x_lr"].shape)
    print("[DATA] y_hr:", data["y_hr"].shape)
    print(
        "[DATA] split:",
        f"train={len(data['train_idx'])}",
        f"val={len(data['val_idx'])}",
        f"test={len(data['test_idx'])}",
    )
    return data


# ============================================================
# Dataset objects
# ============================================================

class SupervisedWindDataset(Dataset):
    def __init__(
        self,
        data,
        indices,
    ):
        idx = np.asarray(
            indices,
            dtype=np.int64,
        )

        self.x = data["x_lr"][
            idx
        ].astype(np.float32)

        self.y = data["y_hr"][
            idx
        ].astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return (
            torch.from_numpy(
                self.x[i]
            ),
            torch.from_numpy(
                self.y[i]
            ),
        )


class PhysicsWindDataset(Dataset):
    """
    Same-time data used by PINNSR, PINO, Physics-DDPM.

    Physics consistency advances y_true(t) and y_pred(t) through
    the same PDE operator Phi. A t-1/t target pair is therefore
    not required by the physics loss itself.
    """

    def __init__(
        self,
        data,
        indices,
    ):
        idx = np.asarray(
            indices,
            dtype=np.int64,
        )

        self.x = data["x_lr"][
            idx
        ].astype(np.float32)

        self.y = data["y_hr"][
            idx
        ].astype(np.float32)

        self.u = data["u_hr"][
            idx
        ].astype(np.float32)

        self.v = data["v_hr"][
            idx
        ].astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return (
            torch.from_numpy(
                self.x[i]
            ),
            torch.from_numpy(
                self.y[i]
            ),
            torch.from_numpy(
                self.u[i]
            ),
            torch.from_numpy(
                self.v[i]
            ),
        )


def make_loader(
    dataset,
    cfg: TrainConfig,
    shuffle=True,
):
    loader_generator = torch.Generator()
    loader_generator.manual_seed(
        int(cfg.seed)
    )

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=loader_generator,
    )


# ============================================================
# Shared neural-network blocks
# ============================================================

class GroupNorm2d(nn.GroupNorm):
    def __init__(self, channels):
        super().__init__(
            num_groups=min(
                8,
                channels,
            ),
            num_channels=channels,
        )


class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
            ),
            GroupNorm2d(
                channels
            ),
            nn.PReLU(),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
            ),
            GroupNorm2d(
                channels
            ),
        )

    def forward(self, x):
        return x + self.net(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        reduction=16,
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            4,
        )

        self.pool = nn.AdaptiveAvgPool2d(
            1
        )

        self.fc = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                1,
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Conv2d(
                hidden,
                channels,
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(
            self.pool(x)
        )


# ============================================================
# SR-GAN / PINNSR generator
# ============================================================

class SRGenerator(nn.Module):
    def __init__(
        self,
        hr_shape=(1, 161, 161),
        tanh_final=True,
    ):
        super().__init__()

        _, H, W = hr_shape

        self.out_h = H
        self.out_w = W
        self.tanh_final = bool(
            tanh_final
        )

        self.c1 = nn.Sequential(
            nn.Conv2d(
                1,
                64,
                3,
                padding=1,
            ),
            nn.PReLU(),
            ResBlock(64),
        )

        self.d1 = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                3,
                stride=2,
                padding=1,
            ),
            nn.PReLU(),
            ResBlock(128),
        )

        self.d2 = nn.Sequential(
            nn.Conv2d(
                128,
                256,
                3,
                stride=2,
                padding=1,
            ),
            nn.PReLU(),
            ResBlock(256),
        )

        self.b = nn.Sequential(
            nn.Conv2d(
                256,
                512,
                3,
                stride=2,
                padding=1,
            ),
            nn.PReLU(),
            ResBlock(512),
        )

        self.u1_conv = nn.Sequential(
            nn.Conv2d(
                512,
                256,
                3,
                padding=1,
            ),
            nn.PReLU(),
        )

        self.u1_merge = nn.Sequential(
            nn.Conv2d(
                512,
                256,
                1,
            ),
            ResBlock(256),
        )

        self.att2 = AttentionBlock(
            256
        )

        self.u2_conv = nn.Sequential(
            nn.Conv2d(
                256,
                128,
                3,
                padding=1,
            ),
            nn.PReLU(),
        )

        self.u2_merge = nn.Sequential(
            nn.Conv2d(
                256,
                128,
                1,
            ),
            ResBlock(128),
        )

        self.att1 = AttentionBlock(
            128
        )

        self.u3_conv = nn.Sequential(
            nn.Conv2d(
                128,
                64,
                3,
                padding=1,
            ),
            nn.PReLU(),
        )

        self.u3_merge = nn.Sequential(
            nn.Conv2d(
                128,
                64,
                1,
            ),
            ResBlock(64),
        )

        self.att0 = AttentionBlock(
            64
        )

        self.final = nn.Conv2d(
            64,
            1,
            3,
            padding=1,
        )

    def forward(self, x):
        c1 = self.c1(x)
        d1 = self.d1(c1)
        d2 = self.d2(d1)
        b = self.b(d2)

        u1 = F.interpolate(
            b,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        u1 = self.u1_conv(
            u1
        )

        d2r = F.interpolate(
            self.att2(d2),
            size=u1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        u1 = self.u1_merge(
            torch.cat(
                [u1, d2r],
                dim=1,
            )
        )

        u2 = F.interpolate(
            u1,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        u2 = self.u2_conv(
            u2
        )

        d1r = F.interpolate(
            self.att1(d1),
            size=u2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        u2 = self.u2_merge(
            torch.cat(
                [u2, d1r],
                dim=1,
            )
        )

        u3 = F.interpolate(
            u2,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        u3 = self.u3_conv(
            u3
        )

        c1r = F.interpolate(
            self.att0(c1),
            size=u3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        u3 = self.u3_merge(
            torch.cat(
                [u3, c1r],
                dim=1,
            )
        )

        u3 = F.interpolate(
            u3,
            size=(
                self.out_h,
                self.out_w,
            ),
            mode="bilinear",
            align_corners=False,
        )

        out = self.final(
            u3
        )

        if self.tanh_final:
            out = torch.tanh(
                out
            )

        return out


class PINNSRGenerator(nn.Module):
    def __init__(
        self,
        cfg: TrainConfig,
        hr_shape=(1, 161, 161),
    ):
        super().__init__()

        self.base = SRGenerator(
            hr_shape=hr_shape,
            tanh_final=True,
        )

        raw_k0 = torch.tensor(
            float(cfg.physics_common_raw_k0)
        )
        raw_k1 = torch.tensor(
            float(cfg.physics_common_raw_k1)
        )

        self.register_buffer(
            "raw_K0",
            raw_k0,
        )
        self.register_buffer(
            "raw_K1",
            raw_k1,
        )

    def coefficients(self):
        return (
            F.softplus(
                self.raw_K0
            ),
            F.softplus(
                self.raw_K1
            ),
        )

    def forward(self, x):
        y = self.base(x)
        K0, K1 = self.coefficients()

        return y, K0, K1


# ============================================================
# Discriminator
# ============================================================

class Discriminator(nn.Module):
    def __init__(
        self,
        hr_shape=(1, 161, 161),
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                1,
                64,
                3,
                padding=1,
            ),
            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),
            nn.Dropout(
                0.25
            ),
        ]

        in_ch = 64

        for k, out_ch in enumerate(
            [64, 128, 128, 256, 256]
        ):
            stride = (
                2
                if k in (0, 2, 4)
                else 1
            )

            layers.extend([
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    3,
                    stride=stride,
                    padding=1,
                ),
                GroupNorm2d(
                    out_ch
                ),
                nn.LeakyReLU(
                    0.2,
                    inplace=True,
                ),
            ])

            in_ch = out_ch

        self.conv = nn.Sequential(
            *layers
        )

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                *hr_shape,
            )

            flat = self.conv(
                dummy
            ).numel()

        self.head = nn.Sequential(
            nn.Dropout(
                0.25
            ),
            nn.Flatten(),
            nn.Linear(
                flat,
                512,
            ),
            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),
            nn.Linear(
                512,
                1,
            ),
        )

    def forward(self, x):
        return self.head(
            self.conv(x)
        )


def gradient_penalty(
    discriminator,
    real,
    fake,
    device,
):
    """
    Original GAN regularization used in the development implementation.

    GP = E[(||grad_x D(x_hat)||_2 - 1)^2]

    The interpolated field and gradient calculation are forced to fp32.
    """
    real = real.float()
    fake = fake.float()

    batch_size = real.shape[0]

    alpha = torch.rand(
        batch_size,
        1,
        1,
        1,
        device=device,
        dtype=torch.float32,
    )

    interpolated = (
        alpha * real
        + (1.0 - alpha) * fake
    ).requires_grad_(True)

    pred = discriminator(
        interpolated
    )

    grad = torch.autograd.grad(
        outputs=pred,
        inputs=interpolated,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad_norm = torch.sqrt(
        torch.sum(
            grad ** 2,
            dim=(1, 2, 3),
        )
        + 1e-12
    )

    return torch.mean(
        (grad_norm - 1.0) ** 2
    )


# ============================================================
# FNO / PINO
# ============================================================

class SpectralConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes_h,
        modes_w,
    ):
        super().__init__()

        self.in_channels = int(
            in_channels
        )
        self.out_channels = int(
            out_channels
        )
        self.modes_h = int(
            modes_h
        )
        self.modes_w = int(
            modes_w
        )

        scale = (
            1.0
            / max(
                1,
                in_channels * out_channels,
            )
        )

        self.weights = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes_h,
                modes_w,
                dtype=torch.cfloat,
            )
        )

    def forward(self, x):
        b, _, h, w = x.shape

        x_ft = torch.fft.rfft2(
            x.float(),
            norm="ortho",
        )

        out_ft = torch.zeros(
            b,
            self.out_channels,
            h,
            w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        mh = min(
            self.modes_h,
            h,
        )
        mw = min(
            self.modes_w,
            w // 2 + 1,
        )

        out_ft[
            :,
            :,
            :mh,
            :mw,
        ] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[
                :,
                :,
                :mh,
                :mw,
            ],
            self.weights[
                :,
                :,
                :mh,
                :mw,
            ],
        )

        return torch.fft.irfft2(
            out_ft,
            s=(h, w),
            norm="ortho",
        )


class FNOBlock2d(nn.Module):
    def __init__(
        self,
        width,
        modes_h,
        modes_w,
    ):
        super().__init__()

        self.spectral = SpectralConv2d(
            width,
            width,
            modes_h,
            modes_w,
        )

        self.pointwise = nn.Conv2d(
            width,
            width,
            1,
        )

        self.norm = GroupNorm2d(
            width
        )

    def forward(self, x):
        return F.gelu(
            self.norm(
                self.spectral(x)
                + self.pointwise(x)
            )
        )


class FNOGenerator(nn.Module):
    def __init__(
        self,
        cfg: TrainConfig,
        hr_shape=(1, 161, 161),
    ):
        super().__init__()

        _, H, W = hr_shape

        self.out_h = int(H)
        self.out_w = int(W)

        width = int(
            cfg.fno_width
        )

        self.lift = nn.Conv2d(
            3,
            width,
            1,
        )

        self.blocks = nn.ModuleList([
            FNOBlock2d(
                width,
                cfg.fno_modes_h,
                cfg.fno_modes_w,
            )
            for _ in range(
                cfg.fno_layers
            )
        ])

        self.proj = nn.Sequential(
            nn.Conv2d(
                width,
                128,
                1,
            ),
            nn.GELU(),
            nn.Conv2d(
                128,
                1,
                1,
            ),
        )

    def make_grid(self, x):
        b, _, h, w = x.shape

        yy = torch.linspace(
            0,
            1,
            h,
            device=x.device,
            dtype=x.dtype,
        ).view(
            1,
            1,
            h,
            1,
        ).expand(
            b,
            1,
            h,
            w,
        )

        xx = torch.linspace(
            0,
            1,
            w,
            device=x.device,
            dtype=x.dtype,
        ).view(
            1,
            1,
            1,
            w,
        ).expand(
            b,
            1,
            h,
            w,
        )

        return torch.cat(
            [x, yy, xx],
            dim=1,
        )

    def forward(self, x):
        x = F.interpolate(
            x,
            size=(
                self.out_h,
                self.out_w,
            ),
            mode="bilinear",
            align_corners=False,
        )

        x = self.make_grid(
            x.float()
        )

        x = self.lift(
            x
        )

        for block in self.blocks:
            x = block(
                x
            )

        return self.proj(
            x
        )


class PINOGenerator(nn.Module):
    def __init__(
        self,
        cfg: TrainConfig,
        hr_shape=(1, 161, 161),
    ):
        super().__init__()

        self.fno = FNOGenerator(
            cfg,
            hr_shape=hr_shape,
        )

        raw_k0 = torch.tensor(
            float(cfg.physics_common_raw_k0)
        )
        raw_k1 = torch.tensor(
            float(cfg.physics_common_raw_k1)
        )

        self.register_buffer(
            "raw_K0",
            raw_k0,
        )
        self.register_buffer(
            "raw_K1",
            raw_k1,
        )

    def coefficients(self):
        return (
            F.softplus(
                self.raw_K0
            ),
            F.softplus(
                self.raw_K1
            ),
        )

    def forward(self, x):
        y = self.fno(x)
        K0, K1 = self.coefficients()

        return y, K0, K1


# ============================================================
# DDPM / Physics-DDPM
# ============================================================

class DiffusionSchedule:
    def __init__(
        self,
        cfg: TrainConfig,
        device,
    ):
        self.timesteps = int(
            cfg.diffusion_steps
        )

        self.betas = torch.linspace(
            cfg.diffusion_beta_start,
            cfg.diffusion_beta_end,
            self.timesteps,
            dtype=torch.float32,
            device=device,
        )

        self.alphas = (
            1.0
            - self.betas
        )

        self.alpha_bars = torch.cumprod(
            self.alphas,
            dim=0,
        )

    @staticmethod
    def gather(
        arr,
        t,
        x,
    ):
        return arr[t].reshape(
            x.shape[0],
            1,
            1,
            1,
        )


def q_sample(
    x0,
    t,
    noise,
    schedule: DiffusionSchedule,
):
    alpha_bar = schedule.gather(
        schedule.alpha_bars,
        t,
        x0,
    )

    return (
        torch.sqrt(
            alpha_bar
        )
        * x0
        + torch.sqrt(
            1.0 - alpha_bar
        )
        * noise
    )


def predict_x0(
    x_t,
    eps_pred,
    t,
    schedule: DiffusionSchedule,
):
    alpha_bar = schedule.gather(
        schedule.alpha_bars,
        t,
        x_t,
    )

    return (
        x_t
        - torch.sqrt(
            1.0 - alpha_bar
        )
        * eps_pred
    ) / torch.sqrt(
        alpha_bar + 1e-8
    )


def make_lr_up(
    x_lr,
    hr_shape=(1, 161, 161),
):
    _, H, W = hr_shape

    return F.interpolate(
        x_lr,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )


def estimate_train_diffusion_noise_std(
    data,
    cfg: TrainConfig,
    hr_shape=(1, 161, 161),
):
    """
    Estimate std(HR - LR_up) from a deterministic subset of the training split.

    The value is saved with DDPM checkpoints and can be reused to initialize
    """
    indices = np.asarray(
        data["train_idx"],
        dtype=np.int64,
    )

    max_samples = int(
        cfg.diffusion_noise_max_samples
    )

    if (
        max_samples > 0
        and len(indices) > max_samples
    ):
        pick = np.linspace(
            0,
            len(indices) - 1,
            max_samples,
        ).round().astype(np.int64)

        indices = indices[pick]

    sums = 0.0
    sums2 = 0.0
    count = 0

    batch = 32

    for start in range(
        0,
        len(indices),
        batch,
    ):
        idx = indices[
            start:start + batch
        ]

        x = torch.from_numpy(
            data["x_lr"][idx]
        ).float()

        y = torch.from_numpy(
            data["y_hr"][idx]
        ).float()

        lr_up = make_lr_up(
            x,
            hr_shape=hr_shape,
        )

        residual = (
            y - lr_up
        ).double()

        sums += float(
            residual.sum().item()
        )
        sums2 += float(
            (residual ** 2).sum().item()
        )
        count += int(
            residual.numel()
        )

    if count <= 1:
        return float(
            cfg.diffusion_init_noise_std
        )

    mean = sums / count

    variance = max(
        sums2 / count
        - mean * mean,
        1e-12,
    )

    return float(
        math.sqrt(variance)
    )


def sinusoidal_time_embedding(
    t,
    dim,
):
    half = dim // 2

    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(
            half,
            device=t.device,
        ).float()
        / max(
            half - 1,
            1,
        )
    )

    args = (
        t.float().unsqueeze(1)
        * freqs.unsqueeze(0)
    )

    emb = torch.cat(
        [
            torch.sin(args),
            torch.cos(args),
        ],
        dim=1,
    )

    if dim % 2 == 1:
        emb = F.pad(
            emb,
            (0, 1),
        )

    return emb


class DiffResBlock(nn.Module):
    def __init__(
        self,
        channels,
        temb_dim,
    ):
        super().__init__()

        self.temb = nn.Linear(
            temb_dim,
            channels,
        )

        self.block = ResBlock(
            channels
        )

    def forward(
        self,
        x,
        temb,
    ):
        return self.block(
            x
            + self.temb(
                temb
            )[:, :, None, None]
        )


class CondDiffusionUNet(nn.Module):
    def __init__(
        self,
        base=64,
        hr_shape=(1, 161, 161),
    ):
        super().__init__()

        _, H, W = hr_shape

        self.H = H
        self.W = W

        temb_dim = base * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(
                128,
                temb_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                temb_dim,
                temb_dim,
            ),
        )

        self.c1 = nn.Conv2d(
            2,
            base,
            3,
            padding=1,
        )

        self.r1 = DiffResBlock(
            base,
            temb_dim,
        )

        self.d1 = nn.Conv2d(
            base,
            base * 2,
            3,
            stride=2,
            padding=1,
        )

        self.r2 = DiffResBlock(
            base * 2,
            temb_dim,
        )

        self.d2 = nn.Conv2d(
            base * 2,
            base * 4,
            3,
            stride=2,
            padding=1,
        )

        self.r3 = DiffResBlock(
            base * 4,
            temb_dim,
        )

        self.b1 = DiffResBlock(
            base * 4,
            temb_dim,
        )

        self.b2 = DiffResBlock(
            base * 4,
            temb_dim,
        )

        self.u1_conv = nn.Conv2d(
            base * 4 + base * 2,
            base * 2,
            3,
            padding=1,
        )

        self.u1 = DiffResBlock(
            base * 2,
            temb_dim,
        )

        self.u2_conv = nn.Conv2d(
            base * 2 + base,
            base,
            3,
            padding=1,
        )

        self.u2 = DiffResBlock(
            base,
            temb_dim,
        )

        self.out = nn.Conv2d(
            base,
            1,
            3,
            padding=1,
        )

    def forward(
        self,
        x_noisy,
        x_lr,
        t,
    ):
        lr_up = F.interpolate(
            x_lr,
            size=(
                self.H,
                self.W,
            ),
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [
                x_noisy,
                lr_up,
            ],
            dim=1,
        )

        temb = self.time_mlp(
            sinusoidal_time_embedding(
                t,
                128,
            )
        )

        c1 = self.r1(
            F.silu(
                self.c1(x)
            ),
            temb,
        )

        d1 = self.r2(
            F.silu(
                self.d1(c1)
            ),
            temb,
        )

        d2 = self.r3(
            F.silu(
                self.d2(d1)
            ),
            temb,
        )

        b = self.b2(
            self.b1(
                d2,
                temb,
            ),
            temb,
        )

        u1 = F.interpolate(
            b,
            size=d1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        u1 = self.u1(
            F.silu(
                self.u1_conv(
                    torch.cat(
                        [u1, d1],
                        dim=1,
                    )
                )
            ),
            temb,
        )

        u2 = F.interpolate(
            u1,
            size=c1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        u2 = self.u2(
            F.silu(
                self.u2_conv(
                    torch.cat(
                        [u2, c1],
                        dim=1,
                    )
                )
            ),
            temb,
        )

        return self.out(
            u2
        )


class PhysicsDiffusionModel(nn.Module):
    def __init__(
        self,
        cfg: TrainConfig,
        hr_shape=(1, 161, 161),
    ):
        super().__init__()

        self.unet = CondDiffusionUNet(
            base=cfg.diffusion_base_channels,
            hr_shape=hr_shape,
        )

        raw_k0 = torch.tensor(
            float(cfg.physics_common_raw_k0)
        )
        raw_k1 = torch.tensor(
            float(cfg.physics_common_raw_k1)
        )

        self.register_buffer(
            "raw_K0",
            raw_k0,
        )
        self.register_buffer(
            "raw_K1",
            raw_k1,
        )

    def coefficients(self):
        return (
            F.softplus(
                self.raw_K0
            ),
            F.softplus(
                self.raw_K1
            ),
        )

    def forward(
        self,
        x_noisy,
        x_lr,
        t,
    ):
        return self.unet(
            x_noisy,
            x_lr,
            t,
        )


# ============================================================
# Physics
# ============================================================

def central_difference(
    x,
    axis,
    spacing,
):
    if axis == 2:
        xp = F.pad(
            x,
            (0, 0, 1, 1),
            mode="reflect",
        )

        return (
            xp[:, :, 2:, :]
            - xp[:, :, :-2, :]
        ) / (
            2.0 * spacing
        )

    if axis == 3:
        xp = F.pad(
            x,
            (1, 1, 0, 0),
            mode="reflect",
        )

        return (
            xp[:, :, :, 2:]
            - xp[:, :, :, :-2]
        ) / (
            2.0 * spacing
        )

    raise ValueError(
        "axis must be 2 or 3"
    )


def second_difference(
    x,
    axis,
    spacing,
):
    if axis == 2:
        xp = F.pad(
            x,
            (0, 0, 1, 1),
            mode="reflect",
        )

        return (
            xp[:, :, 2:, :]
            - 2.0 * xp[:, :, 1:-1, :]
            + xp[:, :, :-2, :]
        ) / (
            spacing ** 2
        )

    if axis == 3:
        xp = F.pad(
            x,
            (1, 1, 0, 0),
            mode="reflect",
        )

        return (
            xp[:, :, :, 2:]
            - 2.0 * xp[:, :, :, 1:-1]
            + xp[:, :, :, :-2]
        ) / (
            spacing ** 2
        )

    raise ValueError(
        "axis must be 2 or 3"
    )


def compute_delta_t(
    u,
    v,
    K,
    dx,
    dy,
    cfl=0.5,
    diffusion_cfl=0.20,
    max_delta_t=1.0,
):
    """
    Per-sample explicit time step satisfying both advection and diffusion
    stability constraints. This is the same numerical rule used by the
    test-time PDE consistency evaluation.
    """
    umax = torch.amax(
        torch.abs(u),
        dim=(1, 2, 3),
        keepdim=True,
    )

    vmax = torch.amax(
        torch.abs(v),
        dim=(1, 2, 3),
        keepdim=True,
    )

    velocity = torch.clamp(
        torch.maximum(
            umax,
            vmax,
        ),
        min=1e-6,
    )

    dt_adv = (
        float(cfl)
        * min(dx, dy)
        / velocity
    )

    kmax = torch.clamp(
        torch.amax(
            torch.abs(K),
            dim=(1, 2, 3),
            keepdim=True,
        ),
        min=1e-6,
    )

    dt_diff = (
        float(diffusion_cfl)
        * min(
            dx * dx,
            dy * dy,
        )
        / kmax
    )

    dt_cap = torch.full_like(
        dt_adv,
        float(max_delta_t),
    )

    return torch.minimum(
        torch.minimum(
            dt_adv,
            dt_diff,
        ),
        dt_cap,
    )


def diffusion_coefficient(
    u,
    v,
    K0,
    K1,
    dx,
    dy,
):
    dudx = central_difference(
        u,
        3,
        dx,
    )

    dudy = central_difference(
        u,
        2,
        dy,
    )

    dvdx = central_difference(
        v,
        3,
        dx,
    )

    dvdy = central_difference(
        v,
        2,
        dy,
    )

    shear = torch.sqrt(
        dudx ** 2
        + dudy ** 2
        + dvdx ** 2
        + dvdy ** 2
        + 1e-8
    )

    return (
        K0
        + K1
        * torch.log1p(
            shear
        )
    )


def pde_rhs(
    y,
    u,
    v,
    K,
    dx,
    dy,
):
    dydx = central_difference(
        y,
        3,
        dx,
    )

    dydy = central_difference(
        y,
        2,
        dy,
    )

    adv = (
        u * dydx
        + v * dydy
    )

    lap = (
        second_difference(
            y,
            3,
            dx,
        )
        + second_difference(
            y,
            2,
            dy,
        )
    )

    return (
        K * lap
        - adv
    )


def heun_step(
    y,
    u,
    v,
    K,
    cfg: TrainConfig,
):
    dt = compute_delta_t(
        u,
        v,
        K,
        cfg.dx_scaled,
        cfg.dy_scaled,
        cfg.cfl,
        cfg.diffusion_cfl,
        cfg.max_delta_t,
    )

    f0 = pde_rhs(
        y,
        u,
        v,
        K,
        cfg.dx_scaled,
        cfg.dy_scaled,
    )

    predictor = (
        y
        + dt * f0
    )

    f1 = pde_rhs(
        predictor,
        u,
        v,
        K,
        cfg.dx_scaled,
        cfg.dy_scaled,
    )

    return (
        y
        + 0.5
        * dt
        * (f0 + f1)
    )


def physics_raw_loss(
    y_true,
    y_pred,
    u,
    v,
    K0,
    K1,
    cfg: TrainConfig,
):
    """
    Same physical intent as the original implementation:

        K = K0 + K1 * log(1 + shear)

        true_next = Phi(y_true)
        pred_next = Phi(y_pred)

        raw loss = 0.10 * MSE(pred_next, true_next)

    PDE calculations are explicitly fp32.
    """

    y_true = y_true.float()
    y_pred = y_pred.float()
    u = u.float()
    v = v.float()
    K0 = K0.float()
    K1 = K1.float()

    K = diffusion_coefficient(
        u,
        v,
        K0,
        K1,
        cfg.dx_scaled,
        cfg.dy_scaled,
    )

    true_next = heun_step(
        y_true,
        u,
        v,
        K,
        cfg,
    )

    pred_next = heun_step(
        y_pred,
        u,
        v,
        K,
        cfg,
    )

    return (
        cfg.physics_clip_scale
        * torch.mean(
            (pred_next - true_next) ** 2
        )
    )


class PhysicsGradientBalancer:
    """Physics-loss gradient scaling."""

    def __init__(
        self,
        cfg: TrainConfig,
        mode: str = "linear",
    ):
        self.decay = float(
            cfg.physics_grad_balance_ema_decay
        )
        self.eps = float(
            cfg.physics_grad_balance_eps
        )
        self.min_scale = float(
            cfg.physics_grad_balance_min_scale
        )
        self.max_scale = float(
            cfg.physics_grad_balance_max_scale
        )
        self.match_ratio = float(
            cfg.physics_grad_match_ratio
        )

        mode = str(mode).strip().lower()

        if mode not in {
            "linear",
            "sqrt",
        }:
            raise ValueError(
                f"Unknown physics gradient balancing mode: {mode}"
            )

        self.mode = mode

        self.ema_scale: Optional[
            torch.Tensor
        ] = None

        self.steps = 0

    def __call__(
        self,
        reference_loss,
        physics_loss,
        prediction,
    ):
        g_ref = torch.autograd.grad(
            reference_loss,
            prediction,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]

        g_phy = torch.autograd.grad(
            physics_loss,
            prediction,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]

        ref_norm = torch.sqrt(
            torch.mean(
                g_ref.detach().float() ** 2
            )
        )

        phy_norm = torch.sqrt(
            torch.mean(
                g_phy.detach().float() ** 2
            )
        )

        ratio = (
            self.match_ratio
            * (
                ref_norm
                + self.eps
            )
            / (
                phy_norm
                + self.eps
            )
        )

        if self.mode == "sqrt":
            raw_scale = torch.sqrt(
                torch.clamp(
                    ratio,
                    min=0.0,
                )
            )
        else:
            raw_scale = ratio

        raw_scale = raw_scale.clamp(
            self.min_scale,
            self.max_scale,
        )

        if self.ema_scale is None:
            self.ema_scale = raw_scale.detach()
        else:
            self.ema_scale = (
                self.decay
                * self.ema_scale
                + (1.0 - self.decay)
                * raw_scale.detach()
            )

        scale = self.ema_scale.clamp(
            self.min_scale,
            self.max_scale,
        )

        flat_ref = g_ref.detach().float().reshape(
            g_ref.shape[0],
            -1,
        )
        flat_phy = g_phy.detach().float().reshape(
            g_phy.shape[0],
            -1,
        )

        dot = torch.sum(
            flat_ref * flat_phy,
            dim=1,
        )

        denom = (
            torch.linalg.vector_norm(
                flat_ref,
                dim=1,
            )
            * torch.linalg.vector_norm(
                flat_phy,
                dim=1,
            )
            + self.eps
        )

        cosine = torch.mean(
            dot / denom
        )

        self.steps += 1

        scaled_loss = (
            physics_loss.float()
            * scale.detach()
        )

        return scaled_loss, {
            "scale": float(
                scale.detach().cpu()
            ),
            "ref_grad_rms": float(
                ref_norm.detach().cpu()
            ),
            "phy_grad_rms": float(
                phy_norm.detach().cpu()
            ),
            "grad_cosine": float(
                cosine.detach().cpu()
            ),
        }

    def state_dict(self):
        return {
            "ema_scale": (
                None
                if self.ema_scale is None
                else float(
                    self.ema_scale
                    .detach()
                    .cpu()
                )
            ),
            "steps": int(
                self.steps
            ),
            "decay": self.decay,
            "eps": self.eps,
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
            "match_ratio": self.match_ratio,
            "mode": self.mode,
        }


def physics_ramp_factor(
    epoch: int,
    cfg: TrainConfig,
) -> float:
    warmup = max(
        0,
        int(
            cfg.physics_warmup_epochs
        ),
    )
    ramp = max(
        0,
        int(
            cfg.physics_ramp_epochs
        ),
    )

    if epoch <= warmup:
        return 0.0

    if ramp <= 0:
        return 1.0

    progress = (
        epoch - warmup
    ) / float(ramp)

    return float(
        np.clip(
            progress,
            0.0,
            1.0,
        )
    )


def physics_balanced_loss(
    y_true,
    y_pred,
    u,
    v,
    K0,
    K1,
    reference_loss,
    cfg: TrainConfig,
    balancer: PhysicsGradientBalancer,
):
    raw = physics_raw_loss(
        y_true,
        y_pred,
        u,
        v,
        K0,
        K1,
        cfg,
    )

    scaled, stats = balancer(
        reference_loss,
        raw,
        y_pred,
    )

    return (
        raw,
        scaled,
        stats,
    )


# ============================================================
# Checkpoint helpers
# ============================================================

def checkpoint_dir(
    cfg: TrainConfig,
    name: str,
) -> Path:
    p = (
        Path(cfg.output_dir)
        / "checkpoints"
        / name
    )

    p.mkdir(
        parents=True,
        exist_ok=True,
    )

    return p


def history_path(
    cfg: TrainConfig,
    name: str,
) -> Path:
    p = (
        Path(cfg.output_dir)
        / "history"
    )

    p.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        p
        / f"{name}.csv"
    )


def final_checkpoint_path(
    cfg: TrainConfig,
    name: str,
) -> Path:
    return (
        checkpoint_dir(
            cfg,
            name,
        )
        / "final.pt"
    )


def save_single_model_checkpoint(
    path,
    name,
    epoch,
    cfg,
    model,
    optimizer,
    history,
    extra=None,
):
    payload = {
        "model_name": name,
        "epoch": int(epoch),
        "config": asdict(cfg),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
    }

    if extra:
        payload.update(
            extra
        )

    torch.save(
        payload,
        path,
    )


def save_gan_checkpoint(
    path,
    name,
    epoch,
    cfg,
    generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    history,
    extra=None,
):
    payload = {
        "model_name": name,
        "epoch": int(epoch),
        "config": asdict(cfg),

        "generator_state": generator.state_dict(),
        "discriminator_state": discriminator.state_dict(),

        "generator_optimizer_state": g_optimizer.state_dict(),
        "discriminator_optimizer_state": d_optimizer.state_dict(),

        "history": history,
    }

    if extra:
        payload.update(
            extra
        )

    torch.save(
        payload,
        path,
    )


def should_skip(
    cfg: TrainConfig,
    name: str,
):
    final_path = final_checkpoint_path(
        cfg,
        name,
    )

    return (
        final_path.exists()
        and cfg.skip_existing_models
        and not cfg.force_retrain
    )


def maybe_save_single_epoch(
    cfg,
    name,
    epoch,
    model,
    optimizer,
    history,
    extra=None,
):
    every = int(
        cfg.save_every_epochs
    )

    if (
        every <= 0
        or epoch % every != 0
    ):
        return

    path = (
        checkpoint_dir(
            cfg,
            name,
        )
        / f"epoch_{epoch:03d}.pt"
    )

    save_single_model_checkpoint(
        path,
        name,
        epoch,
        cfg,
        model,
        optimizer,
        history,
        extra=extra,
    )


def maybe_save_gan_epoch(
    cfg,
    name,
    epoch,
    generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    history,
    extra=None,
):
    every = int(
        cfg.save_every_epochs
    )

    if (
        every <= 0
        or epoch % every != 0
    ):
        return

    path = (
        checkpoint_dir(
            cfg,
            name,
        )
        / f"epoch_{epoch:03d}.pt"
    )

    save_gan_checkpoint(
        path,
        name,
        epoch,
        cfg,
        generator,
        discriminator,
        g_optimizer,
        d_optimizer,
        history,
        extra=extra,
    )


# ============================================================
# FNO training
# ============================================================

def train_fno(
    data,
    cfg: TrainConfig,
    device,
):
    name = "fno"

    if should_skip(
        cfg,
        name,
    ):
        print(
            "[SKIP] fno:",
            final_checkpoint_path(
                cfg,
                name,
            ),
        )
        return

    print(
        "\n"
        + "=" * 90
        + "\nTRAIN: FNO\n"
        + "=" * 90
    )

    model = FNOGenerator(
        cfg
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=1e-6,
    )

    loader = make_loader(
        SupervisedWindDataset(
            data,
            data["train_idx"],
        ),
        cfg,
        shuffle=True,
    )

    history = []

    for epoch in range(
        1,
        cfg.epochs + 1,
    ):
        model.train()
        rows = []

        for x, y in loader:
            x = x.to(
                device,
                non_blocking=True,
            ).float()

            y = y.to(
                device,
                non_blocking=True,
            ).float()

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(
                x
            )

            rec_loss = F.l1_loss(
                pred,
                y,
            )

            rec_loss.backward()

            clip_grad(
                model.parameters(),
                cfg.grad_clip_norm,
            )

            optimizer.step()

            rows.append([
                float(
                    rec_loss
                    .detach()
                    .cpu()
                )
            ])

        m = mean_rows(
            rows
        )

        history.append([
            epoch,
            float(m[0]),
        ])

        print(
            f"[fno] "
            f"epoch {epoch:03d}/{cfg.epochs:03d} "
            f"| l1={m[0]:.6f}"
        )

        extra = {
            "fno_modes_h": cfg.fno_modes_h,
            "fno_modes_w": cfg.fno_modes_w,
            "fno_width": cfg.fno_width,
            "fno_layers": cfg.fno_layers,
        }

        maybe_save_single_epoch(
            cfg,
            name,
            epoch,
            model,
            optimizer,
            history,
            extra=extra,
        )

    save_single_model_checkpoint(
        final_checkpoint_path(
            cfg,
            name,
        ),
        name,
        cfg.epochs,
        cfg,
        model,
        optimizer,
        history,
        extra={
            "fno_modes_h": cfg.fno_modes_h,
            "fno_modes_w": cfg.fno_modes_w,
            "fno_width": cfg.fno_width,
            "fno_layers": cfg.fno_layers,
        },
    )

    save_csv(
        history_path(
            cfg,
            name,
        ),
        [
            "epoch",
            "reconstruction_l1",
        ],
        history,
    )


# ============================================================
# PINO training
# ============================================================

def train_pino(
    data,
    cfg: TrainConfig,
    device,
):
    name = "pino"

    if should_skip(
        cfg,
        name,
    ):
        print(
            "[SKIP] pino:",
            final_checkpoint_path(
                cfg,
                name,
            ),
        )
        return

    print(
        "\n"
        + "=" * 90
        + "\nTRAIN: PINO\n"
        + "=" * 90
    )

    model = PINOGenerator(
        cfg
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=cfg.learning_rate,
        weight_decay=1e-6,
    )

    loader = make_loader(
        PhysicsWindDataset(
            data,
            data["train_idx"],
        ),
        cfg,
        shuffle=True,
    )

    balancer = PhysicsGradientBalancer(
        cfg,
        mode="sqrt",
    )

    history = []

    for epoch in range(
        1,
        cfg.epochs + 1,
    ):
        model.train()
        rows = []

        ramp = physics_ramp_factor(
            epoch,
            cfg,
        )

        for x, y, u, v in loader:
            x = x.to(
                device,
                non_blocking=True,
            ).float()

            y = y.to(
                device,
                non_blocking=True,
            ).float()

            u = u.to(
                device,
                non_blocking=True,
            ).float()

            v = v.to(
                device,
                non_blocking=True,
            ).float()

            optimizer.zero_grad(
                set_to_none=True
            )

            pred, K0, K1 = model(
                x
            )

            rec_loss = F.l1_loss(
                pred,
                y,
            )

            reference_loss = (
                cfg.pino_rec_weight
                * rec_loss
            )

            if ramp > 0.0:
                (
                    phy_raw,
                    phy_scaled,
                    balance_stats,
                ) = physics_balanced_loss(
                    y,
                    pred,
                    u,
                    v,
                    K0,
                    K1,
                    reference_loss,
                    cfg,
                    balancer,
                )
            else:
                with torch.no_grad():
                    phy_raw = physics_raw_loss(
                        y,
                        pred.detach(),
                        u,
                        v,
                        K0.detach(),
                        K1.detach(),
                        cfg,
                    )

                phy_scaled = torch.zeros(
                    (),
                    device=device,
                )

                balance_stats = {
                    "scale": 0.0,
                    "ref_grad_rms": 0.0,
                    "phy_grad_rms": 0.0,
                    "grad_cosine": 0.0,
                }

            total = (
                reference_loss
                + ramp
                * cfg.pino_phys_weight
                * phy_scaled
            )

            total.backward()

            clip_grad(
                [
                    p
                    for p in model.parameters()
                    if p.requires_grad
                ],
                cfg.grad_clip_norm,
            )

            optimizer.step()

            rows.append([
                float(
                    total.detach().cpu()
                ),
                float(
                    rec_loss.detach().cpu()
                ),
                float(
                    phy_raw.detach().cpu()
                ),
                float(
                    phy_scaled.detach().cpu()
                ),
                float(ramp),
                float(
                    balance_stats["scale"]
                ),
                float(
                    balance_stats["ref_grad_rms"]
                ),
                float(
                    balance_stats["phy_grad_rms"]
                ),
                float(
                    balance_stats["grad_cosine"]
                ),
                float(
                    K0.detach().cpu()
                ),
                float(
                    K1.detach().cpu()
                ),
            ])

        m = mean_rows(
            rows
        )

        history.append([
            epoch,
            *m.tolist(),
        ])

        print(
            f"[pino] "
            f"epoch {epoch:03d}/{cfg.epochs:03d} "
            f"| total={m[0]:.6f} "
            f"| rec_l1={m[1]:.6f} "
            f"| phy_raw={m[2]:.6f} "
            f"phy_scaled={m[3]:.6f} "
            f"| ramp={m[4]:.3f} "
            f"scale={m[5]:.3e} "
            f"| grad_ref={m[6]:.3e} "
            f"grad_phy={m[7]:.3e} "
            f"cos={m[8]:+.3f} "
            f"| K0={m[9]:.4f} "
            f"K1={m[10]:.4f}"
        )

        extra = {
            "rec_weight": cfg.pino_rec_weight,
            "phys_weight": cfg.pino_phys_weight,
        }

        maybe_save_single_epoch(
            cfg,
            name,
            epoch,
            model,
            optimizer,
            history,
            extra=extra,
        )

    save_single_model_checkpoint(
        final_checkpoint_path(
            cfg,
            name,
        ),
        name,
        cfg.epochs,
        cfg,
        model,
        optimizer,
        history,
        extra={
            "rec_weight": cfg.pino_rec_weight,
            "phys_weight": cfg.pino_phys_weight,
        },
    )

    save_csv(
        history_path(
            cfg,
            name,
        ),
        [
            "epoch",
            "total",
            "reconstruction_l1",
            "physics_raw",
            "physics_scaled",
            "physics_ramp",
            "physics_grad_scale",
            "reference_grad_rms",
            "physics_grad_rms",
            "gradient_cosine",
            "K0",
            "K1",
        ],
        history,
    )


# ============================================================
# SR-GAN / PINNSR training
# ============================================================

def train_gan(
    data,
    cfg: TrainConfig,
    device,
    physics=False,
):
    name = (
        "pinnsr"
        if physics
        else "srgan"
    )

    if should_skip(
        cfg,
        name,
    ):
        print(
            f"[SKIP] {name}:",
            final_checkpoint_path(
                cfg,
                name,
            ),
        )
        return

    print(
        "\n"
        + "=" * 90
        + f"\nTRAIN: {name.upper()}\n"
        + "=" * 90
    )

    if physics:
        generator = PINNSRGenerator(
            cfg
        ).to(device)

        dataset = PhysicsWindDataset(
            data,
            data["train_idx"],
        )

        balancer = PhysicsGradientBalancer(
            cfg,
            mode="sqrt",
        )

        rec_weight = float(
            cfg.pinnsr_rec_weight
        )

        phys_weight = float(
            cfg.pinnsr_phys_weight
        )

    else:
        generator = SRGenerator(
            tanh_final=True
        ).to(device)

        dataset = SupervisedWindDataset(
            data,
            data["train_idx"],
        )

        balancer = None

        rec_weight = float(
            cfg.srgan_rec_weight
        )

        phys_weight = 0.0

    adv_weight = float(
        cfg.adv_weight
    )

    discriminator = Discriminator(
    ).to(device)

    g_optimizer = torch.optim.Adam(
        [
            p
            for p in generator.parameters()
            if p.requires_grad
        ],
        lr=cfg.learning_rate,
        betas=(0.5, 0.999),
    )

    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=cfg.learning_rate,
        betas=(0.5, 0.999),
    )

    loader = make_loader(
        dataset,
        cfg,
        shuffle=True,
    )

    bce_logits = nn.BCEWithLogitsLoss()

    history = []

    for epoch in range(
        1,
        cfg.epochs + 1,
    ):
        generator.train()
        discriminator.train()

        rows = []

        ramp = (
            physics_ramp_factor(
                epoch,
                cfg,
            )
            if physics
            else 0.0
        )

        for batch in loader:
            if physics:
                x, y, u, v = [
                    z.to(
                        device,
                        non_blocking=True,
                    ).float()
                    for z in batch
                ]
            else:
                x, y = [
                    z.to(
                        device,
                        non_blocking=True,
                    ).float()
                    for z in batch
                ]

                u = None
                v = None

            # ------------------------------------------------
            # Discriminator
            # ------------------------------------------------
            set_requires_grad(
                discriminator,
                True,
            )

            d_optimizer.zero_grad(
                set_to_none=True
            )

            with torch.no_grad():
                fake_detached = generator(
                    x
                )

                if isinstance(
                    fake_detached,
                    tuple,
                ):
                    fake_detached = (
                        fake_detached[0]
                    )

            real_logits = discriminator(
                y
            )

            fake_logits = discriminator(
                fake_detached
            )

            d_bce = (
                bce_logits(
                    real_logits,
                    torch.ones_like(
                        real_logits
                    ),
                )
                + bce_logits(
                    fake_logits,
                    torch.zeros_like(
                        fake_logits
                    ),
                )
            )

            gp = gradient_penalty(
                discriminator,
                y,
                fake_detached,
                device,
            )

            d_loss = (
                d_bce.float()
                + cfg.gp_weight
                * gp.float()
            )

            d_loss.backward()

            clip_grad(
                discriminator.parameters(),
                cfg.grad_clip_norm,
            )

            d_optimizer.step()

            # ------------------------------------------------
            # Generator
            # ------------------------------------------------
            set_requires_grad(
                discriminator,
                False,
            )

            g_optimizer.zero_grad(
                set_to_none=True
            )

            if physics:
                fake, K0, K1 = generator(
                    x
                )
            else:
                fake = generator(
                    x
                )

                K0 = None
                K1 = None

            fake_logits_for_g = discriminator(
                fake
            )

            adv_loss = bce_logits(
                fake_logits_for_g,
                torch.ones_like(
                    fake_logits_for_g
                ),
            )

            rec_loss = F.mse_loss(
                fake,
                y,
            )

            reference_loss = (
                adv_weight
                * adv_loss
                + rec_weight
                * rec_loss
            )

            if physics:
                if ramp > 0.0:
                    (
                        phy_raw,
                        phy_scaled,
                        balance_stats,
                    ) = physics_balanced_loss(
                        y,
                        fake,
                        u,
                        v,
                        K0,
                        K1,
                        reference_loss,
                        cfg,
                        balancer,
                    )
                else:
                    with torch.no_grad():
                        phy_raw = physics_raw_loss(
                            y,
                            fake.detach(),
                            u,
                            v,
                            K0.detach(),
                            K1.detach(),
                            cfg,
                        )

                    phy_scaled = torch.zeros(
                        (),
                        device=device,
                    )

                    balance_stats = {
                        "scale": 0.0,
                        "ref_grad_rms": 0.0,
                        "phy_grad_rms": 0.0,
                        "grad_cosine": 0.0,
                    }

                g_loss = (
                    reference_loss
                    + ramp
                    * phys_weight
                    * phy_scaled
                )
            else:
                phy_raw = torch.zeros(
                    (),
                    device=device,
                )
                phy_scaled = torch.zeros(
                    (),
                    device=device,
                )

                balance_stats = {
                    "scale": 0.0,
                    "ref_grad_rms": 0.0,
                    "phy_grad_rms": 0.0,
                    "grad_cosine": 0.0,
                }

                g_loss = reference_loss

            g_loss.backward()

            clip_grad(
                [
                    p
                    for p in generator.parameters()
                    if p.requires_grad
                ],
                cfg.grad_clip_norm,
            )

            g_optimizer.step()

            set_requires_grad(
                discriminator,
                True,
            )

            rows.append([
                float(
                    d_loss.detach().cpu()
                ),
                float(
                    d_bce.detach().cpu()
                ),
                float(
                    gp.detach().cpu()
                ),
                float(
                    g_loss.detach().cpu()
                ),
                float(
                    rec_loss.detach().cpu()
                ),
                float(
                    adv_loss.detach().cpu()
                ),
                float(
                    phy_raw.detach().cpu()
                ),
                float(
                    phy_scaled.detach().cpu()
                ),
                float(ramp),
                float(
                    balance_stats["scale"]
                ),
                float(
                    balance_stats["ref_grad_rms"]
                ),
                float(
                    balance_stats["phy_grad_rms"]
                ),
                float(
                    balance_stats["grad_cosine"]
                ),
                (
                    float(
                        K0.detach().cpu()
                    )
                    if K0 is not None
                    else np.nan
                ),
                (
                    float(
                        K1.detach().cpu()
                    )
                    if K1 is not None
                    else np.nan
                ),
            ])

        m = mean_rows(
            rows
        )

        history.append([
            epoch,
            *m.tolist(),
        ])

        if physics:
            print(
                f"[pinnsr] "
                f"epoch {epoch:03d}/{cfg.epochs:03d} "
                f"| d={m[0]:.6f} "
                f"(bce={m[1]:.6f}, gp={m[2]:.6f}) "
                f"g={m[3]:.6f} "
                f"| rec_mse={m[4]:.6f} "
                f"adv={m[5]:.6f} "
                f"| phy_raw={m[6]:.6f} "
                f"phy_scaled={m[7]:.6f} "
                f"| ramp={m[8]:.3f} "
                f"scale={m[9]:.3e} "
                f"| grad_ref={m[10]:.3e} "
                f"grad_phy={m[11]:.3e} "
                f"cos={m[12]:+.3f} "
                f"| K0={m[13]:.4f} "
                f"K1={m[14]:.4f}"
            )
        else:
            print(
                f"[srgan] "
                f"epoch {epoch:03d}/{cfg.epochs:03d} "
                f"| d={m[0]:.6f} "
                f"(bce={m[1]:.6f}, gp={m[2]:.6f}) "
                f"g={m[3]:.6f} "
                f"| rec_mse={m[4]:.6f} "
                f"adv={m[5]:.6f}"
            )

        extra = {
            "adv_weight": adv_weight,
            "rec_weight": rec_weight,
            "phys_weight": phys_weight,
        }

        maybe_save_gan_epoch(
            cfg,
            name,
            epoch,
            generator,
            discriminator,
            g_optimizer,
            d_optimizer,
            history,
            extra=extra,
        )

    final_extra = {
        "adv_weight": adv_weight,
        "rec_weight": rec_weight,
        "phys_weight": phys_weight,
    }

    save_gan_checkpoint(
        final_checkpoint_path(
            cfg,
            name,
        ),
        name,
        cfg.epochs,
        cfg,
        generator,
        discriminator,
        g_optimizer,
        d_optimizer,
        history,
        extra=final_extra,
    )

    save_csv(
        history_path(
            cfg,
            name,
        ),
        [
            "epoch",
            "discriminator_loss",
            "discriminator_bce",
            "gradient_penalty",
            "generator_loss",
            "reconstruction_mse",
            "adversarial",
            "physics_raw",
            "physics_scaled",
            "physics_ramp",
            "physics_grad_scale",
            "reference_grad_rms",
            "physics_grad_rms",
            "gradient_cosine",
            "K0",
            "K1",
        ],
        history,
    )


# ============================================================
# DDPM / Physics-DDPM training
# ============================================================

def train_ddpm(
    data,
    cfg: TrainConfig,
    device,
    physics=False,
):
    name = (
        "physics_ddpm"
        if physics
        else "ddpm"
    )

    if should_skip(
        cfg,
        name,
    ):
        print(
            f"[SKIP] {name}:",
            final_checkpoint_path(
                cfg,
                name,
            ),
        )
        return

    print(
        "\n"
        + "=" * 90
        + f"\nTRAIN: {name.upper()} "
        + "\n"
        + "=" * 90
    )

    if physics:
        model = PhysicsDiffusionModel(
            cfg
        ).to(device)

        dataset = PhysicsWindDataset(
            data,
            data["train_idx"],
        )

        balancer = PhysicsGradientBalancer(
            cfg,
            mode="linear",
        )
    else:
        model = CondDiffusionUNet(
            base=cfg.diffusion_base_channels
        ).to(device)

        dataset = SupervisedWindDataset(
            data,
            data["train_idx"],
        )

        balancer = None

    optimizer = torch.optim.Adam(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=cfg.learning_rate,
    )

    loader = make_loader(
        dataset,
        cfg,
        shuffle=True,
    )

    schedule = DiffusionSchedule(
        cfg,
        device,
    )

    if bool(
        cfg.diffusion_auto_noise_std
    ):
        diffusion_noise_std = (
            estimate_train_diffusion_noise_std(
                data,
                cfg,
            )
        )
    else:
        diffusion_noise_std = float(
            cfg.diffusion_init_noise_std
        )

    print(
        f"[{name}] diffusion init std="
        f"{diffusion_noise_std:.6f}"
    )

    history = []

    for epoch in range(
        1,
        cfg.epochs + 1,
    ):
        model.train()
        rows = []

        ramp = (
            physics_ramp_factor(
                epoch,
                cfg,
            )
            if physics
            else 0.0
        )

        for batch in loader:
            if physics:
                x, y, u, v = [
                    z.to(
                        device,
                        non_blocking=True,
                    ).float()
                    for z in batch
                ]
            else:
                x, y = [
                    z.to(
                        device,
                        non_blocking=True,
                    ).float()
                    for z in batch
                ]

                u = None
                v = None

            optimizer.zero_grad(
                set_to_none=True
            )

            bs = y.shape[0]

            t = torch.randint(
                low=0,
                high=cfg.diffusion_steps,
                size=(bs,),
                device=device,
                dtype=torch.long,
            )

            eps = torch.randn_like(
                y
            )

            # ------------------------------------------------
            # ------------------------------------------------
            lr_up = make_lr_up(
                x,
                hr_shape=y.shape[1:],
            )

            residual_0 = (
                y - lr_up
            )

            residual_t = q_sample(
                residual_0,
                t,
                eps,
                schedule,
            )

            # The U-Net still receives a wind-field-like state rather than a
            x_state = torch.clamp(
                lr_up + residual_t,
                -1.0,
                1.0,
            )

            eps_pred = model(
                x_state,
                x,
                t,
            )

            residual_0_pred = predict_x0(
                residual_t,
                eps_pred,
                t,
                schedule,
            )

            x0_pred = (
                lr_up
                + residual_0_pred
            )

            if cfg.diffusion_train_x0_clamp:
                x0_for_loss = torch.clamp(
                    x0_pred,
                    -1.0,
                    1.0,
                )
            else:
                x0_for_loss = x0_pred

            eps_loss = F.mse_loss(
                eps_pred,
                eps,
            )

            rec_loss = F.l1_loss(
                x0_for_loss,
                y,
            )

            denoise_loss = (
                cfg.ddpm_eps_weight
                * eps_loss
                + cfg.ddpm_rec_weight
                * rec_loss
            )

            if physics:
                K0, K1 = model.coefficients()

                if ramp > 0.0:
                    # Balance physics against the reconstruction gradient with
                    # respect to the reconstructed HR field. epsilon loss is a
                    # denoising-space objective and therefore is not used as
                    # the output-space reference gradient.
                    rec_reference = (
                        cfg.ddpm_rec_weight
                        * rec_loss
                    )

                    (
                        phy_raw,
                        phy_scaled,
                        balance_stats,
                    ) = physics_balanced_loss(
                        y,
                        x0_for_loss,
                        u,
                        v,
                        K0,
                        K1,
                        rec_reference,
                        cfg,
                        balancer,
                    )
                else:
                    with torch.no_grad():
                        phy_raw = physics_raw_loss(
                            y,
                            x0_for_loss.detach(),
                            u,
                            v,
                            K0.detach(),
                            K1.detach(),
                            cfg,
                        )

                    phy_scaled = torch.zeros(
                        (),
                        device=device,
                    )

                    balance_stats = {
                        "scale": 0.0,
                        "ref_grad_rms": 0.0,
                        "phy_grad_rms": 0.0,
                        "grad_cosine": 0.0,
                    }

                total = (
                    denoise_loss
                    + ramp
                    * cfg.physics_ddpm_phys_weight
                    * phy_scaled
                )

            else:
                K0 = None
                K1 = None

                phy_raw = torch.zeros(
                    (),
                    device=device,
                )

                phy_scaled = torch.zeros(
                    (),
                    device=device,
                )

                balance_stats = {
                    "scale": 0.0,
                    "ref_grad_rms": 0.0,
                    "phy_grad_rms": 0.0,
                    "grad_cosine": 0.0,
                }

                total = denoise_loss

            total.backward()

            clip_grad(
                [
                    p
                    for p in model.parameters()
                    if p.requires_grad
                ],
                cfg.grad_clip_norm,
            )

            optimizer.step()

            rows.append([
                float(
                    total.detach().cpu()
                ),
                float(
                    rec_loss.detach().cpu()
                ),
                float(
                    eps_loss.detach().cpu()
                ),
                float(
                    phy_raw.detach().cpu()
                ),
                float(
                    phy_scaled.detach().cpu()
                ),
                float(ramp),
                float(
                    balance_stats["scale"]
                ),
                float(
                    balance_stats["ref_grad_rms"]
                ),
                float(
                    balance_stats["phy_grad_rms"]
                ),
                float(
                    balance_stats["grad_cosine"]
                ),
                (
                    float(
                        K0.detach().cpu()
                    )
                    if K0 is not None
                    else np.nan
                ),
                (
                    float(
                        K1.detach().cpu()
                    )
                    if K1 is not None
                    else np.nan
                ),
            ])

        m = mean_rows(
            rows
        )

        history.append([
            epoch,
            *m.tolist(),
        ])

        if physics:
            print(
                f"[physics_ddpm] "
                f"epoch {epoch:03d}/{cfg.epochs:03d} "
                f"| total={m[0]:.6f} "
                f"| rec_l1={m[1]:.6f} "
                f"eps_mse={m[2]:.6f} "
                f"| phy_raw={m[3]:.6f} "
                f"phy_scaled={m[4]:.6f} "
                f"| ramp={m[5]:.3f} "
                f"scale={m[6]:.3e} "
                f"| grad_ref={m[7]:.3e} "
                f"grad_phy={m[8]:.3e} "
                f"cos={m[9]:+.3f} "
                f"| K0={m[10]:.4f} "
                f"K1={m[11]:.4f}"
            )
        else:
            print(
                f"[ddpm] "
                f"epoch {epoch:03d}/{cfg.epochs:03d} "
                f"| total={m[0]:.6f} "
                f"| rec_l1={m[1]:.6f} "
                f"eps_mse={m[2]:.6f}"
            )

        extra = {
            "eps_weight": cfg.ddpm_eps_weight,
            "rec_weight": cfg.ddpm_rec_weight,
            "phys_weight": (
                cfg.physics_ddpm_phys_weight
                if physics
                else 0.0
            ),
            "diffusion_noise_std": float(
                diffusion_noise_std
            ),
        }

        maybe_save_single_epoch(
            cfg,
            name,
            epoch,
            model,
            optimizer,
            history,
            extra=extra,
        )

    final_extra = {
        "eps_weight": cfg.ddpm_eps_weight,
        "rec_weight": cfg.ddpm_rec_weight,
        "phys_weight": (
            cfg.physics_ddpm_phys_weight
            if physics
            else 0.0
        ),
        "diffusion_noise_std": float(
            diffusion_noise_std
        ),
    }

    save_single_model_checkpoint(
        final_checkpoint_path(
            cfg,
            name,
        ),
        name,
        cfg.epochs,
        cfg,
        model,
        optimizer,
        history,
        extra=final_extra,
    )

    save_csv(
        history_path(
            cfg,
            name,
        ),
        [
            "epoch",
            "total",
            "reconstruction_l1",
            "epsilon_mse",
            "physics_raw",
            "physics_scaled",
            "physics_ramp",
            "physics_grad_scale",
            "reference_grad_rms",
            "physics_grad_rms",
            "gradient_cosine",
            "K0",
            "K1",
        ],
        history,
    )


# ============================================================
# Dispatcher
# ============================================================

def train_one(
    name,
    data,
    cfg,
    device,
):
    if name == "ddpm":
        train_ddpm(
            data,
            cfg,
            device,
            physics=False,
        )
        return

    if name == "physics_ddpm":
        train_ddpm(
            data,
            cfg,
            device,
            physics=True,
        )
        return

    if name == "fno":
        train_fno(
            data,
            cfg,
            device,
        )
        return

    if name == "pino":
        train_pino(
            data,
            cfg,
            device,
        )
        return

    if name == "srgan":
        train_gan(
            data,
            cfg,
            device,
            physics=False,
        )
        return

    if name == "pinnsr":
        train_gan(
            data,
            cfg,
            device,
            physics=True,
        )
        return

    raise ValueError(
        f"Unknown model: {name}"
    )



# ============================================================
# Multi-GPU worker
# ============================================================

def parse_gpu_ids(gpu_ids: str):
    ids = []

    for token in str(gpu_ids).split(","):
        token = token.strip()

        if not token:
            continue

        gpu_id = int(token)

        if gpu_id < 0:
            raise ValueError(
                f"Invalid GPU id: {gpu_id}"
            )

        ids.append(gpu_id)

    if not ids:
        raise ValueError(
            "No GPU ids were specified."
        )

    return ids


def validate_requested_gpus(
    gpu_ids,
):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "This script requires NVIDIA GPU training."
        )

    count = torch.cuda.device_count()

    for gpu_id in gpu_ids:
        if gpu_id >= count:
            raise RuntimeError(
                f"Requested cuda:{gpu_id}, "
                f"but only {count} CUDA device(s) are visible."
            )


def gpu_worker_thread(
    gpu_id: int,
    task_queue: queue.Queue,
    error_queue: queue.Queue,
    cfg_dict,
):
    """
    Jupyter-safe dual-GPU worker.

    A Python thread owns one CUDA device and trains one complete model at a
    time. Models are not split across GPUs. When a model finishes, the worker
    pulls the next model from the shared queue.

    Threads are used instead of multiprocessing.spawn because code executed
    directly in a Jupyter cell is not importable as a normal Python module;
    spawn therefore cannot unpickle a worker defined in __main__.
    """
    cfg = TrainConfig(
        **cfg_dict
    )

    device = torch.device(
        f"cuda:{gpu_id}"
    )

    try:
        torch.cuda.set_device(
            gpu_id
        )

        # Seed only the CUDA RNG owned by this worker.
        # DataLoader shuffling is independently seeded in make_loader().
        torch.cuda.manual_seed(
            cfg.seed
        )

        print(
            f"\n[GPU WORKER START] "
            f"cuda:{gpu_id} | "
            f"{torch.cuda.get_device_name(gpu_id)}",
            flush=True,
        )

        # Each worker loads the same NPZ read-only into its own CPU arrays.
        data = load_preprocessed_dataset(
            default_preprocessed_path(
                cfg
            )
        )

        while True:
            try:
                model_name = task_queue.get_nowait()
            except queue.Empty:
                break

            print(
                f"\n[GPU ASSIGN] "
                f"cuda:{gpu_id} -> {model_name}",
                flush=True,
            )

            try:
                train_one(
                    model_name,
                    data,
                    cfg,
                    device,
                )

                print(
                    f"[GPU COMPLETE] "
                    f"cuda:{gpu_id} -> {model_name}",
                    flush=True,
                )

            except Exception:
                error_queue.put(
                    (
                        gpu_id,
                        model_name,
                        traceback.format_exc(),
                    )
                )
                return

            finally:
                torch.cuda.empty_cache()
                task_queue.task_done()

        print(
            f"[GPU WORKER STOP] cuda:{gpu_id}",
            flush=True,
        )

    except Exception:
        error_queue.put(
            (
                gpu_id,
                "<worker-startup>",
                traceback.format_exc(),
            )
        )


# ============================================================
# CLI
# ============================================================

def apply_cli_overrides(
    cfg: TrainConfig,
):
    parser = argparse.ArgumentParser(
        description=(
            "TGRS reproduction training"
        )
    )

    parser.add_argument(
        "--models",
        default=None,
        help=(
            'Comma-separated model selection, e.g. "pino,pinnsr", '
            '"ddpm,physics_ddpm", or "all".'
        ),
    )

    parser.add_argument(
        "--model",
        choices=MODEL_NAMES + ["all"],
        default=None,
        help=argparse.SUPPRESS,
    )


    parser.add_argument(
        "--output_dir",
        default=None,
    )

    parser.add_argument(
        "--preprocessed_npz",
        default=None,
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--gpu_ids",
        default=None,
        help="Comma-separated CUDA ids, e.g. 0,1",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )


    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Retrain the selected models even when final.pt already exists; "
            "the newly completed run overwrites that model's final.pt."
        ),
    )

    parser.add_argument(
        "--single_gpu",
        action="store_true",
        help="Disable two-GPU queue and run sequentially on one GPU.",
    )

    args, _ = parser.parse_known_args()

    if args.models is not None:
        cfg.models = args.models
    elif args.model is not None:
        cfg.models = args.model


    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    if args.preprocessed_npz is not None:
        cfg.preprocessed_npz = args.preprocessed_npz

    if args.device is not None:
        cfg.device = args.device

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids

    if args.epochs is not None:
        cfg.epochs = int(
            args.epochs
        )

    if args.batch_size is not None:
        cfg.batch_size = int(
            args.batch_size
        )


    if args.force_retrain or args.overwrite:
        cfg.force_retrain = True
        cfg.overwrite_selected_models = True

    if args.single_gpu:
        cfg.parallel_gpu_training = False

    return cfg


# ============================================================
# Main
# ============================================================

def main():
    cfg = TrainConfig()
    cfg = apply_cli_overrides(
        cfg
    )

    if cfg.overwrite_selected_models:
        cfg.force_retrain = True

    set_seed(
        cfg.seed
    )

    Path(
        cfg.output_dir
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 90
    )
    print(
        "TGRS REPRODUCTION TRAINING"
    )
    print(
        "=" * 90
    )

    print(
        "seed:",
        cfg.seed,
    )
    print(
        "epochs:",
        cfg.epochs,
    )
    print(
        "batch_size:",
        cfg.batch_size,
    )
    print(
        "models:",
        cfg.models,
    )
    print(
        "overwrite_selected_models:",
        cfg.overwrite_selected_models,
    )
    print(
        "output_dir:",
        cfg.output_dir,
    )
    print(
        "preprocessed_npz:",
        default_preprocessed_path(
            cfg
        ),
    )

    cache_path = default_preprocessed_path(
        cfg
    )

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found: {cache_path}"
        )

    config_path = (
        Path(cfg.output_dir)
        / "run_config.json"
    )

    config_path.write_text(
        json.dumps(
            asdict(cfg),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    names = parse_model_selection(
        cfg.models
    )

    print(
        "[SELECTED MODELS]",
        ", ".join(
            names
        ),
    )

    if cfg.force_retrain:
        print(
            "[OVERWRITE] Existing final checkpoints for selected models "
            "will not be skipped and will be replaced on successful completion."
        )

    priority = [
        "physics_ddpm",
        "ddpm",
        "pinnsr",
        "srgan",
        "pino",
        "fno",
    ]

    names = [
        name
        for name in priority
        if name in names
    ]

    # --------------------------------------------------------
    # Two-GPU dynamic queue
    # --------------------------------------------------------
    if (
        cfg.parallel_gpu_training
        and len(names) > 1
    ):
        gpu_ids = parse_gpu_ids(
            cfg.gpu_ids
        )

        validate_requested_gpus(
            gpu_ids
        )

        if len(gpu_ids) < 2:
            print(
                "[MULTI-GPU] Only one GPU id was supplied. "
                "Falling back to single-GPU sequential training."
            )
        else:
            # User requested two GPUs; use the first two ids.
            gpu_ids = gpu_ids[:2]

            print(
                "[MULTI-GPU] Dynamic model queue enabled."
            )
            print(
                "[MULTI-GPU] GPUs:",
                ", ".join(
                    f"cuda:{i} "
                    f"({torch.cuda.get_device_name(i)})"
                    for i in gpu_ids
                ),
            )
            print(
                "[MULTI-GPU] Queue:",
                " -> ".join(
                    names
                ),
            )

            # Jupyter-safe shared queues.
            task_queue = queue.Queue()
            error_queue = queue.Queue()

            for name in names:
                task_queue.put(
                    name
                )

            cfg_dict = asdict(
                cfg
            )

            workers = []

            for gpu_id in gpu_ids:
                th = threading.Thread(
                    target=gpu_worker_thread,
                    args=(
                        gpu_id,
                        task_queue,
                        error_queue,
                        cfg_dict,
                    ),
                    name=f"cuda-{gpu_id}-worker",
                    daemon=False,
                )

                th.start()
                workers.append(
                    th
                )

            for th in workers:
                th.join()

            errors = []

            while not error_queue.empty():
                errors.append(
                    error_queue.get()
                )

            if errors:
                lines = [
                    "At least one GPU worker failed."
                ]

                for gpu_id, model_name, tb in errors:
                    lines.append(
                        f"\n--- cuda:{gpu_id} / {model_name} ---\n{tb}"
                    )

                raise RuntimeError(
                    "\n".join(lines)
                )

            print(
                "\n"
                + "=" * 90
            )
            print(
                "TWO-GPU TRAINING COMPLETE"
            )
            print(
                "=" * 90
            )

            for name in names:
                print(
                    f"{name}:",
                    final_checkpoint_path(
                        cfg,
                        name,
                    ),
                )

            return

    # --------------------------------------------------------
    # Single-GPU fallback / explicitly requested single model
    # --------------------------------------------------------
    requested_device = cfg.device

    if requested_device == "cuda":
        requested_device = "cuda:0"

    device = get_device(
        requested_device
    )

    print(
        "[SINGLE GPU] device:",
        device,
    )

    if device.type == "cuda":
        idx = (
            0
            if device.index is None
            else int(
                device.index
            )
        )

        print(
            "[SINGLE GPU] gpu:",
            torch.cuda.get_device_name(
                idx
            ),
        )

    data = load_preprocessed_dataset(
        cache_path
    )

    for name in names:
        set_seed(
            cfg.seed
        )

        train_one(
            name,
            data,
            cfg,
            device,
        )

    print(
        "\n"
        + "=" * 90
    )
    print(
        "TRAINING COMPLETE"
    )
    print(
        "=" * 90
    )

    for name in names:
        print(
            f"{name}:",
            final_checkpoint_path(
                cfg,
                name,
            ),
        )


if __name__ == "__main__":
    main()
