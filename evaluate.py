#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for the six downscaling models.

Metrics:
    - PSNR
    - SSIM
    - LPIPS
    - one-step PDE RMSE
    - typhoon-event spectral RMSE
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception as exc:
    raise ImportError(
        "scikit-image is required. Install with: pip install scikit-image"
    ) from exc

try:
    import lpips
except Exception as exc:
    raise ImportError(
        "lpips is required. Install with: pip install lpips"
    ) from exc


EPS = 1e-12

MODEL_ORDER = (
    "ddpm",
    "physics_ddpm",
    "fno",
    "pino",
    "srgan",
    "pinnsr",
)

MODEL_LABELS = {
    "ddpm": "DDPM",
    "physics_ddpm": "Physics-DDPM",
    "fno": "FNO",
    "pino": "PINO",
    "srgan": "SR-GAN",
    "pinnsr": "PINNSR",
}


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Training output directory containing checkpoints/.",
    )

    parser.add_argument(
        "--preprocessed_npz",
        type=str,
        default="",
        help=(
            "Optional explicit preprocessing cache. If omitted, the evaluator "
            "searches common locations."
        ),
    )

    parser.add_argument(
        "--ibtracs",
        type=str,
        default="./data/ibtracs.WP.list.v04r00.csv",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--diffusion_seed",
        type=int,
        default=1234,
    )

    # Jupyter/IPython launches the kernel with its own arguments, typically:
    #     -f /root/.local/share/jupyter/runtime/kernel-....json
    # Plain parser.parse_args() treats those as user CLI arguments and exits
    # with SystemExit: 2.  Keep all evaluator arguments, but ignore only the
    # kernel connection-file arguments injected by Jupyter.
    args, unknown = parser.parse_known_args()

    if unknown:
        filtered = []
        i = 0
        while i < len(unknown):
            token = unknown[i]

            if token == "-f":
                # Jupyter uses "-f <kernel-connection-file.json>"
                if i + 1 < len(unknown):
                    i += 2
                    continue

            if token.startswith("-f="):
                i += 1
                continue

            filtered.append(token)
            i += 1

        if filtered:
            parser.error(
                "unrecognized arguments: "
                + " ".join(filtered)
            )

    return args



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

    # Save intermediate checkpoints every N epochs.
    # 0 = only final.pt
    save_every_epochs: int = 10

    # Existing final checkpoint is skipped unless force_retrain=True.
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

    gpu_ids: str = "0,1"
    parallel_gpu_training: bool = True


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


# =============================================================================
# Data loading
# =============================================================================

def decode_json_scalar(value):
    if isinstance(value, np.ndarray):
        value = value.item()

    if isinstance(value, bytes):
        value = value.decode("utf-8")

    return json.loads(str(value))


def resolve_preprocessed_path(
    output_dir: Path,
    explicit_path: str,
) -> Path:
    candidates = []

    if explicit_path:
        candidates.append(
            Path(explicit_path)
        )

    candidates.extend([
        output_dir
        / "preprocessed_boxmean_wind_torch.npz",

        Path(
            "./outputs_pinnsr_v2_boxmean_torch/"
            "preprocessed_boxmean_wind_torch.npz"
        ),
    ])

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Preprocessed NPZ not found. Tried:\n"
        + "\n".join(
            f"  - {path}"
            for path in candidates
        )
    )


def normalize_time_strings(values):
    result = []

    for value in np.asarray(
        values
    ).ravel():
        if isinstance(
            value,
            bytes,
        ):
            value = value.decode(
                "utf-8"
            )

        result.append(
            str(value)
        )

    return np.asarray(
        result,
        dtype=str,
    )


def load_test_data(path: Path):
    z = np.load(
        path,
        allow_pickle=False,
    )

    required = {
        "x_lr",
        "y_hr",
        "u_hr",
        "v_hr",
        "time_iso",
        "lat",
        "lon",
        "test_idx",
    }

    missing = sorted(
        required.difference(
            z.files
        )
    )

    if missing:
        raise KeyError(
            f"Preprocessed NPZ missing keys: {missing}"
        )

    test_idx = z[
        "test_idx"
    ].astype(
        np.int64
    )

    data = {
        "x": z["x_lr"][
            test_idx
        ].astype(
            np.float32
        ),

        "y": z["y_hr"][
            test_idx
        ].astype(
            np.float32
        ),

        "u": z["u_hr"][
            test_idx
        ].astype(
            np.float32
        ),

        "v": z["v_hr"][
            test_idx
        ].astype(
            np.float32
        ),

        "time_iso": normalize_time_strings(
            z["time_iso"][
                test_idx
            ]
        ),

        "lat": z[
            "lat"
        ].astype(
            np.float64
        ),

        "lon": z[
            "lon"
        ].astype(
            np.float64
        ),
    }

    if "ws_scaler" in z.files:
        data["ws_scaler"] = (
            decode_json_scalar(
                z["ws_scaler"]
            )
        )
    else:
        data[
            "ws_scaler"
        ] = {}

    return data


# =============================================================================
# Checkpoint loading
# =============================================================================

def final_checkpoint(
    output_dir: Path,
    model_name: str,
) -> Path:
    path = (
        output_dir
        / "checkpoints"
        / model_name
        / "final.pt"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for {model_name}: {path}"
        )

    return path


def config_from_checkpoint(
    TRAIN,
    payload,
):
    cfg = TRAIN.TrainConfig()

    saved = payload.get(
        "config",
        {}
    )

    for key, value in saved.items():
        if hasattr(
            cfg,
            key,
        ):
            setattr(
                cfg,
                key,
                value,
            )

    return cfg


def build_model(
    TRAIN,
    model_name: str,
    cfg,
    hr_shape,
):
    if model_name == "srgan":
        return TRAIN.SRGenerator(
            hr_shape=hr_shape,
            tanh_final=True,
        )

    if model_name == "pinnsr":
        return TRAIN.PINNSRGenerator(
            cfg,
            hr_shape=hr_shape,
        )

    if model_name == "fno":
        return TRAIN.FNOGenerator(
            cfg,
            hr_shape=hr_shape,
        )

    if model_name == "pino":
        return TRAIN.PINOGenerator(
            cfg,
            hr_shape=hr_shape,
        )

    if model_name == "ddpm":
        return TRAIN.CondDiffusionUNet(
            base=cfg.diffusion_base_channels,
            hr_shape=hr_shape,
        )

    if model_name == "physics_ddpm":
        return TRAIN.PhysicsDiffusionModel(
            cfg,
            hr_shape=hr_shape,
        )

    raise ValueError(
        model_name
    )


def load_model_checkpoint(
    TRAIN,
    output_dir,
    model_name,
    device,
    hr_shape,
):
    path = final_checkpoint(
        output_dir,
        model_name,
    )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = config_from_checkpoint(
        TRAIN,
        payload,
    )

    model = build_model(
        TRAIN,
        model_name,
        cfg,
        hr_shape,
    )

    if model_name in {
        "srgan",
        "pinnsr",
    }:
        key = "generator_state"
    else:
        key = "model_state"

    if key not in payload:
        raise KeyError(
            f"{path} does not contain '{key}'. "
            f"Available keys: {list(payload.keys())}"
        )

    model.load_state_dict(
        payload[key],
        strict=True,
    )

    model.to(
        device
    )
    model.eval()

    return (
        model,
        cfg,
        payload,
        path,
    )


# =============================================================================
# Deterministic model inference
# =============================================================================

@torch.no_grad()
def predict_deterministic(
    model_name,
    model,
    x_np,
    device,
    batch_size,
):
    outputs = []

    for start in range(
        0,
        len(x_np),
        batch_size,
    ):
        x = torch.from_numpy(
            x_np[
                start:start
                + batch_size
            ]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        out = model(
            x
        )

        if isinstance(
            out,
            tuple,
        ):
            out = out[0]

        out = torch.clamp(
            out,
            -1.0,
            1.0,
        )

        outputs.append(
            out.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

    return np.concatenate(
        outputs,
        axis=0,
    )


# =============================================================================
# DDIM sampling
# =============================================================================

def make_step_indices(
    total_steps,
    sample_steps,
):
    sample_steps = max(
        1,
        min(
            int(sample_steps),
            int(total_steps),
        ),
    )

    values = torch.linspace(
        total_steps - 1,
        0,
        sample_steps,
    ).round().long()

    values = torch.unique_consecutive(
        values
    )

    if int(
        values[-1].item()
    ) != 0:
        values = torch.cat([
            values,
            torch.zeros(
                1,
                dtype=torch.long,
            ),
        ])

    return values


@torch.no_grad()
def predict_residual_ddim(
    TRAIN,
    model,
    cfg,
    payload,
    x_np,
    device,
    hr_shape,
    batch_size,
    ddim_steps,
    seed,
):
    """Deterministic eta=0 DDIM sampling."""

    diffusion_noise_std = float(
        payload.get(
            "diffusion_noise_std",
            getattr(
                cfg,
                "diffusion_init_noise_std",
                0.15,
            ),
        )
    )

    schedule = TRAIN.DiffusionSchedule(
        cfg,
        device,
    )

    step_indices = make_step_indices(
        schedule.timesteps,
        ddim_steps,
    ).to(
        device
    )

    generator = torch.Generator(
        device=device
    )
    generator.manual_seed(
        int(seed)
    )

    outputs = []

    for start in range(
        0,
        len(x_np),
        batch_size,
    ):
        x_lr = torch.from_numpy(
            x_np[
                start:start
                + batch_size
            ]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        lr_up = TRAIN.make_lr_up(
            x_lr,
            hr_shape=hr_shape,
        )

        residual = (
            diffusion_noise_std
            * torch.randn(
                lr_up.shape,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
        )

        for step_pos, t_value in enumerate(
            step_indices
        ):
            ti = int(
                t_value.item()
            )

            t = torch.full(
                (
                    len(x_lr),
                ),
                ti,
                device=device,
                dtype=torch.long,
            )

            x_state = torch.clamp(
                lr_up
                + residual,
                -1.0,
                1.0,
            )

            eps_pred = model(
                x_state,
                x_lr,
                t,
            )

            residual_0_raw = TRAIN.predict_x0(
                residual,
                eps_pred,
                t,
                schedule,
            )

            hr_0_raw = (
                lr_up
                + residual_0_raw
            )

            # Match training-time reconstructed-field clamp.
            if bool(
                cfg.diffusion_train_x0_clamp
            ):
                hr_0 = torch.clamp(
                    hr_0_raw,
                    -1.0,
                    1.0,
                )
            else:
                hr_0 = hr_0_raw

            residual_0 = (
                hr_0
                - lr_up
            )

            if (
                step_pos
                == len(
                    step_indices
                ) - 1
            ):
                residual = residual_0
                break

            t_next_value = int(
                step_indices[
                    step_pos + 1
                ].item()
            )

            t_next = torch.full(
                (
                    len(x_lr),
                ),
                t_next_value,
                device=device,
                dtype=torch.long,
            )

            alpha_bar_next = (
                schedule.gather(
                    schedule.alpha_bars,
                    t_next,
                    residual,
                )
            )

            # eta = 0 DDIM update.
            residual = (
                torch.sqrt(
                    alpha_bar_next
                )
                * residual_0
                + torch.sqrt(
                    torch.clamp(
                        1.0
                        - alpha_bar_next,
                        min=0.0,
                    )
                )
                * eps_pred
            )

        prediction = torch.clamp(
            lr_up
            + residual,
            -1.0,
            1.0,
        )

        outputs.append(
            prediction.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        done = min(
            start + batch_size,
            len(x_np),
        )

        print(
            f"    DDIM {done}/{len(x_np)}",
            end="\r",
            flush=True,
        )

    print(
        " " * 72,
        end="\r",
    )

    return np.concatenate(
        outputs,
        axis=0,
    )


# =============================================================================
# Image metrics
# =============================================================================

def psnr_per_frame(
    y_true,
    y_pred,
):
    mse = np.mean(
        (
            y_true.astype(
                np.float64
            )
            - y_pred.astype(
                np.float64
            )
        ) ** 2,
        axis=(
            1,
            2,
            3,
        ),
    )

    return (
        20.0
        * np.log10(
            2.0
        )
        - 10.0
        * np.log10(
            np.maximum(
                mse,
                EPS,
            )
        )
    )


def ssim_per_frame(
    y_true,
    y_pred,
):
    values = np.empty(
        len(y_true),
        dtype=np.float64,
    )

    for i in range(
        len(y_true)
    ):
        values[i] = (
            skimage_ssim(
                y_true[
                    i,
                    0,
                ].astype(
                    np.float32
                ),

                y_pred[
                    i,
                    0,
                ].astype(
                    np.float32
                ),

                data_range=2.0,
            )
        )

    return values


@torch.no_grad()
def lpips_per_frame(
    y_true,
    y_pred,
    metric_model,
    device,
    batch_size,
):
    values = []

    for start in range(
        0,
        len(y_true),
        batch_size,
    ):
        yt = torch.from_numpy(
            y_true[
                start:start
                + batch_size
            ]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        yp = torch.from_numpy(
            y_pred[
                start:start
                + batch_size
            ]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        yt = torch.clamp(
            yt,
            -1.0,
            1.0,
        )

        yp = torch.clamp(
            yp,
            -1.0,
            1.0,
        )

        if yt.shape[1] == 1:
            yt = yt.repeat(
                1,
                3,
                1,
                1,
            )

        if yp.shape[1] == 1:
            yp = yp.repeat(
                1,
                3,
                1,
                1,
            )

        score = metric_model(
            yt,
            yp,
        ).reshape(
            -1
        )

        values.append(
            score.detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

    return np.concatenate(
        values,
        axis=0,
    )


# =============================================================================
# Common PDE operator / one-step RMSE
# =============================================================================

def common_physics_config(
    TRAIN,
    checkpoint_payloads,
):
    physics_names = (
        "physics_ddpm",
        "pino",
        "pinnsr",
    )

    found = []

    for name in physics_names:
        if name not in checkpoint_payloads:
            continue

        payload = checkpoint_payloads[
            name
        ]

        cfg = config_from_checkpoint(
            TRAIN,
            payload,
        )

        values = (
            float(
                cfg.physics_common_raw_k0
            ),
            float(
                cfg.physics_common_raw_k1
            ),
            float(
                cfg.dx_scaled
            ),
            float(
                cfg.dy_scaled
            ),
            float(
                cfg.cfl
            ),
            float(
                cfg.diffusion_cfl
            ),
            float(
                cfg.max_delta_t
            ),
        )

        found.append(
            (
                name,
                cfg,
                values,
            )
        )

    if not found:
        raise RuntimeError(
            "No physics-informed checkpoint was available "
            "to define the common PDE operator."
        )

    reference_name, reference_cfg, reference_values = (
        found[0]
    )

    for name, _, values in found[1:]:
        if not np.allclose(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            np.asarray(
                reference_values,
                dtype=np.float64,
            ),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Physics checkpoints do not share the same PDE configuration: "
                f"{reference_name} vs {name}"
            )

    return reference_cfg


@torch.no_grad()
def pde_rmse_per_frame(
    TRAIN,
    y_true,
    y_pred,
    u_np,
    v_np,
    cfg,
    device,
    batch_size,
):
    """One-step physical-consistency error."""

    K0 = F.softplus(
        torch.tensor(
            float(
                cfg.physics_common_raw_k0
            ),
            device=device,
            dtype=torch.float32,
        )
    )

    K1 = F.softplus(
        torch.tensor(
            float(
                cfg.physics_common_raw_k1
            ),
            device=device,
            dtype=torch.float32,
        )
    )

    values = []

    for start in range(
        0,
        len(y_true),
        batch_size,
    ):
        gt = torch.from_numpy(
            y_true[
                start:start
                + batch_size
            ]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        pred = torch.from_numpy(
            y_pred[
                start:start
                + batch_size
            ]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        u = torch.from_numpy(
            u_np[
                start:start
                + batch_size
            ]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        v = torch.from_numpy(
            v_np[
                start:start
                + batch_size
            ]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        K = TRAIN.diffusion_coefficient(
            u,
            v,
            K0,
            K1,
            cfg.dx_scaled,
            cfg.dy_scaled,
        )

        gt_next = TRAIN.heun_step(
            gt,
            u,
            v,
            K,
            cfg,
        )

        pred_next = TRAIN.heun_step(
            pred,
            u,
            v,
            K,
            cfg,
        )

        error = torch.sqrt(
            torch.mean(
                (
                    pred_next
                    - gt_next
                ) ** 2,
                dim=(
                    1,
                    2,
                    3,
                ),
            )
            + EPS
        )

        values.append(
            error.detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

    return np.concatenate(
        values,
        axis=0,
    )


# =============================================================================
# Wind scaling
# =============================================================================

def inverse_wind_scaling(
    values,
    ws_scaler,
):
    if "log_max" in ws_scaler:
        log_max = float(
            ws_scaler[
                "log_max"
            ]
        )

    elif "vmax" in ws_scaler:
        log_max = float(
            np.log1p(
                float(
                    ws_scaler[
                        "vmax"
                    ]
                )
            )
        )

    else:
        raise KeyError(
            "ws_scaler must contain 'log_max' or 'vmax'."
        )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    z = 0.5 * (
        values + 1.0
    )

    physical = np.expm1(
        z * log_max
    )

    return np.maximum(
        physical,
        0.0,
    )


# =============================================================================
# IBTrACS / typhoon-frame matching
# =============================================================================

def choose_column(
    columns,
    candidates,
):
    mapping = {
        str(column).upper():
        column
        for column in columns
    }

    for candidate in candidates:
        result = mapping.get(
            candidate.upper()
        )

        if result is not None:
            return result

    return None


def load_ibtracs(
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"IBTrACS CSV not found: {path}"
        )

    raw = pd.read_csv(
        path,
        low_memory=False,
    )

    time_col = choose_column(
        raw.columns,
        (
            "ISO_TIME",
            "TIME",
            "DATETIME",
        ),
    )

    lat_col = choose_column(
        raw.columns,
        (
            "LAT",
            "USA_LAT",
            "WMO_LAT",
        ),
    )

    lon_col = choose_column(
        raw.columns,
        (
            "LON",
            "USA_LON",
            "WMO_LON",
        ),
    )

    sid_col = choose_column(
        raw.columns,
        (
            "SID",
            "USA_ATCF_ID",
            "ID",
        ),
    )

    name_col = choose_column(
        raw.columns,
        (
            "NAME",
            "STORM_NAME",
        ),
    )

    required = (
        time_col,
        lat_col,
        lon_col,
        sid_col,
    )

    if any(
        value is None
        for value in required
    ):
        raise KeyError(
            "Required IBTrACS columns are missing."
        )

    wind_columns = []

    for candidate in (
        "USA_WIND",
        "WMO_WIND",
        "TOKYO_WIND",
        "CMA_WIND",
        "HKO_WIND",
    ):
        column = choose_column(
            raw.columns,
            (
                candidate,
            ),
        )

        if column is not None:
            wind_columns.append(
                column
            )

    result = pd.DataFrame({
        "sid":
        raw[
            sid_col
        ].astype(
            str
        ).str.strip(),

        "storm_name":
        (
            raw[
                name_col
            ].astype(
                str
            ).str.strip()
            if name_col is not None
            else "UNKNOWN"
        ),

        "time":
        pd.to_datetime(
            raw[
                time_col
            ],
            errors="coerce",
            utc=True,
        ),

        "latitude":
        pd.to_numeric(
            raw[
                lat_col
            ],
            errors="coerce",
        ),

        "longitude":
        pd.to_numeric(
            raw[
                lon_col
            ],
            errors="coerce",
        ),
    })

    if wind_columns:
        winds = pd.concat(
            [
                pd.to_numeric(
                    raw[
                        column
                    ],
                    errors="coerce",
                )
                for column
                in wind_columns
            ],
            axis=1,
        )

        # Fixed source priority, matching the previous validation code.
        result[
            "wind_kt"
        ] = (
            winds
            .bfill(
                axis=1
            )
            .iloc[
                :,
                0,
            ]
        )

    else:
        result[
            "wind_kt"
        ] = np.nan

    result = result.dropna(
        subset=[
            "time",
            "latitude",
            "longitude",
        ]
    )

    result[
        "time"
    ] = (
        result[
            "time"
        ]
        .dt.tz_convert(
            None
        )
        .dt.floor(
            "s"
        )
    )

    result[
        "longitude"
    ] = np.mod(
        result[
            "longitude"
        ],
        360.0,
    )

    return (
        result
        .drop_duplicates(
            [
                "sid",
                "time",
            ]
        )
        .sort_values(
            [
                "sid",
                "time",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def longitude_in_domain(
    value,
    longitude,
):
    domain = np.asarray(
        longitude,
        dtype=np.float64,
    )

    if (
        np.nanmax(
            domain
        ) <= 180
        and value > 180
    ):
        value = (
            (
                value
                + 180
            )
            % 360
        ) - 180

    elif (
        np.nanmax(
            domain
        ) > 180
        and value < 0
    ):
        value = (
            value
            % 360
        )

    return (
        np.nanmin(
            domain
        )
        <= value
        <= np.nanmax(
            domain
        )
    )


def exact_typhoon_frames(
    test_times,
    latitude,
    longitude,
    ibtracs,
):
    test_table = pd.DataFrame({
        "frame_index":
        np.arange(
            len(
                test_times
            ),
            dtype=int,
        ),

        "frame_time":
        pd.DatetimeIndex(
            test_times
        ).floor(
            "s"
        ),
    })

    merged = test_table.merge(
        ibtracs,
        left_on="frame_time",
        right_on="time",
        how="inner",
    )

    inside = (
        merged[
            "latitude"
        ].between(
            np.nanmin(
                latitude
            ),
            np.nanmax(
                latitude
            ),
        )
        & merged[
            "longitude"
        ].apply(
            lambda value:
            longitude_in_domain(
                value,
                longitude,
            )
        )
    )

    merged = merged[
        inside
    ].copy()

    if merged.empty:
        raise RuntimeError(
            "No exact-time IBTrACS records were found "
            "inside the HR test domain."
        )

    # If more than one storm lies inside the domain at one timestamp,
    # retain the strongest available exact-time record for the frame mask.
    frame_metadata = (
        merged
        .sort_values(
            [
                "frame_index",
                "wind_kt",
            ],
            ascending=[
                True,
                False,
            ],
            na_position="last",
        )
        .drop_duplicates(
            "frame_index"
        )
        .reset_index(
            drop=True
        )
    )

    return frame_metadata


# =============================================================================
# GT-defined event mask
# =============================================================================

def nearest_grid_index(
    storm_latitude,
    storm_longitude,
    latitude,
    longitude,
):
    lon = float(
        storm_longitude
    )

    lon_values = np.asarray(
        longitude,
        dtype=np.float64,
    )

    if (
        np.nanmax(
            lon_values
        ) <= 180.0
        and lon > 180.0
    ):
        lon = (
            (
                lon
                + 180.0
            )
            % 360.0
        ) - 180.0

    elif (
        np.nanmax(
            lon_values
        ) > 180.0
        and lon < 0.0
    ):
        lon = lon % 360.0

    row = int(
        np.nanargmin(
            np.abs(
                np.asarray(
                    latitude
                )
                - storm_latitude
            )
        )
    )

    col = int(
        np.nanargmin(
            np.abs(
                lon_values
                - lon
            )
        )
    )

    return (
        row,
        col,
    )


def otsu_threshold(
    values,
):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(
            x
        )
    ]

    if x.size == 0:
        raise ValueError(
            "Cannot compute Otsu threshold from an empty field."
        )

    unique = np.unique(
        x
    )

    if unique.size == 1:
        return float(
            unique[
                0
            ]
        )

    histogram, edges = np.histogram(
        x,
        bins="auto",
    )

    histogram = histogram.astype(
        np.float64
    )

    centers = 0.5 * (
        edges[
            :-1
        ]
        + edges[
            1:
        ]
    )

    probability = (
        histogram
        / np.maximum(
            histogram.sum(),
            1.0,
        )
    )

    cumulative_probability = np.cumsum(
        probability
    )

    cumulative_mean = np.cumsum(
        probability
        * centers
    )

    global_mean = cumulative_mean[
        -1
    ]

    denominator = (
        cumulative_probability
        * (
            1.0
            - cumulative_probability
        )
    )

    variance = np.zeros_like(
        denominator
    )

    valid = (
        denominator
        > 0
    )

    variance[
        valid
    ] = (
        (
            global_mean
            * cumulative_probability[
                valid
            ]
            - cumulative_mean[
                valid
            ]
        ) ** 2
        / denominator[
            valid
        ]
    )

    index = int(
        np.nanargmax(
            variance
        )
    )

    return float(
        centers[
            index
        ]
    )


def select_typhoon_component(
    gt_field,
    center_row,
    center_col,
):
    field = np.asarray(
        gt_field,
        dtype=np.float64,
    )

    threshold = otsu_threshold(
        field
    )

    binary = (
        np.isfinite(
            field
        )
        & (
            field
            >= threshold
        )
    )

    labels, number = ndimage.label(
        binary
    )

    if number == 0:
        raise RuntimeError(
            "Otsu segmentation produced no connected component."
        )

    center_label = int(
        labels[
            center_row,
            center_col,
        ]
    )

    if center_label > 0:
        selected = center_label

    else:
        yy, xx = np.indices(
            field.shape
        )

        selected = None
        best_distance = np.inf
        best_peak = -np.inf

        for label_id in range(
            1,
            number + 1,
        ):
            component = (
                labels
                == label_id
            )

            if not np.any(
                component
            ):
                continue

            distance_squared = (
                (
                    yy[
                        component
                    ]
                    - center_row
                ) ** 2
                + (
                    xx[
                        component
                    ]
                    - center_col
                ) ** 2
            )

            minimum_distance = float(
                np.min(
                    distance_squared
                )
            )

            component_peak = float(
                np.nanmax(
                    field[
                        component
                    ]
                )
            )

            if (
                minimum_distance
                < best_distance
                or (
                    np.isclose(
                        minimum_distance,
                        best_distance,
                    )
                    and component_peak
                    > best_peak
                )
            ):
                best_distance = (
                    minimum_distance
                )
                best_peak = (
                    component_peak
                )
                selected = (
                    label_id
                )

        if selected is None:
            raise RuntimeError(
                "Could not associate a GT event component with IBTrACS."
            )

    return (
        labels
        == selected
    )


def apodize_event_mask(
    mask,
):
    binary = np.asarray(
        mask,
        dtype=bool,
    )

    distance = (
        ndimage
        .distance_transform_edt(
            binary
        )
    )

    maximum = float(
        np.max(
            distance
        )
    )

    if maximum <= 0:
        return binary.astype(
            np.float64
        )

    normalized = np.clip(
        distance
        / maximum,
        0.0,
        1.0,
    )

    weights = (
        np.sin(
            0.5
            * np.pi
            * normalized
        ) ** 2
    )

    weights[
        ~binary
    ] = 0.0

    return weights


def event_weighted_anomaly(
    field,
    event_weights,
):
    x = np.asarray(
        field,
        dtype=np.float64,
    )

    weights = np.asarray(
        event_weights,
        dtype=np.float64,
    )

    finite = np.isfinite(
        x
    )

    effective_weights = (
        weights
        * finite
    )

    weight_sum = float(
        np.sum(
            effective_weights
        )
    )

    if weight_sum <= EPS:
        raise RuntimeError(
            "GT-defined event mask has no valid weighted pixels."
        )

    weighted_mean = float(
        np.sum(
            np.nan_to_num(
                x
            )
            * effective_weights
        )
        / weight_sum
    )

    return (
        np.nan_to_num(
            x,
            nan=weighted_mean,
        )
        - weighted_mean
    ) * weights


# =============================================================================
# RAPSD / spectral error
# =============================================================================

def rapsd_pysteps_style(
    field,
):
    x = np.asarray(
        field,
        dtype=np.float64,
    )

    if x.ndim != 2:
        raise ValueError(
            f"RAPSD requires 2-D input, got {x.shape}"
        )

    if not np.all(
        np.isfinite(
            x
        )
    ):
        finite_mean = np.nanmean(
            x
        )

        x = np.where(
            np.isfinite(
                x
            ),
            x,
            finite_mean,
        )

    x = (
        x
        - np.mean(
            x
        )
    )

    fourier = np.fft.fftshift(
        np.fft.fft2(
            x
        )
    )

    power_2d = (
        np.abs(
            fourier
        ) ** 2
    )

    height, width = x.shape

    yy, xx = np.indices(
        (
            height,
            width,
        )
    )

    center_y = (
        height
        // 2
    )

    center_x = (
        width
        // 2
    )

    radius = np.sqrt(
        (
            yy
            - center_y
        ) ** 2
        + (
            xx
            - center_x
        ) ** 2
    )

    radius_index = radius.astype(
        int
    )

    maximum_radius = (
        min(
            height,
            width,
        )
        // 2
    )

    spectrum = np.full(
        maximum_radius,
        np.nan,
        dtype=np.float64,
    )

    for ring in range(
        maximum_radius
    ):
        mask = (
            radius_index
            == ring
        )

        if np.any(
            mask
        ):
            spectrum[
                ring
            ] = np.mean(
                power_2d[
                    mask
                ]
            )

    frequency = np.abs(
        np.fft.fftfreq(
            max(
                height,
                width,
            ),
            d=1.0,
        )[
            :maximum_radius
        ]
    )

    # Remove DC.
    spectrum = spectrum[
        1:
    ]

    frequency = frequency[
        1:
    ]

    valid = (
        np.isfinite(
            spectrum
        )
        & (
            spectrum
            >= 0
        )
    )

    return (
        spectrum[
            valid
        ],
        frequency[
            valid
        ],
    )


def log_spectral_rmse(
    gt_power,
    model_power,
    frequency,
):
    length = min(
        len(
            gt_power
        ),
        len(
            model_power
        ),
        len(
            frequency
        ),
    )

    gt = np.asarray(
        gt_power[
            :length
        ],
        dtype=np.float64,
    )

    pred = np.asarray(
        model_power[
            :length
        ],
        dtype=np.float64,
    )

    freq = np.asarray(
        frequency[
            :length
        ],
        dtype=np.float64,
    )

    valid = (
        np.isfinite(
            gt
        )
        & np.isfinite(
            pred
        )
        & np.isfinite(
            freq
        )
        & (
            gt
            > EPS
        )
        & (
            pred
            > EPS
        )
        & (
            freq
            > 0
        )
    )

    gt = gt[
        valid
    ]

    pred = pred[
        valid
    ]

    if len(
        gt
    ) < 3:
        return np.nan

    difference = (
        np.log10(
            pred
        )
        - np.log10(
            gt
        )
    )

    return float(
        np.sqrt(
            np.mean(
                difference ** 2
            )
        )
    )


def prepare_typhoon_event_info(
    y_true,
    test_times,
    latitude,
    longitude,
    ws_scaler,
    ibtracs_path,
):
    ibtracs = load_ibtracs(
        ibtracs_path
    )

    metadata = exact_typhoon_frames(
        test_times,
        latitude,
        longitude,
        ibtracs,
    )

    frame_indices = (
        metadata[
            "frame_index"
        ].to_numpy(
            dtype=int
        )
    )

    gt_physical = inverse_wind_scaling(
        y_true[
            frame_indices,
            0,
        ],
        ws_scaler,
    )

    info = []

    for local_index, (
        _,
        row,
    ) in enumerate(
        metadata.iterrows()
    ):
        center_row, center_col = (
            nearest_grid_index(
                float(
                    row[
                        "latitude"
                    ]
                ),
                float(
                    row[
                        "longitude"
                    ]
                ),
                latitude,
                longitude,
            )
        )

        gt_field = gt_physical[
            local_index
        ]

        event_mask = select_typhoon_component(
            gt_field,
            center_row,
            center_col,
        )

        weights = apodize_event_mask(
            event_mask
        )

        gt_event = event_weighted_anomaly(
            gt_field,
            weights,
        )

        gt_power, frequency = (
            rapsd_pysteps_style(
                gt_event
            )
        )

        info.append({
            "frame_index":
            int(
                row[
                    "frame_index"
                ]
            ),

            "sid":
            str(
                row[
                    "sid"
                ]
            ),

            "weights":
            weights,

            "gt_power":
            gt_power,

            "frequency":
            frequency,
        })

    return (
        metadata,
        info,
    )


def typhoon_spectral_error(
    y_pred,
    typhoon_info,
    ws_scaler,
):
    frame_indices = np.asarray(
        [
            item[
                "frame_index"
            ]
            for item
            in typhoon_info
        ],
        dtype=int,
    )

    pred_physical = inverse_wind_scaling(
        y_pred[
            frame_indices,
            0,
        ],
        ws_scaler,
    )

    rows = []

    for local_index, info in enumerate(
        typhoon_info
    ):
        pred_event = event_weighted_anomaly(
            pred_physical[
                local_index
            ],
            info[
                "weights"
            ],
        )

        pred_power, pred_frequency = (
            rapsd_pysteps_style(
                pred_event
            )
        )

        length = min(
            len(
                info[
                    "gt_power"
                ]
            ),
            len(
                pred_power
            ),
            len(
                info[
                    "frequency"
                ]
            ),
            len(
                pred_frequency
            ),
        )

        error = log_spectral_rmse(
            info[
                "gt_power"
            ][
                :length
            ],

            pred_power[
                :length
            ],

            info[
                "frequency"
            ][
                :length
            ],
        )

        rows.append({
            "sid":
            info[
                "sid"
            ],

            "error":
            error,
        })

    frame_df = pd.DataFrame(
        rows
    ).dropna()

    # Storm-balanced mean:
    # first average exact-matched frames inside each storm,
    # then give each storm equal weight.
    storm_mean = (
        frame_df
        .groupby(
            "sid"
        )[
            "error"
        ]
        .mean()
    )

    return float(
        storm_mean.mean()
    )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    ibtracs_path = Path(
        args.ibtracs
    )

    if (
        args.device.startswith(
            "cuda"
        )
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    device = torch.device(
        args.device
    )

    if device.type == "cuda":
        torch.cuda.set_device(
            device
        )

    # Standalone: model/PDE definitions are embedded above.
    TRAIN = sys.modules[__name__]

    preprocessed_path = (
        resolve_preprocessed_path(
            output_dir,
            args.preprocessed_npz,
        )
    )

    data = load_test_data(
        preprocessed_path
    )

    x_test = data[
        "x"
    ]

    y_test = data[
        "y"
    ]

    u_test = data[
        "u"
    ]

    v_test = data[
        "v"
    ]

    test_times = pd.to_datetime(
        data[
            "time_iso"
        ],
        errors="raise",
    )

    latitude = data[
        "lat"
    ]

    longitude = data[
        "lon"
    ]

    hr_shape = tuple(
        y_test.shape[
            1:
        ]
    )

    print(
        f"[DEVICE] {device}"
    )

    if device.type == "cuda":
        print(
            "[GPU]",
            torch.cuda.get_device_name(
                device
            ),
        )

    print(
        "[MODEL DEFINITIONS] embedded standalone evaluator"
    )

    print(
        "[PREPROCESSED]",
        preprocessed_path,
    )

    print(
        f"[TEST] N={len(y_test)} "
        f"LR={tuple(x_test.shape[2:])} "
        f"HR={tuple(y_test.shape[2:])}"
    )

    # Read all checkpoint payloads first so the PDE operator can be verified
    # before model-by-model evaluation.
    checkpoint_payloads = {}

    for model_name in MODEL_ORDER:
        path = final_checkpoint(
            output_dir,
            model_name,
        )

        checkpoint_payloads[
            model_name
        ] = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    pde_cfg = common_physics_config(
        TRAIN,
        checkpoint_payloads,
    )

    common_k0 = float(
        F.softplus(
            torch.tensor(
                float(
                    pde_cfg.physics_common_raw_k0
                )
            )
        ).item()
    )

    common_k1 = float(
        F.softplus(
            torch.tensor(
                float(
                    pde_cfg.physics_common_raw_k1
                )
            )
        ).item()
    )

    print(
        f"[COMMON PDE] "
        f"K0={common_k0:.8f} "
        f"K1={common_k1:.8f}"
    )

    typhoon_metadata, typhoon_info = (
        prepare_typhoon_event_info(
            y_test,
            test_times,
            latitude,
            longitude,
            data[
                "ws_scaler"
            ],
            ibtracs_path,
        )
    )

    print(
        f"[TYPHOON] "
        f"exact-matched frames={len(typhoon_metadata)} "
        f"storms={typhoon_metadata['sid'].nunique()}"
    )

    lpips_model = lpips.LPIPS(
        net="alex"
    ).to(
        device
    )

    lpips_model.eval()

    results = []

    for index, model_name in enumerate(
        MODEL_ORDER,
        start=1,
    ):
        print(
            "\n"
            + "=" * 88
        )

        print(
            f"[{index}/{len(MODEL_ORDER)}] "
            f"{MODEL_LABELS[model_name]}"
        )

        (
            model,
            cfg,
            payload,
            ckpt_path,
        ) = load_model_checkpoint(
            TRAIN,
            output_dir,
            model_name,
            device,
            hr_shape,
        )

        print(
            "[CKPT]",
            ckpt_path,
        )

        if model_name in {
            "ddpm",
            "physics_ddpm",
        }:
            prediction = predict_residual_ddim(
                TRAIN,
                model,
                cfg,
                payload,
                x_test,
                device,
                hr_shape,
                args.batch_size,
                args.ddim_steps,
                args.diffusion_seed,
            )

        else:
            prediction = predict_deterministic(
                model_name,
                model,
                x_test,
                device,
                args.batch_size,
            )

        prediction = np.clip(
            prediction.astype(
                np.float32
            ),
            -1.0,
            1.0,
        )

        psnr = psnr_per_frame(
            y_test,
            prediction,
        )

        ssim = ssim_per_frame(
            y_test,
            prediction,
        )

        lpips_values = lpips_per_frame(
            y_test,
            prediction,
            lpips_model,
            device,
            args.batch_size,
        )

        pde = pde_rmse_per_frame(
            TRAIN,
            y_test,
            prediction,
            u_test,
            v_test,
            pde_cfg,
            device,
            args.batch_size,
        )

        spectral = typhoon_spectral_error(
            prediction,
            typhoon_info,
            data[
                "ws_scaler"
            ],
        )

        row = {
            "model":
            MODEL_LABELS[
                model_name
            ],

            "PSNR":
            float(
                np.nanmean(
                    psnr
                )
            ),

            "SSIM":
            float(
                np.nanmean(
                    ssim
                )
            ),

            "LPIPS":
            float(
                np.nanmean(
                    lpips_values
                )
            ),

            "PDE_RMSE":
            float(
                np.nanmean(
                    pde
                )
            ),

            "SPECTRAL_RMSE":
            float(
                spectral
            ),
        }

        results.append(
            row
        )

        print(
            "  "
            f"PSNR={row['PSNR']:.4f} | "
            f"SSIM={row['SSIM']:.6f} | "
            f"LPIPS={row['LPIPS']:.6f} | "
            f"PDE_RMSE={row['PDE_RMSE']:.6f} | "
            f"Typhoon_Spectral_RMSE={row['SPECTRAL_RMSE']:.6f}"
        )

        del model
        del prediction

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        "\n"
        + "=" * 106
    )

    print(
        f"{'Model':<16}"
        f"{'PSNR':>13}"
        f"{'SSIM':>13}"
        f"{'LPIPS':>13}"
        f"{'PDE RMSE':>16}"
        f"{'Typhoon Spec.':>18}"
    )

    print(
        "-" * 106
    )

    for row in results:
        print(
            f"{row['model']:<16}"
            f"{row['PSNR']:>13.4f}"
            f"{row['SSIM']:>13.6f}"
            f"{row['LPIPS']:>13.6f}"
            f"{row['PDE_RMSE']:>16.6f}"
            f"{row['SPECTRAL_RMSE']:>18.6f}"
        )

    print(
        "=" * 106
    )

    # Compact paired deltas make the revised physics effect immediately visible.
    result_map = {
        row[
            "model"
        ]:
        row
        for row
        in results
    }

    pairs = (
        (
            "DDPM",
            "Physics-DDPM",
        ),
        (
            "FNO",
            "PINO",
        ),
        (
            "SR-GAN",
            "PINNSR",
        ),
    )

    print(
        "\n[PHYSICS PAIR DELTAS]"
    )

    print(
        "positive dPSNR/dSSIM = improvement; "
        "negative dLPIPS/dPDE/dSpec = improvement"
    )

    for baseline, physics in pairs:
        b = result_map[
            baseline
        ]

        p = result_map[
            physics
        ]

        print(
            f"{baseline:>12s} -> {physics:<12s} | "
            f"dPSNR={p['PSNR'] - b['PSNR']:+.4f} | "
            f"dSSIM={p['SSIM'] - b['SSIM']:+.6f} | "
            f"dLPIPS={p['LPIPS'] - b['LPIPS']:+.6f} | "
            f"dPDE={p['PDE_RMSE'] - b['PDE_RMSE']:+.6f} | "
            f"dSpec={p['SPECTRAL_RMSE'] - b['SPECTRAL_RMSE']:+.6f}"
        )


if __name__ == "__main__":
    main()
