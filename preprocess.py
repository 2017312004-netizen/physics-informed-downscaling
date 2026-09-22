#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import netCDF4 as nc


@dataclass
class PreprocessConfig:
    nc_path: str = "./data/era5_2014.nc"
    output_path: str = "./outputs/preprocessed_boxmean_wind_torch.npz"

    lat_min: float = 14.0
    lat_max: float = 54.0
    lon_min: float = 114.0
    lon_max: float = 154.0

    train_start: str = "2014-01-01 00:00"
    train_end: str = "2014-09-01 00:00"
    val_start: str = "2014-09-01 00:00"
    val_end: str = "2014-10-01 00:00"
    test_start: str = "2014-10-01 00:00"
    test_end: str = "2015-01-01 00:00"

    lr_h: int = 32
    lr_w: int = 21


def parse_datetime(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


def netcdf_time_to_datetime(ds: nc.Dataset) -> np.ndarray:
    time_var = ds.variables["time"]
    units = time_var.units
    calendar = getattr(time_var, "calendar", "standard")
    t = nc.num2date(
        time_var[:],
        units=units,
        calendar=calendar,
    )
    return np.array([
        datetime(x.year, x.month, x.day, x.hour)
        for x in t
    ])


def crop_indices(coord: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    lo, hi = min(vmin, vmax), max(vmin, vmax)
    idx = np.where((coord >= lo) & (coord <= hi))[0]
    if len(idx) == 0:
        raise ValueError(f"No coordinate values found within [{lo}, {hi}]")
    return idx


def read_era5_wind(cfg: PreprocessConfig):
    print(f"[LOAD] {cfg.nc_path}")
    ds = nc.Dataset(cfg.nc_path, mode="r")

    lat_all = np.asarray(ds.variables["latitude"][:], dtype=np.float32)
    lon_all = np.asarray(ds.variables["longitude"][:], dtype=np.float32)
    times_all = netcdf_time_to_datetime(ds)

    t0 = parse_datetime(cfg.train_start)
    t1 = parse_datetime(cfg.test_end)
    tidx = np.where((times_all >= t0) & (times_all < t1))[0]

    lat_idx = crop_indices(lat_all, cfg.lat_min, cfg.lat_max)
    lon_idx = crop_indices(lon_all, cfg.lon_min, cfg.lon_max)

    if len(tidx) < 2:
        ds.close()
        raise ValueError("Selected time interval contains fewer than 2 timesteps.")

    u10_full = np.ma.filled(
        ds.variables["u10"][tidx, :, :],
        np.nan,
    ).astype(np.float32)
    v10_full = np.ma.filled(
        ds.variables["v10"][tidx, :, :],
        np.nan,
    ).astype(np.float32)

    u10 = u10_full[:, lat_idx, :][:, :, lon_idx]
    v10 = v10_full[:, lat_idx, :][:, :, lon_idx]
    lat = lat_all[lat_idx]
    lon = lon_all[lon_idx]
    times = times_all[tidx]
    ds.close()

    if lat[0] > lat[-1]:
        lat = lat[::-1].copy()
        u10 = u10[:, ::-1, :].copy()
        v10 = v10[:, ::-1, :].copy()

    print("[ERA5] u10:", u10.shape)
    print("[ERA5] v10:", v10.shape)
    print("[ERA5] lat:", lat.shape, f"{lat.min():.2f} ~ {lat.max():.2f}")
    print("[ERA5] lon:", lon.shape, f"{lon.min():.2f} ~ {lon.max():.2f}")
    print("[ERA5] time:", times[0], "~", times[-1])

    return {
        "u10": u10,
        "v10": v10,
        "lat": lat,
        "lon": lon,
        "time": times,
    }


def compute_wind_speed(u10, v10):
    return np.sqrt(np.square(u10) + np.square(v10)).astype(np.float32)


def fill_nan_per_frame(arr: np.ndarray):
    arr = arr.copy()
    for t in range(arr.shape[0]):
        if np.isnan(arr[t]).any():
            mean_val = float(np.nanmean(arr[t]))
            arr[t] = np.nan_to_num(arr[t], nan=mean_val)
    return arr


def box_mean_2d(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = img.shape
    y_edges = np.linspace(0, h, out_h + 1)
    x_edges = np.linspace(0, w, out_w + 1)
    out = np.empty((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        y0 = max(0, int(np.floor(y_edges[i])))
        y1 = min(h, int(np.ceil(y_edges[i + 1])))
        for j in range(out_w):
            x0 = max(0, int(np.floor(x_edges[j])))
            x1 = min(w, int(np.ceil(x_edges[j + 1])))
            out[i, j] = np.nanmean(img[y0:y1, x0:x1])
    return out


def box_mean_time(data: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    out = np.empty((data.shape[0], out_h, out_w), dtype=np.float32)
    for t in range(data.shape[0]):
        out[t] = box_mean_2d(data[t], out_h, out_w)
        if (t + 1) % 500 == 0 or (t + 1) == data.shape[0]:
            print(f"[BOX-MEAN] {t + 1}/{data.shape[0]}")
    return out


class WindSpeedLogScaler:
    def __init__(self, vmax: float):
        self.vmax = float(vmax)
        if not np.isfinite(self.vmax) or self.vmax <= 0:
            raise ValueError(f"Invalid vmax: {self.vmax}")
        self.log_max = float(np.log1p(self.vmax))

    @classmethod
    def fit(cls, train_ws_hr):
        return cls(float(np.nanmax(train_ws_hr)))

    def transform(self, ws):
        x = np.log1p(np.maximum(ws, 0.0)) / self.log_max
        return (2.0 * x - 1.0).astype(np.float32)

    def to_dict(self):
        return {"vmax": self.vmax, "log_max": self.log_max}


class UVScaler:
    def __init__(self, max_abs: float):
        self.max_abs = float(max_abs)
        if not np.isfinite(self.max_abs) or self.max_abs <= 0:
            raise ValueError(f"Invalid max_abs: {self.max_abs}")

    @classmethod
    def fit(cls, train_u, train_v):
        max_abs = max(
            float(np.nanmax(np.abs(train_u))),
            float(np.nanmax(np.abs(train_v))),
        )
        return cls(max_abs)

    def transform(self, arr):
        return (arr / self.max_abs).astype(np.float32)

    def to_dict(self):
        return {"max_abs": self.max_abs}


def build_preprocessed_dataset(cfg: PreprocessConfig):
    full = read_era5_wind(cfg)

    u10 = fill_nan_per_frame(full["u10"])
    v10 = fill_nan_per_frame(full["v10"])
    lat = full["lat"]
    lon = full["lon"]
    times = full["time"]

    ws_hr = compute_wind_speed(u10, v10)
    ws_lr = box_mean_time(ws_hr, cfg.lr_h, cfg.lr_w)

    train_idx = np.where(
        (times >= parse_datetime(cfg.train_start))
        & (times < parse_datetime(cfg.train_end))
    )[0]
    val_idx = np.where(
        (times >= parse_datetime(cfg.val_start))
        & (times < parse_datetime(cfg.val_end))
    )[0]
    test_idx = np.where(
        (times >= parse_datetime(cfg.test_start))
        & (times < parse_datetime(cfg.test_end))
    )[0]

    if len(train_idx) < 2 or len(val_idx) < 2 or len(test_idx) < 2:
        raise ValueError(
            "Invalid split sizes: "
            f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )

    ws_scaler = WindSpeedLogScaler.fit(ws_hr[train_idx])
    uv_scaler = UVScaler.fit(u10[train_idx], v10[train_idx])

    x_lr = ws_scaler.transform(ws_lr)[:, None, :, :]
    y_hr = ws_scaler.transform(ws_hr)[:, None, :, :]
    u_hr = uv_scaler.transform(u10)[:, None, :, :]
    v_hr = uv_scaler.transform(v10)[:, None, :, :]

    data = {
        "x_lr": x_lr,
        "y_hr": y_hr,
        "u_hr": u_hr,
        "v_hr": v_hr,
        "time_iso": np.array([t.isoformat() for t in times]),
        "lat": lat,
        "lon": lon,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "ws_scaler": ws_scaler.to_dict(),
        "uv_scaler": uv_scaler.to_dict(),
    }

    print("[DATA] x_lr:", x_lr.shape)
    print("[DATA] y_hr:", y_hr.shape)
    print(
        "[DATA] split:",
        f"train={len(train_idx)}",
        f"val={len(val_idx)}",
        f"test={len(test_idx)}",
    )
    return data


def save_preprocessed_dataset(data, path: Path, cfg: PreprocessConfig):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_lr=data["x_lr"],
        y_hr=data["y_hr"],
        u_hr=data["u_hr"],
        v_hr=data["v_hr"],
        time_iso=data["time_iso"],
        lat=data["lat"],
        lon=data["lon"],
        train_idx=data["train_idx"],
        val_idx=data["val_idx"],
        test_idx=data["test_idx"],
        ws_scaler=json.dumps(data["ws_scaler"]),
        uv_scaler=json.dumps(data["uv_scaler"]),
        config=json.dumps(asdict(cfg), ensure_ascii=False),
    )
    print("[SAVE]", path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--lat_min", type=float, default=None)
    parser.add_argument("--lat_max", type=float, default=None)
    parser.add_argument("--lon_min", type=float, default=None)
    parser.add_argument("--lon_max", type=float, default=None)
    parser.add_argument("--lr_h", type=int, default=None)
    parser.add_argument("--lr_w", type=int, default=None)
    return parser.parse_args()


def main():
    cfg = PreprocessConfig()
    args = parse_args()

    for name in ("nc_path", "lat_min", "lat_max", "lon_min", "lon_max", "lr_h", "lr_w"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)

    if args.output is not None:
        cfg.output_path = args.output

    data = build_preprocessed_dataset(cfg)
    save_preprocessed_dataset(data, Path(cfg.output_path), cfg)


if __name__ == "__main__":
    main()
