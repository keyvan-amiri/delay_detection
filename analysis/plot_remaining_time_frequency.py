#!/usr/bin/env python3
"""
Create a frequency plot (histogram) of remaining-time values from two .pt files.

This script is designed for DALSTM artifacts where remaining-time targets are in
`DALSTM_y_*.pt`. If `DALSTM_X_*.pt` paths are passed by mistake, the script
automatically tries the corresponding `DALSTM_y_*.pt` files in the same folder.
"""

from __future__ import annotations

import argparse
import pickle
import struct
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List


def _resolve_remaining_time_file(path: Path) -> Path:
    """Map X_* path to sibling y_* path when possible."""
    name = path.name
    if "DALSTM_X_" in name:
        candidate = path.with_name(name.replace("DALSTM_X_", "DALSTM_y_", 1))
        if candidate.exists():
            return candidate
    return path


def _load_with_torch_if_available(path: Path) -> List[float] | None:
    try:
        import torch  # type: ignore

        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        if hasattr(obj, "detach"):
            obj = obj.detach().cpu()
        if hasattr(obj, "reshape"):
            obj = obj.reshape(-1)
        if hasattr(obj, "tolist"):
            raw = obj.tolist()
            return [float(x) for x in raw]
        return None
    except Exception:
        return None


def _load_float_tensor_without_torch(path: Path) -> List[float]:
    """
    Fallback loader for torch-saved zip archives containing a single storage.

    Supports common float storages used in this project:
      - FloatStorage  (float32)
      - HalfStorage   (float16)
      - DoubleStorage (float64)
      - BFloat16Storage (interpreted approximately as float16 for plotting only)
    """
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is not a torch zip archive.")

    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        data_pkl = next((n for n in names if n.endswith("/data.pkl")), None)
        storage_blob = next((n for n in names if "/data/0" in n), None)
        byteorder_name = next((n for n in names if n.endswith("/byteorder")), None)
        if data_pkl is None or storage_blob is None or byteorder_name is None:
            raise ValueError(f"Could not locate storage metadata in {path}.")

        pkl_bytes = zf.read(data_pkl)
        raw = zf.read(storage_blob)
        byteorder = zf.read(byteorder_name).decode("utf-8").strip()

    text = pkl_bytes.decode("latin1", errors="ignore")
    if "FloatStorage" in text:
        fmt = "f"
        size = 4
    elif "DoubleStorage" in text:
        fmt = "d"
        size = 8
    elif "HalfStorage" in text:
        fmt = "e"
        size = 2
    elif "BFloat16Storage" in text:
        # Python's struct has no bfloat16 format; approximate using float16.
        fmt = "e"
        size = 2
    else:
        raise ValueError(f"Unsupported storage type in {path}.")

    if len(raw) % size != 0:
        raise ValueError(f"Corrupt storage length in {path}.")
    endian = "<" if byteorder == "little" else ">"
    return [float(x[0]) for x in struct.iter_unpack(f"{endian}{fmt}", raw)]


def load_remaining_times(path: Path) -> List[float]:
    resolved = _resolve_remaining_time_file(path)
    vals = _load_with_torch_if_available(resolved)
    if vals is None:
        vals = _load_float_tensor_without_torch(resolved)
    if not vals:
        raise ValueError(f"No values found in {resolved}.")
    return vals


def _hist_counts(values: Iterable[float], bins: int) -> tuple[list[int], float, float]:
    vals = list(values)
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        vmax = vmin + 1e-9
    width = (vmax - vmin) / bins
    counts = [0] * bins
    for v in vals:
        idx = int((v - vmin) / width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    return counts, vmin, vmax


def _hist_counts_in_range(
    values: Iterable[float], bins: int, vmin: float, vmax: float
) -> list[int]:
    vals = list(values)
    if vmax == vmin:
        vmax = vmin + 1e-9
    width = (vmax - vmin) / bins
    counts = [0] * bins
    for v in vals:
        idx = int((v - vmin) / width)
        if idx < 0:
            idx = 0
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    return counts


def write_png_histogram_two_distributions(
    values1: List[float],
    values2: List[float],
    out_png: Path,
    bins: int,
    title: str,
    label1: str,
    label2: str,
) -> None:
    """Create one PNG with two histogram distributions using matplotlib + seaborn."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        raise RuntimeError(
            "matplotlib and seaborn are required for PNG output. "
            "Install with: pip install matplotlib seaborn"
        ) from exc

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        values1,
        bins=bins,
        kde=False,
        color="#4682B4",
        edgecolor="black",
        alpha=0.45,
        label=label1,
        ax=ax,
    )
    sns.histplot(
        values2,
        bins=bins,
        kde=False,
        color="#DD8452",
        edgecolor="black",
        alpha=0.45,
        label=label2,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Remaining time (days)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read two .pt files and create a frequency plot for remaining-time values."
    )
    parser.add_argument("--pt1", required=True, help="First .pt file path.")
    parser.add_argument("--pt2", required=True, help="Second .pt file path.")
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Output directory for separate values and plot.",
    )
    parser.add_argument("--bins", type=int, default=40, help="Number of histogram bins.")
    parser.add_argument(
        "--prefix",
        default="remaining_time_train_val",
        help="Output file prefix.",
    )
    args = parser.parse_args()

    pt1 = Path(args.pt1).expanduser().resolve()
    pt2 = Path(args.pt2).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved1 = _resolve_remaining_time_file(pt1)
    resolved2 = _resolve_remaining_time_file(pt2)
    vals1 = load_remaining_times(pt1)
    vals2 = load_remaining_times(pt2)

    out_pkl_1 = out_dir / f"{args.prefix}_list1.pkl"
    out_txt_1 = out_dir / f"{args.prefix}_list1.txt"
    out_pkl_2 = out_dir / f"{args.prefix}_list2.pkl"
    out_txt_2 = out_dir / f"{args.prefix}_list2.txt"
    out_png = out_dir / f"{args.prefix}_histogram.png"
    out_csv = out_dir / f"{args.prefix}_frequency_table.csv"

    with out_pkl_1.open("wb") as f:
        pickle.dump(vals1, f)
    out_txt_1.write_text("\n".join(f"{v:.10g}" for v in vals1) + "\n", encoding="utf-8")
    with out_pkl_2.open("wb") as f:
        pickle.dump(vals2, f)
    out_txt_2.write_text("\n".join(f"{v:.10g}" for v in vals2) + "\n", encoding="utf-8")

    all_values = vals1 + vals2
    _, vmin, vmax = _hist_counts(all_values, bins=args.bins)
    counts_1 = _hist_counts_in_range(vals1, bins=args.bins, vmin=vmin, vmax=vmax)
    counts_2 = _hist_counts_in_range(vals2, bins=args.bins, vmin=vmin, vmax=vmax)
    bin_width = (vmax - vmin) / args.bins
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("bin_idx,bin_left,bin_right,count_list1,count_list2\n")
        for i, (c1, c2) in enumerate(zip(counts_1, counts_2)):
            left = vmin + i * bin_width
            right = vmin + (i + 1) * bin_width
            f.write(f"{i},{left:.10g},{right:.10g},{c1},{c2}\n")

    write_png_histogram_two_distributions(
        vals1,
        vals2,
        out_png=out_png,
        bins=args.bins,
        title="Remaining Time Frequency (Two Separate Distributions)",
        label1=resolved1.stem,
        label2=resolved2.stem,
    )

    print(f"Resolved input 1: {resolved1}")
    print(f"Resolved input 2: {resolved2}")
    print(f"Values: file1={len(vals1)}, file2={len(vals2)}")
    print(f"Wrote: {out_pkl_1}")
    print(f"Wrote: {out_txt_1}")
    print(f"Wrote: {out_pkl_2}")
    print(f"Wrote: {out_txt_2}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
