#!/usr/bin/env python3
"""Render the heat example's outputs (out/) into figures (out/fig_*.png).

Needs matplotlib + Pillow.  Run after heat_source_inversion:
    python3 plot.py [--image ../IMG_3293.jpeg]

Figures: truth with observation points overlaid (full-resolution photo),
reconstruction, posterior samples, pointwise posterior std, the spectrum of
the prior-preconditioned misfit Hessian, and the QC-vs-probes curve.
"""
import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--image", default="../IMG_3293.jpeg")
parser.add_argument("--out", default="out")
args = parser.parse_args()
out = pathlib.Path(args.out)


def read_pgm(path):
    with Image.open(path) as im:
        return np.asarray(im, dtype=float) / 255.0


# --- truth + observation points over the full-resolution photo ----------
photo = Image.open(args.image).convert("L")
obs = np.loadtxt(out / "obs_points.txt")
W, H = photo.size
Lx = obs_hdr = None
with open(out / "obs_points.txt") as f:
    hdr = f.readline()
Lx = float(hdr.split("[0, ")[1].split("]")[0])
fig, ax = plt.subplots(figsize=(8, 8 * H / W), dpi=140)
ax.imshow(photo, cmap="gray", extent=[0, Lx, 0, 1], origin="upper")
ax.scatter(obs[:, 0], obs[:, 1], s=4, c="red", marker="o", linewidths=0,
           alpha=0.8)
ax.set_title(f"true source with {len(obs)} observation points")
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(out / "fig_truth_obs.png")
plt.close(fig)

# --- reconstruction / samples / std (upsampled smoothly) ----------------
def show_field(name, title, cmap="gray", vmin=0.0, vmax=1.0):
    a = read_pgm(out / f"{name}.pgm")
    fig, ax = plt.subplots(figsize=(8, 8 * a.shape[0] / a.shape[1]), dpi=140)
    ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bicubic")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / f"fig_{name}.png")
    plt.close(fig)


show_field("truth", "true source (mesh resolution)")
show_field("recon", "deterministic reconstruction  s*")
for i in range(16):
    p = out / f"sample_{i}.pgm"
    if p.exists():
        show_field(f"sample_{i}", f"posterior draw {i}")
show_field("std", "pointwise posterior std (own scale)", cmap="magma")

# --- spectrum ------------------------------------------------------------
eigs = np.loadtxt(out / "eigs.txt")
fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
ax.semilogy(eigs[:, 0], eigs[:, 1], lw=1.2)
ax.set_xlabel("mode index")
ax.set_ylabel(r"$\lambda_i$")
ax.set_title("spectrum of the prior-preconditioned misfit Hessian")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(out / "fig_eigs.png")
plt.close(fig)

# --- QC vs probes --------------------------------------------------------
qc = np.loadtxt(out / "qc_vs_k.txt")
qc = np.atleast_2d(qc)
fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
ax.plot(qc[:, 0], qc[:, 1], "o-", label="energy QC  (decides)")
ax.plot(qc[:, 0], qc[:, 2], "s--", label="row-mean QC  (legacy)")
ax.set_xlabel("probes  k")
ax.set_ylabel("held-out QC")
ax.set_title("fit quality vs probe count")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(out / "fig_qc_vs_k.png")
plt.close(fig)

print("figures written to", out)
