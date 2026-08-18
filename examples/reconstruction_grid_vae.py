"""Reconstruction grid for the diffusers VAE on a VAE-intended dataset.

Loads 64 deterministic images from the flowers102 training split through the
exact harness pipeline (download_dataset + load_split), encodes and decodes them
with the public diffusers SD-VAE (deterministic mean latents), prints the batch
MSE, and writes an 8x8 original/reconstruction grid as a PDF.

Usage:
    python examples/reconstruction_grid_vae.py [--dataset flowers102] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from image_benchmarks.datasets.hf_loader import download_dataset, load_split
from image_benchmarks.encoders.registry import build_encoder

CHECKPOINT_ID = "stabilityai/sd-vae-ft-mse"
CHECKPOINT_SHA256 = (
    "6bfd0395790b0bde85baee0d32a525d4d0b14fc5bfb1a9aaeb6fe563415e317d"
)
CACHE_DIR = EXAMPLES_DIR.parent / "data" / "huggingface"
BATCH_SIZE = 64
GRID_SIDE = 8


def build_tile_grid(tiles: np.ndarray, side: int = GRID_SIDE, pad: int = 2) -> np.ndarray:
    """Stack (side*side, H, W, C) tiles into one padded image grid."""
    padded = np.stack(
        [
            np.pad(tile, ((pad, pad), (pad, pad), (0, 0)), constant_values=1.0)
            for tile in tiles
        ]
    )
    rows = [np.hstack(padded[r * side : (r + 1) * side]) for r in range(side)]
    return np.vstack(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="flowers102")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=4, help="images per encode/decode pass to bound RAM")
    parser.add_argument("--output", type=Path, default=EXAMPLES_DIR / "reconstruction_vae_flowers256.pdf")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    print(f"device: {[str(d) for d in jax.devices()]}")
    print(f"dataset: {args.dataset} @ {args.resolution}")

    encoder = build_encoder(
        {
            "type": "vae",
            "implementation": "diffusers",
            "checkpoint_id": CHECKPOINT_ID,
            "sample_posterior": True,
            "expected_sha256": CHECKPOINT_SHA256,
        },
        (args.resolution, args.resolution, 3),
        seed=args.seed,
    )
    print(f"VAE loaded: {encoder.checkpoint_id} digest {encoder.checkpoint_sha256[:16]}...")

    manifest = download_dataset(
        args.dataset,
        CACHE_DIR,
        resolution=args.resolution,
        crop="center_square",
        offline=False,
    )
    iterator = load_split(
        manifest,
        "train",
        BATCH_SIZE,
        args.seed,
        shuffle=False,
        offline=False,
    )
    batch = next(iter(iterator))
    images = jnp.asarray(batch["image"], dtype=jnp.float32)
    if images.shape != (BATCH_SIZE, args.resolution, args.resolution, 3):
        raise ValueError(f"Unexpected batch shape {images.shape}")
    print(f"batch: {images.shape} in [{float(images.min())}, {float(images.max())}]")

    # Encode/decode in chunks to bound peak activation memory on CPU hosts.
    chunk = max(1, int(args.chunk))
    recon_chunks = []
    for start in range(0, BATCH_SIZE, chunk):
        piece = images[start : start + chunk]
        piece_mean, _ = encoder.encode_stats(piece)
        piece_recon = jnp.asarray(encoder.decode(piece_mean), dtype=jnp.float32)
        recon_chunks.append(piece_recon)
        print(f"  encoded/decoded images {start}:{start + len(piece)}")
    recon_np = np.concatenate([np.asarray(c) for c in recon_chunks], axis=0)
    orig_np = np.asarray(images)
    mse = float(np.mean((recon_np - orig_np) ** 2))
    print(f"batch MSE over {BATCH_SIZE} images: {mse:.6f}")
    orig_grid = build_tile_grid(orig_np)
    recon_grid = build_tile_grid(recon_np)

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    axes[0].imshow((orig_grid + 1.0) / 2.0)
    axes[0].set_title(f"Originals — {args.dataset} @ {args.resolution} (train)", fontsize=11)
    axes[1].imshow((recon_grid + 1.0) / 2.0)
    axes[1].set_title(f"Diffusers VAE reconstructions ({CHECKPOINT_ID})", fontsize=11)
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(f"MSE over batch of {BATCH_SIZE} images: {mse:.6f}", fontsize=13, y=1.0)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
