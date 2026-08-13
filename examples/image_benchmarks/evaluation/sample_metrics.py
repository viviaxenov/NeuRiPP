"""Distribution metrics evaluated directly in Flow Matching model state space."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from neuripp.functionals.MMD import bandwidth_median, gaussian_mmd


def evaluate_sample_metrics(
    real_states: Any,
    generated_states: Any,
    config: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Evaluate configured metrics on equally sized real and generated states."""

    real = jnp.asarray(real_states).reshape(len(real_states), -1)
    generated = jnp.asarray(generated_states).reshape(len(generated_states), -1)
    if real.shape != generated.shape or real.shape[0] < 2:
        raise ValueError(
            "Sample metrics require equally shaped real/generated arrays with at least two samples"
        )
    result: dict[str, Any] = {
        "sample_metrics_num_samples": int(real.shape[0]),
        "sample_metrics_state_dimension": int(real.shape[1]),
    }
    mmd = config["mmd"]
    if mmd["enabled"]:
        if "bandwidths" in mmd:
            bandwidths = jnp.asarray(mmd["bandwidths"], dtype=real.dtype)
            source = "explicit"
            median = None
        else:
            median_value = bandwidth_median(real)
            if not bool(jnp.isfinite(median_value)) or not float(median_value) > 0.0:
                raise ValueError(
                    "MMD median bandwidth is not positive and finite; configure explicit bandwidths"
                )
            bandwidths = median_value * jnp.asarray(
                mmd["bw_multipliers"], dtype=real.dtype
            )
            source = "median_multipliers"
            median = float(median_value)
        result.update(
            {
                "mmd": float(gaussian_mmd(real, generated, bandwidths)),
                "mmd_bandwidths": np.asarray(bandwidths).tolist(),
                "mmd_bandwidth_source": source,
                "mmd_bandwidth_median": median,
            }
        )
    sliced = config["sliced_wasserstein"]
    if sliced["enabled"]:
        from ott.tools.sliced import sliced_wasserstein

        distance, _ = sliced_wasserstein(
            real,
            generated,
            n_proj=sliced["num_projections"],
            rng=jax.random.key(seed),
        )
        result.update(
            {
                "sliced_wasserstein": float(distance),
                "sliced_wasserstein_num_projections": sliced["num_projections"],
            }
        )
    return result
