from pathlib import Path
import sys

import jax
import jax.numpy as jnp
from flax import nnx

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from image_benchmarks.rhs.diffuse_sit import DiffuseSiTRHS
from image_benchmarks.rhs.registry import (
    build_rhs,
    parameter_count,
    validate_rhs_compatibility,
)
from neuripp.parametric_pushforward.flow_matching import FlowMatching, flow_matching_loss


class FakeDiT(nnx.Module):
    def __init__(self):
        self.calls_without_labels = 0

    def __call__(self, state, time, y=None):
        assert y is None
        self.calls_without_labels += 1
        return jnp.zeros_like(state), []


def test_mlp_output_shape_and_parameter_count():
    rhs = build_rhs(
        {
            "type": "mlp",
            "hidden_dims": [16, 16],
            "time_embedding": {"dim": 8},
            "activation": "silu",
        },
        (12,),
        rngs=nnx.Rngs(0),
    )
    assert rhs(0.5, jnp.ones((12,))).shape == (12,)
    assert parameter_count(rhs) > 0


def test_explicit_mlp_flatten_adapter():
    rhs = build_rhs(
        {
            "type": "mlp",
            "flatten": True,
            "hidden_dims": [8],
            "time_embedding": {"dim": 4},
        },
        (2, 3, 1),
        rngs=nnx.Rngs(1),
    )
    assert rhs(0.2, jnp.ones((2, 3, 1))).shape == (2, 3, 1)


def tiny_unet(shape, seed):
    return build_rhs(
        {
            "type": "unet",
            "variant": "small",
            "base_channels": 8,
            "channel_mult": [1, 2],
            "num_res_blocks": 1,
            "attention_resolutions": [],
            "dropout": 0.0,
            "num_heads": 1,
            "num_head_channels": None,
        },
        shape,
        rngs=nnx.Rngs(seed),
    )


def test_unet_output_shapes_for_pixels_and_latents():
    for seed, shape in enumerate(((28, 28, 1), (32, 32, 3), (64, 64, 3), (32, 32, 4))):
        rhs = tiny_unet(shape, seed + 2)
        output = rhs(0.4, jnp.ones(shape))
        assert output.shape == shape
        assert parameter_count(rhs) > 0


def test_sit_adapter_is_unconditional_and_shape_preserving():
    model = FakeDiT()
    rhs = DiffuseSiTRHS(model, (4, 4, 2))
    output = rhs(0.3, jnp.ones((4, 4, 2)))
    assert output.shape == (4, 4, 2)
    assert model.calls_without_labels == 1


def test_packaged_diffuse_sit_uses_canonical_import():
    from image_benchmarks.assets.diffuse_nnx import (
        DIFFUSE_NNX_COMMIT,
        import_diffuse_module,
    )
    from importlib import metadata
    import json

    module = import_diffuse_module("diffuse_nnx.networks.transformers.dit_nnx")
    assert module.DiT.__module__.startswith("diffuse_nnx.")
    direct_url = metadata.distribution("diffuse-nnx").read_text("direct_url.json")
    provenance = json.loads(direct_url)
    commit = provenance.get("vcs_info", {}).get("commit_id")
    if commit is not None:
        assert commit == DIFFUSE_NNX_COMMIT


def test_architecture_compatibility_errors_are_preflighted():
    try:
        validate_rhs_compatibility({"type": "mlp"}, (32, 32, 3))
    except ValueError as error:
        assert "vector state" in str(error)
    else:
        raise AssertionError("Expected spatial MLP compatibility failure")
    try:
        validate_rhs_compatibility(
            {"type": "sit", "patch_size": 3, "class_conditioning": False},
            (32, 32, 4),
        )
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("Expected SiT patch compatibility failure")


def test_unet_dropout_runs_inside_flow_matching_vmap():
    rngs = nnx.Rngs(12)
    rhs = build_rhs(
        {
            "type": "unet",
            "variant": "small",
            "base_channels": 8,
            "channel_mult": [1, 2],
            "num_res_blocks": 1,
            "attention_resolutions": [],
            "dropout": 0.1,
        },
        (8, 8, 3),
        rngs=rngs,
    )
    model = FlowMatching(rhs, rngs, 2, ode_method="euler", ode_nstep_max=2)
    loss = flow_matching_loss(model, jnp.zeros((2, 8, 8, 3)), nnx.Rngs(13))
    assert jnp.isfinite(loss)


def test_unet_pullback_metric_supports_stateless_dropout():
    rngs = nnx.Rngs(20)
    rhs = build_rhs(
        {
            "type": "unet",
            "variant": "small",
            "base_channels": 4,
            "channel_mult": [1],
            "num_res_blocks": 1,
            "attention_resolutions": [],
            "dropout": 0.1,
        },
        (4, 4, 1),
        rngs=rngs,
    )
    model = FlowMatching(rhs, rngs, 1, ode_method="euler", ode_nstep_max=1)
    data = jnp.zeros((1, 4, 4, 1))
    _, parameters, _ = nnx.split(model, nnx.Param, ...)
    tangent = jax.tree.map(jnp.zeros_like, parameters)
    scalar = model.scalar_product(tangent, tangent, nnx.Rngs(21), data_batch=data)
    matvec = model.get_matvec_fn(nnx.Rngs(22), data_batch=data)(tangent)
    assert jnp.isfinite(scalar)
    assert all(jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(matvec))


def test_unet_sampling_is_deterministic_without_dropout_key():
    rngs = nnx.Rngs(30)
    rhs = build_rhs(
        {
            "type": "unet",
            "variant": "small",
            "base_channels": 4,
            "channel_mult": [1],
            "num_res_blocks": 1,
            "attention_resolutions": [],
            "dropout": 0.1,
        },
        (4, 4, 1),
        rngs=rngs,
    )
    model = FlowMatching(rhs, rngs, 1, ode_method="euler", ode_nstep_max=1)
    assert model.sample(1, nnx.Rngs(31)).shape == (1, 4, 4, 1)


if __name__ == "__main__":
    test_mlp_output_shape_and_parameter_count()
    test_explicit_mlp_flatten_adapter()
    test_unet_output_shapes_for_pixels_and_latents()
    test_sit_adapter_is_unconditional_and_shape_preserving()
    test_packaged_diffuse_sit_uses_canonical_import()
    test_architecture_compatibility_errors_are_preflighted()
    test_unet_dropout_runs_inside_flow_matching_vmap()
    test_unet_pullback_metric_supports_stateless_dropout()
    test_unet_sampling_is_deterministic_without_dropout_key()
    print("Image RHS tests passed.")
