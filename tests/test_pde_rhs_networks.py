"""Focused CPU checks; no PDE solvers, training loops, or optional packages."""

import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from rhs_acoustic import AcousticUNetRHS
from rhs_ks import KSDilatedResNetRHS, _conv


class NetworkTests(unittest.TestCase):
    def tearDown(self):
        jax.clear_caches()

    def check_network(self, cls, dim, benchmark_dim, expected_count):
        model = cls(dim, nnx.Rngs(44))
        graph, params, rest = nnx.split(model, nnx.Param, ...)
        self.assertEqual(sum(p.size for p in jax.tree.leaves(params)), expected_count)
        self.assertFalse(jax.tree.leaves(rest))
        restored = nnx.merge(graph, params, rest)
        x = jnp.linspace(-0.4, 0.6, dim)
        y = model(0.3, x, None, rngs=nnx.Rngs(1))
        self.assertEqual(y.shape, x.shape)
        np.testing.assert_allclose(restored(0.3, x), y)
        batch = jnp.stack((x, x[::-1]))
        batched = jax.jit(lambda z: restored(0.3, z))(batch)
        self.assertEqual(batched.shape, batch.shape)
        np.testing.assert_allclose(batched, jax.vmap(lambda z: model(0.3, z))(batch),
                                   atol=2e-5, rtol=2e-5)
        self.assertTrue(np.isfinite(batched).all())

        # Check the state/time differential against a centered finite difference.
        direction = jnp.cos(jnp.arange(dim)) * 0.1
        _, tangent = jax.jvp(lambda t, z: restored(t, z),
                             (jnp.array(0.3), x), (jnp.array(0.2), direction))
        eps = 2e-3
        finite_difference = (restored(0.3 + eps * 0.2, x + eps * direction)
                             - restored(0.3 - eps * 0.2, x - eps * direction)) / (2 * eps)
        np.testing.assert_allclose(tangent, finite_difference, atol=3e-3, rtol=3e-2)
        self.assertGreater(float(jnp.linalg.norm(tangent)), 0)

        def loss(p):
            output = nnx.merge(graph, p, rest)(0.3, x)
            return jnp.mean(output ** 2)

        gradients = jax.jit(jax.grad(loss))(params)
        self.assertEqual(jax.tree.structure(gradients), jax.tree.structure(params))
        leaves = jax.tree.leaves(gradients)
        self.assertTrue(all(np.isfinite(g).all() for g in leaves))
        self.assertGreater(sum(float(jnp.sum(g ** 2)) for g in leaves), 0)
        # Trace the full benchmark grid without allocating full-grid activations.
        full = cls(benchmark_dim)
        shape = jax.eval_shape(lambda z: full(0.3, z),
                               jax.ShapeDtypeStruct((2, benchmark_dim), jnp.float32))
        self.assertEqual(shape.shape, (2, benchmark_dim))
        with self.assertRaises(ValueError):
            model(jnp.ones(2), batch)
        with self.assertRaises(ValueError):
            model(0.3, jnp.zeros(dim + 1))

    def test_acoustic(self):
        self.check_network(AcousticUNetRHS, 64, 256 ** 2, 509153)

    def test_ks(self):
        self.check_network(KSDilatedResNetRHS, 32, 200, 111073)

    def test_ks_periodic_dilation_and_normalization(self):
        # Independent circular cross-correlation, including wraparound where
        # dilation exceeds the grid size; catches input-vs-kernel dilation.
        conv = _conv(2, 3, 3, nnx.Rngs(8), dilation=8)
        x = jnp.arange(20, dtype=jnp.float32).reshape(2, 5, 2) / 20
        expected = jnp.broadcast_to(conv.bias[...], (2, 5, 3))
        for k in range(3):
            expected = expected + jnp.einsum(
                "blc,co->blo", jnp.roll(x, -(k - 1) * 8, axis=1), conv.kernel[k])
        np.testing.assert_allclose(conv(x), expected, atol=1e-6)
        model = KSDilatedResNetRHS(5)
        norm = model.blocks[0].norms[0]
        h = jnp.arange(320, dtype=jnp.float32).reshape(2, 5, 32) / 320
        mean = jnp.mean(h, axis=(1, 2), keepdims=True)
        variance = jnp.mean((h - mean) ** 2, axis=(1, 2), keepdims=True)
        np.testing.assert_allclose(norm(h), (h - mean) / jnp.sqrt(variance + 1e-5),
                                   atol=1e-6)

    def test_invalid_dimensions(self):
        for dim in (0, -1, 65, 36):
            with self.assertRaises(ValueError):
                AcousticUNetRHS(dim)
        with self.assertRaises(ValueError):
            KSDilatedResNetRHS(0)


if __name__ == "__main__":
    unittest.main()
