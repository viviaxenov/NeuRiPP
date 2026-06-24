import jax
from operator import add


def tree_dot_product(tree1: dict, tree2: dict):
    return jax.tree.reduce_associative(
        add, jax.tree.map(lambda _x, _y: (_x * _y).ravel().sum(), tree1, tree2)
    )

