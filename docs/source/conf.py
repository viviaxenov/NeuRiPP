# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../src/"))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "NeuRiPP"
copyright = "2025, Vitalii Aksenov, Sebastian Gutierrez"
author = "Vitalii Aksenov, Sebastian Gutierrez"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "nbsphinx",
]

autodoc_default_options = {
    "members": True,
    # "undoc-members": True,
    # "private-members": True,
}

# so that the documents can be built in a lighter environment
# e.g. during github pages deployment
#
autodoc_mock_imports = [
    "numpy",
    "scipy",
    "jax",
    "jax.numpy",
    "flax",
    "jaxtyping",
    "typing_extensions",
    "optax",
    "matplotlib",
    "tqdm",
    "neuripp.architectures.MMNN" # Importing this causese some weird errors; TODO fix (?)
]
autodoc_typehints = "description"

templates_path = ["_templates"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
