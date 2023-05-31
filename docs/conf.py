# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "light-curve"
copyright = "2023, Konstantin Malanchev, Anastasia Lavrukhina"
author = "Konstantin Malanchev, Anastasia Lavrukhina"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.katex",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]


# KaTeX
katex_inline = ["$", "$"]
katex_display = ["$$", "$$"]


# Convert docstrings from Markdown to reStructuredText
def docstring(app, what, name, obj, options, lines):
    # import re

    import m2r2

    md = "\n".join(lines)

    # Replace '$$' with '```math ... ```'
    # md = re.sub(r'\$\$(.*?)\$\$', r'```math\n\1\n```', md)

    rst = m2r2.convert(md)

    lines.clear()
    lines.extend(rst.splitlines())


def setup(app):
    app.connect("autodoc-process-docstring", docstring)
