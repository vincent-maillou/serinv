# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SDR'
copyright = '2023, Vincent Maillou, Lisa Gaedke-merzhaeuser'
author = 'Vincent Maillou, Lisa Gaedke-merzhaeuser'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.intersphinx", # Links to numpy and scipy
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "github_url": "https://github.com/vincent-maillou/SDR",
    "collapse_navigation": True,
    "icon_links": [],  # See https://github.com/pydata/pydata-sphinx-theme/issues/1220
}

# TODO: html_css_files = ["bsparse.css"]

# TODO: html_context = {"default_mode": "light"}

# TODO: html_logo = "_static/bsparse.png"

html_static_path = ['_static']

html_use_modindex = True

html_file_suffix = ".html"

htmlhelp_basename = "sdr"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

autosummary_generate = True

autodoc_typehints = "none"