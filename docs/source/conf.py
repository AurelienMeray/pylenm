# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import io
import os
import sys
import re
from tabnanny import verbose
print(os.getcwd())
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../pylenm/'))


# from pylenm/pylenm/__init__.py import version

# -- Project information -----------------------------------------------------

project = 'Pylenm'
copyright = '2022, Aurelien Meray'
author = 'Aurelien Meray'
sys.path.insert(0, os.path.abspath('../../pylenm/pylenm'))

from version import __version__

release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',        # to automatically build documentation from docstrings
    'sphinxcontrib.napoleon',
    'nbsphinx',
    'sphinx_gallery.load_style',
    # 'sphinx.ext.napoleon',       # to build from google style docstrings
]
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# nbsphinx_thumbnails = {
#     'notebooks/1) pyLEnM - Basics': 'notebook_thumbnails/tester.png',
#     'notebooks/2) pyLEnM - Unsupervised Learning': 'notebook_thumbnails/tester.png'
# }


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []



# This is processed by Jinja2 and inserted before each notebook 
nbsphinx_prolog = r""" 
 {% set docname = 'docs/source/' + env.doc2path(env.docname, base=None) %} 
  
 .. raw:: html 
  
     <div class="admonition note"> 
       This page was generated from 
       <a class="reference external" href="https://github.com/ALTEMIS-DOE/pylenm/blob/master/{{ docname|e }}">{{ docname|e }}</a>. 
       Interactive online version: 
       <span style="white-space: nowrap;"><a href="https://colab.research.google.com/github/ALTEMIS-DOE/pylenm/blob/master/{{ docname|e }}"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="vertical-align:text-bottom"></a>.</span> 

       <script> 
         if (document.location.host) { 
           $(document.currentScript).replaceWith( 
             '<a class="reference external" ' + 
             'href="https://nbviewer.jupyter.org/url' + 
             (window.location.protocol == 'https:' ? 's/' : '/') + 
             window.location.host + 
             window.location.pathname.slice(0, -4) + 
             'ipynb">View in <em>nbviewer</em></a>.' 
           ); 
         } 
       </script> 
     </div> 
  
 .. raw:: latex 
  
     \nbsphinxstartnotebook{\scriptsize\noindent\strut 
     \textcolor{gray}{The following section was generated from 
     \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}} 
 """ 





# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = "sphinx_rtd_theme"
# html_theme = "agogo"
# html_theme = "blue"
# html_theme_path = ["."]
html_theme = 'piccolo_theme'

html_theme_options = {
    "show_theme_credit": False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


# -- Miscelleneous -----------------------------------------------------------
myst_enable_extensions = [
  "colon_fence",
]

# source_suffix = {
#     '.rst': 'restructuredtext',
#     '.txt': 'markdown',
#     '.md': 'markdown',
# }