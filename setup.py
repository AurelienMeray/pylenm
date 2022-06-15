import setuptools
from setuptools import setup
import io
import os
import re

with open("README.md", "r") as fh:
    long_description = fh.read()

def read(path, encoding='utf-8'):
    path = os.path.join(os.path.dirname(__file__), path)
    with io.open(path, encoding=encoding) as fp:
        return fp.read()

def version(path):
    version_file = read(path)
    version_match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(name='pylenm',
      version=version('pylenm/version.py'),
      description='This package aims to provide machine learning (ML) functions for performing comprehensive soil and groundwater data analysis, and for supporting the establishment of effective long-term monitoring.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/ALTEMIS-DOE/pylenm',
      author='Aurelien Meray',
      author_email='aurelien.meray@gmail.com',
      license='MIT',
      packages=setuptools.find_packages(),
      install_requires=[
        'markdown',
        'pandas>=1.2.3',
        'openpyxl>=2.6.0',
        'pyproj>=3.0.1',
        'richdem',
        'rasterio',
        'numpy',
        'seaborn',
        'matplotlib',
        'statsmodels',
        'scipy',
        'pyproj',
        'datetime',
        'scikit-learn>=0.24.1',
        'supersmoother',
        'ipyleaflet>=0.13.0',
        'ipywidgets>=7.5.0'
        ],
      classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        ],
      python_requires='>=3.6',
      zip_safe=False,
      project_urls={
        "Bug Tracker": "https://github.com/ALTEMIS-DOE/pylenm/issues",
        "Documentation": "https://pylenm.readthedocs.io/",
        "Source": "https://github.com/ALTEMIS-DOE/pylenm",
      },
      )