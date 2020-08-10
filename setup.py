from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='pylenm',
      version='0.1.5',
      description='This package aims to provide machine learning (ML) functions for performing comprehensive soil and groundwater data analysis, and for supporting the establishment of effective long-term monitoring.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/AurelienMeray/pylenm',
      author='Aurelien Meray',
      author_email='aurelien.meray@gmail.com',
      license='MIT',
      packages=['pylenm'],
       install_requires=[
          'markdown',
          'pandas',
          'numpy',
          'seaborn',
          'matplotlib',
          'scipy',
          'datetime',
          'sklearn',
          'supersmoother',
          'ipyleaflet>=0.13.0',
          'ipywidgets>=7.5.0'
          ],
      zip_safe=False)