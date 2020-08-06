from setuptools import setup

setup(name='pylenm',
      version='0.1.5',
      description='Python functions for Analyzing Historical Groundwater Datasets',
      url='https://github.com/AurelienMeray/pylenm',
      author='Aurelien Meray',
      author_email='aurelien.meray@gmail.com',
      license='MIT',
      packages=[],
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