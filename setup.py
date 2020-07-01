from setuptools import setup

setup(name='pylenm',
      version='0.1',
      description='Python functions for Analyzing Historical Groundwater Datasets',
      url='http://github.com/',
      author='Aurelien Meray',
      author_email='aurelien.meray@gmail.com',
      license='LBNL',
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
          ],
      zip_safe=False)