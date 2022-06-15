Usage
=====

.. _installation:

Installation
------------

To use Pylenm, first install it using pip:

.. code-block:: console

   (.venv) $ pip install pylenm

General import statement
------------------------

At the moment, Pylenm only has one module: ``PylenmDataFactory``. 

You can import it as follow:

>>> import pylenm
>>> pylenm.__version__
'0.2'

You can start using pylenm by passing your dataset into ``PylenmDataFactory``:

>>> data = pd.read_csv(PATH)
>>> pylenm_df = pylenm.PylenmDataFactory(data)
Successfully imported the data!


