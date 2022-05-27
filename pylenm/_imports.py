# Required imports
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
import scipy
import random
import datetime
import re
import time
from math import sqrt
import matplotlib.dates as mdates
from matplotlib.dates import date2num, num2date
from sklearn import preprocessing
pd.set_option('display.max_columns', None) # to view all columns
from scipy.optimize import curve_fit
from supersmoother import SuperSmoother
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")
from pyproj import Proj, Transformer
from ipyleaflet import (Map, basemaps, WidgetControl, GeoJSON, 
                        LayersControl, Icon, Marker,FullScreenControl,
                        CircleMarker, Popup, AwesomeIcon) 
from ipywidgets import HTML
plt.rcParams["font.family"] = "Times New Roman"