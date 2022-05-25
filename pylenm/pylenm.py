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
get_ipython().run_line_magic('matplotlib', 'inline')
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

class functions:
    
    def __init__(self, data):
        self.setData(data)
        self.__jointData = [None, 0]

# DATA VALIDATION     
    def __isValid_Data(self, data):
        if(str(type(data)).lower().find('dataframe') == -1):
            return (False, 'Make sure the data is a pandas DataFrame.\n')
        if(not self.__hasColumns_Data(data)):
            return (False, 'Make sure that ALL of the columns specified in the REQUIREMENTS are present.\n')
        else:
            return (True, None)
    
    def __isValid_Construction_Data(self, data):
        if(str(type(data)).lower().find('dataframe') == -1):
            return (False, 'Make sure the data is a pandas DataFrame.\n')
        if(not self.__hasColumns_Construction_Data(data)):
            return (False, 'Make sure that ALL of the columns specified in the REQUIREMENTS are present.\n')
        else:
            return (True, None)
    
# COLUMN VALIDATION 
    def __hasColumns_Data(self, data):
        find = ['COLLECTION_DATE','STATION_ID','ANALYTE_NAME','RESULT','RESULT_UNITS']
        cols = list(data.columns)
        cols = [x.upper() for x in cols]
        hasCols =  all(item in cols for item in find)
        return hasCols

    def __hasColumns_Construction_Data(self, data):
        find = ['STATION_ID', 'AQUIFER', 'WELL_USE', 'LATITUDE', 'LONGITUDE', 'GROUND_ELEVATION', 'TOTAL_DEPTH']
        cols = list(data.columns)
        cols = [x.upper() for x in cols]
        hasCols =  all(item in cols for item in find)
        return hasCols

# SETTING DATA    
    def setData(self, data, verbose=True):
        validation = self.__isValid_Data(data)
        if(validation[0]):
            # Make all columns all caps
            cols_upper = [x.upper() for x in list(data.columns)]
            data.columns = cols_upper
            self.data = data
            if(verbose):
                print('Successfully imported the data!\n')
            self.__set_units()
        else:
            print('ERROR: {}'.format(validation[1]))
            return self.REQUIREMENTS_DATA()
    
    def setConstructionData(self, construction_data, verbose=True):
        validation = self.__isValid_Construction_Data(construction_data)
        if(validation[0]):
            # Make all columns all caps
            cols_upper = [x.upper() for x in list(construction_data.columns)]
            construction_data.columns = cols_upper
            self.construction_data = construction_data.set_index(['STATION_ID'])
            if(verbose):
                print('Successfully imported the construction data!\n')
        else:
            print('ERROR: {}'.format(validation[1]))
            return self.REQUIREMENTS_CONSTRUCTION_DATA()
    
    def jointData_is_set(self, lag):
        if(str(type(self.__jointData[0])).lower().find('dataframe') == -1):
            return False
        else:
            if(self.__jointData[1]==lag):
                return True
            else:
                return False

    def set_jointData(self, data, lag):
        self.__jointData[0] = data
        self.__jointData[1] = lag
    
# GETTING DATA      
    def getData(self):
        return self.data
    
    def get_Construction_Data(self):
        return self.construction_data

# MESSAGES FOR INVALID DATA          
    def REQUIREMENTS_DATA(self):
        print('PYLENM DATA REQUIREMENTS:\nThe imported data needs to meet ALL of the following conditions to have a successful import:')
        print('   1) Data should be a pandas dataframe.')
        print("   2) Data must have these column names: \n      ['COLLECTION_DATE','STATION_ID','ANALYTE_NAME','RESULT','RESULT_UNITS']")

    def REQUIREMENTS_CONSTRUCTION_DATA(self):
        print('PYLENM CONSTRUCTION REQUIREMENTS:\nThe imported construction data needs to meet ALL of the following conditions to have a successful import:')
        print('   1) Data should be a pandas dataframe.')
        print("   2) Data must have these column names: \n      ['station_id', 'aquifer', 'well_use', 'latitude', 'longitude', 'ground_elevation', 'total_depth']")
    
    # Helper function for plot_correlation
    # Sorts analytes in a specific order: 'TRITIUM', 'URANIUM-238','IODINE-129','SPECIFIC CONDUCTANCE', 'PH', 'DEPTH_TO_WATER'
    def __custom_analyte_sort(self, analytes):
        my_order = 'TURISPDABCEFGHJKLMNOQVWXYZ-_abcdefghijklmnopqrstuvwxyz135790 2468'
        return sorted(analytes, key=lambda word: [my_order.index(c) for c in word])
    
    def __plotUpperHalf(self, *args, **kwargs):
        corr_r = args[0].corr(args[1], 'pearson')
        corr_text = f"{corr_r:2.2f}"
        ax = plt.gca()
        ax.set_axis_off()
        marker_size = abs(corr_r) * 10000
        ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
                   vmin=-1, vmax=1, transform=ax.transAxes)
        font_size = abs(corr_r) * 40 + 5
        ax.annotate(corr_text, [.5, .48,],  xycoords="axes fraction", # [.5, .48,]
                    ha='center', va='center', fontsize=font_size, fontweight='bold')
    
    # Description:
    #    Removes all columns except 'COLLECTION_DATE', 'STATION_ID', 'ANALYTE_NAME', 'RESULT', and 'RESULT_UNITS'.
    #    If the user specifies additional columns in addition to the ones listed above, those columns will be kept.
    #    The function returns a dataframe and has an optional parameter to be able to save the dataframe to a csv file.
    # Parameters:
    #    data (dataframe): data to simplify
    #    inplace (bool): save data to current working dataset
    #    columns (list of strings): list of any additional columns on top of  ['COLLECTION_DATE', 'STATION_ID', 'ANALYTE_NAME', 'RESULT', and 'RESULT_UNITS'] to be kept in the dataframe.
    #    save_csv (bool): flag to determine whether or not to save the dataframe to a csv file.
    #    file_name (string): name of the csv file you want to save
    #    save_dir (string): name of the directory you want to save the csv file to
    def simplify_data(self, data=None, inplace=False, columns=None, save_csv=False, file_name= 'data_simplified', save_dir='data/'):
        if(str(type(data)).lower().find('dataframe') == -1):
            data = self.data
        else:
            data = data
        if(columns==None):
            sel_cols = ['COLLECTION_DATE','STATION_ID','ANALYTE_NAME','RESULT','RESULT_UNITS']
        else:
            hasColumns =  all(item in list(data.columns) for item in columns)
            if(hasColumns):            
                sel_cols = ['COLLECTION_DATE','STATION_ID','ANALYTE_NAME','RESULT','RESULT_UNITS'] + columns
            else:
                print('ERROR: specified column(s) do not exist in the data')
                return None

        data = data[sel_cols]
        data.COLLECTION_DATE = pd.to_datetime(data.COLLECTION_DATE)
        data = data.sort_values(by="COLLECTION_DATE")
        dup = data[data.duplicated(['COLLECTION_DATE', 'STATION_ID','ANALYTE_NAME', 'RESULT'])]
        data = data.drop(dup.index)
        data = data.reset_index().drop('index', axis=1)
        if(save_csv):
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            data.to_csv(save_dir + file_name + '.csv')
            print('Successfully saved "' + file_name +'.csv" in ' + save_dir)
        if(inplace):
            self.setData(data, verbose=False)
        return data
    
    # Description:
    #    Returns the Maximum Concentration Limit value for the specified analyte.
    #    Example: 'TRITIUM' returns 1.3
    # Parameters:
    #    analyte_name (string): name of the analyte to be processed
    def get_MCL(self, analyte_name):
        mcl_dictionary = {'TRITIUM': 1.3, 'URANIUM-238': 1.31,  'NITRATE-NITRITE AS NITROGEN': 1,
                          'TECHNETIUM-99': 2.95, 'IODINE-129': 0, 'STRONTIUM-90': 0.9
                         }
        return mcl_dictionary[analyte_name]
    
    def __set_units(self):
        analytes = list(np.unique(self.data[['ANALYTE_NAME']]))
        mask1 = ~self.data[['ANALYTE_NAME','RESULT_UNITS']].duplicated()
        res = self.data[['ANALYTE_NAME','RESULT_UNITS']][mask1]
        mask2 = ~self.data[['ANALYTE_NAME']].duplicated()
        res = res[mask2]
        unit_dictionary = pd.Series(res.RESULT_UNITS.values,index=res.ANALYTE_NAME).to_dict()
        self.unit_dictionary = unit_dictionary
        
    
    # Description:
    #    Returns the unit of the analyte you specify.
    #    Example: 'DEPTH_TO_WATER' returns 'ft'
    # Parameters:
    #    analyte_name (string): name of the analyte to be processed
    def get_unit(self, analyte_name):
        return self.unit_dictionary[analyte_name]

    # Description: 
    #    Filters construction data based on one column. You only specify ONE column to filter by, but can selected MANY values for the entry.
    # Parameters:
    #    data (dataframe): dataframe to filter
    #    col (string): column to filter. Example: col='STATION_ID'
    #    equals (list of strings): values to filter col by. Examples: equals=['FAI001A', 'FAI001B']
    def filter_by_column(self, data=None, col=None, equals=[]):
        if(data is None):
            return 'ERROR: DataFrame was not provided to this function.'
        else:
            if(str(type(data)).lower().find('dataframe') == -1):
                return 'ERROR: Data provided is not a pandas DataFrame.'
            else:
                data = data
        # DATA VALIDATION
        if(col==None):
            return 'ERROR: Specify a column name to filter by.'
        data_cols = list(data.columns)
        if((col in data_cols)==False): # Make sure column name exists 
            return 'Error: Column name "{}" does not exist'.format(col)
        if(equals==[]):
            return 'ERROR: Specify a value that "{}" should equal to'.format(col)
        data_val = list(data[col])
        for value in equals:
            if((value in data_val)==False):
                return 'ERROR: No value equal to "{}" in "{}".'.format(value, col)

        # QUERY
        final_data = pd.DataFrame()
        for value in equals:
            current_data = data[data[col]==value]
            final_data = pd.concat([final_data, current_data])
        return final_data
    
    # Description:
    #    Returns a list of the well names filtered by the unit(s) specified.
    # Parameters:
    #    units (list of strings): Letter of the well to be filtered (e.g. [‘A’] or [‘A’, ‘D’])
    def filter_wells(self, units):
        data = self.data
        if(units==None):
            units= ['A', 'B', 'C', 'D']
        def getUnits():
            wells = list(np.unique(data.STATION_ID))
            wells = pd.DataFrame(wells, columns=['STATION_ID'])
            for index, row in wells.iterrows():
                mo = re.match('.+([0-9])[^0-9]*$', row.STATION_ID)
                last_index = mo.start(1)
                wells.at[index, 'unit'] = row.STATION_ID[last_index+1:]
                u = wells.unit.iloc[index]
                if(len(u)==0): # if has no letter, use D
                    wells.at[index, 'unit'] = 'D'
                if(len(u)>1): # if has more than 1 letter, remove the extra letter
                    if(u.find('R')>0):
                        wells.at[index, 'unit'] = u[:-1]
                    else:
                        wells.at[index, 'unit'] = u[1:]
                u = wells.unit.iloc[index]
                if(u=='A' or u=='B' or u=='C' or u=='D'):
                    pass
                else:
                    wells.at[index, 'unit'] = 'D'
            return wells
        df = getUnits()
        res = df.loc[df.unit.isin(units)]
        return list(res.STATION_ID)

    # Description:
    #    Removes outliers from a dataframe based on the z_scores and returns the new dataframe.
    # Parameters:
    #    data (dataframe): data for the outliers to removed from
    #    z_threshold (float): z_score threshold to eliminate.
    def remove_outliers(self, data, z_threshold=4):
        z = np.abs(stats.zscore(data))
        row_loc = np.unique(np.where(z > z_threshold)[0])
        data = data.drop(data.index[row_loc])
        return data
    
    # Description:
    #    Returns a csv file saved to save_dir with details pertaining to the specified analyte.
    #    Details include the well names, the date ranges and the number of unique samples.
    # Parameters:
    #    analyte_name (string): name of the analyte to be processed
    #    save_dir (string): name of the directory you want to save the csv file to
    def get_analyte_details(self, analyte_name, filter=False, col=None, equals=[], save_to_file = False, save_dir='analyte_details'):
        data = self.data
        data = data[data.ANALYTE_NAME == analyte_name].reset_index().drop('index', axis=1)
        data = data[~data.RESULT.isna()]
        data = data.drop(['ANALYTE_NAME', 'RESULT', 'RESULT_UNITS'], axis=1)
        data.COLLECTION_DATE = pd.to_datetime(data.COLLECTION_DATE)
        if(filter):
            filter_res = self.filter_by_column(data=self.construction_data, col=col, equals=equals)
            if('ERROR:' in str(filter_res)):
                return filter_res
            query_wells = list(data.STATION_ID.unique())
            filter_wells = list(filter_res.index.unique())
            intersect_wells = list(set(query_wells) & set(filter_wells))
            if(len(intersect_wells)<=0):
                return 'ERROR: No results for this query with the specifed filter parameters.'
            data = data[data['STATION_ID'].isin(intersect_wells)]        

        info = []
        wells = np.unique(data.STATION_ID.values)
        for well in wells:
            current = data[data.STATION_ID == well]
            startDate = current.COLLECTION_DATE.min().date()
            endDate = current.COLLECTION_DATE.max().date()
            numSamples = current.duplicated().value_counts()[0]
            info.append({'Well Name': well, 'Start Date': startDate, 'End Date': endDate,
                         'Date Range (days)': endDate-startDate ,
                         'Unique samples': numSamples})
            details = pd.DataFrame(info)
            details.index = details['Well Name']
            details = details.drop('Well Name', axis=1)
            details = details.sort_values(by=['Start Date', 'End Date'])
            details['Date Range (days)'] = (details['Date Range (days)']/ np.timedelta64(1, 'D')).astype(int)
        if(save_to_file):
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            details.to_csv(save_dir + '/' + analyte_name + '_details.csv')
        return details
    
    # Description:
    #    Returns a dataframe with a summary of the data for certain analytes.
    #    Summary includes the date ranges and the number of unique samples and other statistics for the analyte results.
    # Parameters:
    #    analytes (list of strings): list of analyte names to be processed. If left empty, a list of all the analytes in the data will be used.
    #    sort_by (string): {‘date’, ‘samples’, ‘wells’} sorts the data by either the dates by entering: ‘date’, the samples by entering: ‘samples’, or by unique well locations by entering ‘wells’.
    #    ascending (bool): flag to sort in ascending order.
    def get_data_summary(self, analytes=None, sort_by='date', ascending=False, filter=False, col=None, equals=[]):
        data = self.data
        if(analytes == None):
            analytes = data.ANALYTE_NAME.unique()
        data = data.loc[data.ANALYTE_NAME.isin(analytes)].drop(['RESULT_UNITS'], axis=1)
        data = data[~data.duplicated()] # remove duplicates
        data.COLLECTION_DATE = pd.to_datetime(data.COLLECTION_DATE)
        data = data[~data.RESULT.isna()]
        if(filter):
            filter_res = self.filter_by_column(data=self.construction_data, col=col, equals=equals)
            if('ERROR:' in str(filter_res)):
                return filter_res
            query_wells = list(data.STATION_ID.unique())
            filter_wells = list(filter_res.index.unique())
            intersect_wells = list(set(query_wells) & set(filter_wells))
            if(len(intersect_wells)<=0):
                return 'ERROR: No results for this query with the specifed filter parameters.'
            data = data[data['STATION_ID'].isin(intersect_wells)]

        info = []
        for analyte_name in analytes:
            query = data[data.ANALYTE_NAME == analyte_name]
            startDate = min(query.COLLECTION_DATE)
            endDate = max(query.COLLECTION_DATE)
            numSamples = query.shape[0]
            wellCount = len(query.STATION_ID.unique())
            stats = query.RESULT.describe().drop('count', axis=0)
            stats = pd.DataFrame(stats).T
            stats_col = [x for x in stats.columns]

            result = {'Analyte Name': analyte_name, 'Start Date': startDate, 'End Date': endDate,
                      'Date Range (days)':endDate-startDate, '# unique wells': wellCount,'# samples': numSamples,
                      'Unit': self.get_unit(analyte_name) }
            for num in range(len(stats_col)):
                result[stats_col[num]] = stats.iloc[0][num] 

            info.append(result)

            details = pd.DataFrame(info)
            details.index = details['Analyte Name']
            details = details.drop('Analyte Name', axis=1)
            if(sort_by.lower() == 'date'):
                details = details.sort_values(by=['Start Date', 'End Date', 'Date Range (days)'], ascending=ascending)
            elif(sort_by.lower() == 'samples'):
                details = details.sort_values(by=['# samples'], ascending=ascending)
            elif(sort_by.lower() == 'wells'):
                details = details.sort_values(by=['# unique wells'], ascending=ascending)

        return details
    
    # Description: 
    #    Displays the analyte names available at given well locations.
    # Parameters:
    #    well_name (string): name of the well. If left empty, all wells are returned.
    #    filter (bool): flag to indicate filtering
    #    col (string): column to filter results
    #    equals (list of strings): value to match column name. Multiple values are accepted.
    def get_well_analytes(self, well_name=None, filter=False, col=None, equals=[]):
        data = self.data
        bb = "\033[1m"
        be = "\033[0m"
        if(filter):
            filter_res = self.filter_by_column(data=self.construction_data, col=col, equals=equals)
            if('ERROR:' in str(filter_res)):
                return filter_res
            query_wells = list(data.STATION_ID.unique())
            filter_wells = list(filter_res.index.unique())
            intersect_wells = list(set(query_wells) & set(filter_wells))
            if(len(intersect_wells)<=0):
                return 'ERROR: No results for this query with the specifed filter parameters.'
            data = data[data['STATION_ID'].isin(intersect_wells)]
        
        if(well_name==None):
            wells = list(data.STATION_ID.unique())
        else:
            wells = [well_name]
        for well in wells:
            print("{}{}{}".format(bb,str(well), be))
            analytes = sorted(list(data[data.STATION_ID==well].ANALYTE_NAME.unique()))
            print(str(analytes) +'\n')
    
    # Description: 
    #    Filters data by passing the data and specifying the well_name and analyte_name
    # Parameters:
    #    well_name (string): name of the well to be processed
    #    analyte_name (string): name of the analyte to be processed
    def query_data(self, well_name, analyte_name):
        data = self.data
        query = data[data.STATION_ID == well_name]
        query = query[query.ANALYTE_NAME == analyte_name]
        if(query.shape[0] == 0):
            return 0
        else:
            return query
    
    # Description: 
    #    Plot concentrations over time of a specified well and analyte with a smoothed curve on interpolated data points.
    # Parameters:
    #    well_name (string): name of the well to be processed
    #    analyte_name (string): name of the analyte to be processed
    #    log_transform (bool): choose whether or not the data should be transformed to log base 10 values
    #    alpha (int): value between 0 and 10 for line smoothing
    #    year_interval (int): plot by how many years to appear in the axis e.g.(1 = every year, 5 = every 5 years, ...)
    #    plot_inline (bool): choose whether or not to show plot inline
    #    save_dir (string): name of the directory you want to save the plot to
    def plot_data(self, well_name, analyte_name, log_transform=True, alpha=0,
              plot_inline=True, year_interval=2, x_label='Years', y_label='', save_dir='plot_data', filter=False, col=None, equals=[]):
    
        # Gets appropriate data (well_name and analyte_name)
        query = self.query_data(well_name, analyte_name)
        query = self.simplify_data(data=query)

        if(type(query)==int and query == 0):
            return 'No results found for {} and {}'.format(well_name, analyte_name)
        else:
            if(filter):
                filter_res = self.filter_by_column(data=self.construction_data, col=col, equals=equals)
                if('ERROR:' in str(filter_res)):
                    return filter_res
                query_wells = list(query.STATION_ID.unique())
                filter_wells = list(filter_res.index.unique())
                intersect_wells = list(set(query_wells) & set(filter_wells))
                if(len(intersect_wells)<=0):
                    return 'ERROR: No results for this query with the specifed filter parameters.'
                query = query[query['STATION_ID'].isin(intersect_wells)]
            x_data = query.COLLECTION_DATE
            x_data = pd.to_datetime(x_data)
            y_data = query.RESULT
            if(log_transform):
                y_data = np.log10(y_data)

            # Remove any NaN as a result of the log transformation
            nans = ~np.isnan(y_data)
            x_data = x_data[nans]
            y_data = y_data[nans]

            x_RR = x_data.astype(int).to_numpy()

            # Remove any duplicate dates
            unique = ~pd.Series(x_data).duplicated()
            x_data = x_data[unique]
            y_data = y_data[unique]
            unique = ~pd.Series(y_data).duplicated()
            x_data = x_data[unique]
            y_data = y_data[unique]
            x_RR = x_data.astype(int).to_numpy()

            nu = x_data.shape[0]

            result = None
            while result is None:
                if(nu < 5):
                    return 'ERROR: Could not plot {}, {}'.format(well_name, analyte_name)
                    break
                nu = nu - 1
                x_data = x_data[:nu]
                x_RR = x_RR[:nu]
                y_data = y_data[:nu]

                try:
                    # fit the supersmoother model
                    model = SuperSmoother(alpha=alpha)
                    model.fit(x_RR, y_data)
                    y_pred = model.predict(x_RR)
                    r = model.cv_residuals()
                    out = abs(r) > 2.2*np.std(r)
                    out_x = x_data[out]
                    out_y = y_data[out]

                    plt.figure(figsize=(8,8))
                    ax = plt.axes()
                    years = mdates.YearLocator(year_interval)  # every year
                    months = mdates.MonthLocator()  # every month
                    yearsFmt = mdates.DateFormatter('%Y')

                    for label in ax.get_xticklabels():
                        label.set_rotation(30)
                        label.set_horizontalalignment('center')

                    ax = plt.gca()
                    ax.xaxis.set_major_locator(years)
                    ax.xaxis.set_major_formatter(yearsFmt)
                    ax.autoscale_view()

                    unit = query.RESULT_UNITS.values[0]

                    ax.set_title(well_name + ' - ' + analyte_name, fontweight='bold')
                    ttl = ax.title
                    ttl.set_position([.5, 1.05])
                    if(y_label==''):    
                        if(log_transform):
                            ax.set_ylabel('log-Concentration (' + unit + ')')
                        else:
                            ax.set_ylabel('Concentration (' + unit + ')')
                    else:
                        ax.set_ylabel(y_label)
                    ax.set_xlabel(x_label)
                    small_fontSize = 15
                    large_fontSize = 20
                    plt.rc('axes', titlesize=large_fontSize)
                    plt.rc('axes', labelsize=large_fontSize)
                    plt.rc('legend', fontsize=small_fontSize)
                    plt.rc('xtick', labelsize=small_fontSize)
                    plt.rc('ytick', labelsize=small_fontSize) 
                    ax.plot(x_data, y_data, ls='', marker='o', ms=5, color='black', alpha=1)
                    ax.plot(x_data, y_pred, ls='-', marker='', ms=5, color='black', alpha=0.5, label="Super Smoother")
                    ax.plot(out_x , out_y, ls='', marker='o', ms=5, color='red', alpha=1, label="Outliers")
                    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0.)
                    props = dict(boxstyle='round', facecolor='grey', alpha=0.15)       
                    ax.text(1.05, 0.85, 'Samples: {}'.format(nu), transform=ax.transAxes, 
                            fontsize=small_fontSize,
                            fontweight='bold',
                            verticalalignment='top', 
                            bbox=props)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    plt.savefig(save_dir + '/' + well_name + '-' + analyte_name +'.png', bbox_inches="tight")
                    if(plot_inline):
                        plt.show()
                    plt.clf()
                    plt.cla()
                    plt.close()
                    result = 1
                except:
                    pass
    
    # Description: 
    #    Plot concentrations over time for every well and analyte with a smoothed curve on interpolated data points.
    # Parameters:
    #    log_transform (bool): choose whether or not the data should be transformed to log base 10 values
    #    alpha (int): value between 0 and 10 for line smoothing
    #    year_interval (int): plot by how many years to appear in the axis e.g.(1 = every year, 5 = every 5 years, ...)
    #    plot_inline (bool): choose whether or not to show plot inline
    #    save_dir (string): name of the directory you want to save the plot to
    def plot_all_data(self, log_transform=True, alpha=0, year_interval=2, plot_inline=True, save_dir='plot_data'):
        analytes = ['TRITIUM','URANIUM-238','IODINE-129','SPECIFIC CONDUCTANCE', 'PH', 'DEPTH_TO_WATER']
        wells = np.array(data.STATION_ID.values)
        wells = np.unique(wells)
        success = 0
        errors = 0
        for well in wells:
            for analyte in analytes:
                plot = self.plot_data(well, analyte, 
                                 log_transform=log_transform, 
                                 alpha=alpha, 
                                 year_interval=year_interval,
                                 plot_inline=plot_inline,
                                 save_dir=save_dir)
                if 'ERROR:' in str(plot):
                    errors = errors + 1
                else:
                    success = success + 1
        print("Success: ", success)
        print("Errors: ", errors)
    
    # Description: 
    #    Plots a heatmap of the correlations of the important analytes over time for a specified well.
    # Parameters:
    #    well_name (string): name of the well to be processed
    #    show_symmetry (bool): choose whether or not the heatmap should show the same information twice over the diagonal
    #    color (bool): choose whether or not the plot should be in color or in greyscale
    #    save_dir (string): name of the directory you want to save the plot to
    def plot_correlation_heatmap(self, well_name, show_symmetry=True, color=True, save_dir='plot_correlation_heatmap'):
        data = self.data
        query = data[data.STATION_ID == well_name]
        a = list(np.unique(query.ANALYTE_NAME.values))
        b = ['TRITIUM','IODINE-129','SPECIFIC CONDUCTANCE', 'PH','URANIUM-238', 'DEPTH_TO_WATER']
        analytes = self.__custom_analyte_sort(list(set(a) and set(b)))
        query = query.loc[query.ANALYTE_NAME.isin(analytes)]
        analytes = self.__custom_analyte_sort(np.unique(query.ANALYTE_NAME.values))
        x = query[['COLLECTION_DATE', 'ANALYTE_NAME']]
        unique = ~x.duplicated()
        query = query[unique]
        piv = query.reset_index().pivot(index='COLLECTION_DATE',columns='ANALYTE_NAME', values='RESULT')
        piv = piv[analytes]
        totalSamples = piv.shape[0]
        piv = piv.dropna()
        samples = piv.shape[0]
        if(samples < 5):
            return 'ERROR: {} does not have enough samples to plot.'.format(well_name)
        else:
            scaler = StandardScaler()
            pivScaled = scaler.fit_transform(piv)
            pivScaled = pd.DataFrame(pivScaled, columns=piv.columns)
            pivScaled.index = piv.index
            corr_matrix = pivScaled.corr()
            if(show_symmetry):
                mask = None
            else:
                mask = np.triu(corr_matrix)
            if(color):
                cmap = 'RdBu'
            else:
                cmap = 'binary'
            fig, ax = plt.subplots(figsize=(8,6))
            ax.set_title(well_name + '_correlation', fontweight='bold')
            ttl = ax.title
            ttl.set_position([.5, 1.05])
            props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
            ax.text(1.3, 1.05, 'Start date:  {}\nEnd date:    {}\n\nSamples:     {} of {}'.format(piv.index[0], piv.index[-1], samples, totalSamples), transform=ax.transAxes, fontsize=15, fontweight='bold', verticalalignment='bottom', bbox=props)
            ax = sns.heatmap(corr_matrix,
                                   ax=ax,
                                   mask=mask,
                                   vmin=-1, vmax=1,
                                   xticklabels=corr_matrix.columns,
                                   yticklabels=corr_matrix.columns,
                                   cmap=cmap,
                                   annot=True,
                                   linewidths=1,
                                   cbar_kws={'orientation': 'vertical'})
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fig.savefig(save_dir + '/' + well_name + '_correlation.png', bbox_inches="tight")
    
    # Description: 
    #    Plots a heatmap of the correlations of the important analytes over time for each well in the dataset.
    # Parameters:
    #    show_symmetry (bool): choose whether or not the heatmap should show the same information twice over the diagonal
    #    color (bool): choose whether or not the plot should be in color or in greyscale
    #    save_dir (string): name of the directory you want to save the plot to
    def plot_all_correlation_heatmap(self, show_symmetry=True, color=True, save_dir='plot_correlation_heatmap'):
        data = self.data
        wells = np.array(data.STATION_ID.values)
        wells = np.unique(wells)
        for well in wells:
            self.plot_correlation_heatmap(well_name=well,
                                          show_symmetry=show_symmetry,
                                          color=color,
                                          save_dir=save_dir)

    
    # Description: 
    #    Resamples the data based on the frequency specified and interpolates the values of the analytes.
    # Parameters:
    #    well_name (string): name of the well to be processed
    #    analytes (list of strings): list of analyte names to use
    #    frequency (string): {‘D’, ‘W’, ‘M’, ‘Y’} frequency to interpolate. 
    #        See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html for valid frequency inputs. (e.g. ‘W’ = every week, ‘D ’= every day, ‘2W’ = every 2 weeks)
    def interpolate_well_data(self, well_name, analytes, frequency='2W'):
        data = self.data
        inter_series = {}
        query = data[data.STATION_ID == well_name]
        for analyte in analytes:
            series = query[query.ANALYTE_NAME == analyte]
            series = (series[['COLLECTION_DATE', 'RESULT']])
            series.COLLECTION_DATE = pd.to_datetime(series.COLLECTION_DATE)
            series.index = series.COLLECTION_DATE
            original_dates = series.index
            series = series.drop('COLLECTION_DATE', axis=1)
            series = series.rename({'RESULT': analyte}, axis=1)
            upsampled = series.resample(frequency).mean()
            interpolated = upsampled.interpolate(method='linear', order=2)
            inter_series[analyte] = interpolated
        join = inter_series[analytes[0]]
        join = join.drop(analytes[0], axis=1)
        for analyte in analytes:
            join = join.join(inter_series[analyte])
        join = join.dropna()
        return join
    
    # Description: 
    #    Plots the correlations with the physical plots as well as the correlations of the important analytes over time for a specified well.
    # Parameters:
    #    well_name (string): name of the well to be processed
    #    analytes (list of strings): list of analyte names to use
    #    remove_outliers (bool): choose whether or to remove the outliers.
    #    z_threshold (float): z_score threshold to eliminate outliers
    #    interpolate (bool): choose whether or to interpolate the data
    #    frequency (string): {‘D’, ‘W’, ‘M’, ‘Y’} frequency to interpolate. See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html for valid frequency inputs. (e.g. ‘W’ = every week, ‘D ’= every day, ‘2W’ = every 2 weeks)
    #    save_dir (string): name of the directory you want to save the plot to
    def plot_corr_by_well(self, well_name, analytes, remove_outliers=True, z_threshold=4, interpolate=False, frequency='2W', save_dir='plot_correlation', log_transform=False, fontsize=20, returnData=False, remove=[], no_log=None):
        data = self.data
        query = data[data.STATION_ID == well_name]
        a = list(np.unique(query.ANALYTE_NAME.values))# get all analytes from dataset
        for value in analytes:
            if((value in a)==False):
                return 'ERROR: No analyte named "{}" in data.'.format(value)
        analytes = sorted(analytes)
        query = query.loc[query.ANALYTE_NAME.isin(analytes)]
        x = query[['COLLECTION_DATE', 'ANALYTE_NAME']]
        unique = ~x.duplicated()
        query = query[unique]
        piv = query.reset_index().pivot(index='COLLECTION_DATE',columns='ANALYTE_NAME', values='RESULT')
        piv = piv[analytes]
        piv.index = pd.to_datetime(piv.index)
        totalSamples = piv.shape[0]
        piv = piv.dropna()
        if(interpolate):
            piv = self.interpolate_well_data(well_name, analytes, frequency=frequency)
            file_extension = '_interpolated_' + frequency
            title = well_name + '_correlation - interpolated every ' + frequency
        else:
            file_extension = '_correlation'
            title = well_name + '_correlation'
        samples = piv.shape[0]
        if(samples < 5):
            if(interpolate):
                return 'ERROR: {} does not have enough samples to plot.\n Try a different interpolation frequency'.format(well_name)
            return 'ERROR: {} does not have enough samples to plot.'.format(well_name)
        else:
            # scaler = StandardScaler()
            # pivScaled = scaler.fit_transform(piv)
            # pivScaled = pd.DataFrame(pivScaled, columns=piv.columns)
            # pivScaled.index = piv.index
            # piv = pivScaled
            
            if(log_transform):
                piv[piv <= 0] = 0.00000001
                temp = piv.copy()
                piv = np.log10(piv)
                if(no_log !=None):
                    for col in no_log:
                        piv[col] = temp[col]

            # Remove outliers
            if(remove_outliers):
                piv = self.remove_outliers(piv, z_threshold=z_threshold)
            samples = piv.shape[0]

            idx = piv.index.date
            dates = [dates.strftime('%Y-%m-%d') for dates in idx]
            remaining = [i for i in dates if i not in remove]
            piv = piv.loc[remaining]
            
            sns.set_style("white", {"axes.facecolor": "0.95"})
            g = sns.PairGrid(piv, aspect=1.2, diag_sharey=False, despine=False)
            g.fig.suptitle(title, fontweight='bold', y=1.08, fontsize=25)
            g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'red', 'lw': 3},
                                                            scatter_kws={'color': 'black', 's': 20})
            g.map_diag(sns.distplot, kde_kws={'color': 'black', 'lw': 3}, hist_kws={'histtype': 'bar', 'lw': 2, 'edgecolor': 'k', 'facecolor':'grey'})
            g.map_upper(self.__plotUpperHalf)
            for ax in g.axes.flat:
                ax.tick_params("y", labelrotation=0, labelsize=fontsize)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=fontsize)
                ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize, fontweight='bold') #HERE
                ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize,fontweight='bold')
                
            g.fig.subplots_adjust(wspace=0.3, hspace=0.3)
            ax = plt.gca()

            props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
            ax.text(1.3, 6.2, 'Start date:  {}\nEnd date:    {}\n\nOriginal samples:     {}\nSamples used:     {}'.format(piv.index[0].date(), piv.index[-1].date(), totalSamples, samples), transform=ax.transAxes, fontsize=20, fontweight='bold', verticalalignment='bottom', bbox=props)
            # Add titles to the diagonal axes/subplots
            for ax, col in zip(np.diag(g.axes), piv.columns):
                ax.set_title(col, y=0.82, fontsize=15)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            g.fig.savefig(save_dir + '/' + well_name + file_extension + '.png', bbox_inches="tight")
            if(returnData):
                return piv
            
    
    # Description: 
    #    Plots the correlations with the physical plots as well as the important analytes over time for each well in the dataset.
    # Parameters:
    #    analytes (list of strings): list of analyte names to use
    #    remove_outliers (bool): choose whether or to remove the outliers.
    #    z_threshold (float): z_score threshold to eliminate outliers
    #    interpolate (bool): choose whether or to interpolate the data
    #    frequency (string): {‘D’, ‘W’, ‘M’, ‘Y’} frequency to interpolate. See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html for valid frequency inputs. (e.g. ‘W’ = every week, ‘D ’= every day, ‘2W’ = every 2 weeks)
    #    save_dir (string): name of the directory you want to save the plot to
    def plot_all_corr_by_well(self, analytes, remove_outliers=True, z_threshold=4, interpolate=False, frequency='2W', save_dir='plot_correlation', log_transform=False, fontsize=20):
        data = self.data
        wells = np.array(data.STATION_ID.values)
        wells = np.unique(wells)
        for well in wells:
            self.plot_corr_by_well(well_name=well, analytes=analytes,remove_outliers=remove_outliers, z_threshold=z_threshold, interpolate=interpolate, frequency=frequency, save_dir=save_dir, log_transform=log_transform, fontsize=fontsize)
        
    # Description: 
    #    Plots the correlations with the physical plots as well as the correlations of the important analytes for ALL the wells on a specified date or range of dates if a lag greater than 0 is specifed.
    # Parameters:
    #    date (string): date to be analyzed
    #    analytes (list of strings): list of analyte names to use
    #    lag (int): number of days to look ahead and behind the specified date (+/-)
    #    min_samples (int): minimum number of samples the result should contain in order to execute.
    #    save_dir (string): name of the directory you want to save the plot to    
    def plot_corr_by_date_range(self, date, analytes, lag=0, min_samples=10, save_dir='plot_corr_by_date', log_transform=False, fontsize=20, returnData=False, no_log=None):
        if(lag==0):
            data = self.data
            data = self.simplify_data(data=data)
            query = data[data.COLLECTION_DATE == date]
            a = list(np.unique(query.ANALYTE_NAME.values))# get all analytes from dataset
            for value in analytes:
                if((value in a)==False):
                    return 'ERROR: No analyte named "{}" in data.'.format(value)
            analytes = sorted(analytes)
            query = query.loc[query.ANALYTE_NAME.isin(analytes)]
            if(query.shape[0] == 0):
                return 'ERROR: {} has no data for all of the analytes.'.format(date)
            samples = query[['COLLECTION_DATE', 'STATION_ID', 'ANALYTE_NAME']].duplicated().value_counts()[0]
            if(samples < min_samples):
                return 'ERROR: {} does not have at least {} samples.'.format(date, min_samples)
            else:
                piv = query.reset_index().pivot_table(index = 'STATION_ID', columns='ANALYTE_NAME', values='RESULT',aggfunc=np.mean)
                # return piv
        else:
            # If the data has already been calculated with the lag specified, retrieve it
            if(self.jointData_is_set(lag=lag)==True): 
                data = self.__jointData[0]
            # Otherwise, calculate it
            else:
                data = self.getJointData(analytes, lag=lag)
                self.set_jointData(data=data, lag=lag)
            # get new range based on the lag and create the pivor table to be able to do the correlation
            dateStart, dateEnd = self.__getLagDate(date, lagDays=lag)
            dateRange_key = str(dateStart.date()) + " - " + str(dateEnd.date())
            piv = pd.DataFrame(data.loc[dateRange_key]).unstack().T
            piv.index = piv.index.droplevel()
            piv = pd.DataFrame(piv).dropna(axis=0, how='all')
            num_NaNs = int(piv.isnull().sum().sum())
            samples = (piv.shape[0]*piv.shape[1])-num_NaNs
            for col in piv.columns:
                piv[col] = piv[col].astype('float64', errors = 'raise')
            if(lag>0):
                date = dateRange_key
            # return piv
        title = date + '_correlation'
        # scaler = StandardScaler()
        # pivScaled = scaler.fit_transform(piv)
        # pivScaled = pd.DataFrame(pivScaled, columns=piv.columns)
        # pivScaled.index = piv.index
        # piv = pivScaled

        if(log_transform):
            piv[piv <= 0] = 0.00000001
            temp = piv.copy()
            piv = np.log10(piv)
            if(no_log !=None):
                for col in no_log:
                    piv[col] = temp[col]

        sns.set_style("white", {"axes.facecolor": "0.95"})
        g = sns.PairGrid(piv, aspect=1.2, diag_sharey=False, despine=False)
        g.fig.suptitle(title, fontweight='bold', y=1.08, fontsize=25)
        g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'red', 'lw': 3},
                                                        scatter_kws={'color': 'black', 's': 20})
        g.map_diag(sns.distplot, kde_kws={'color': 'black', 'lw': 3}, hist_kws={'histtype': 'bar', 'lw': 2, 'edgecolor': 'k', 'facecolor':'grey'})
        g.map_upper(self.__plotUpperHalf)
        for ax in g.axes.flat:
                ax.tick_params("y", labelrotation=0, labelsize=fontsize)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=fontsize)
                ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize, fontweight='bold') #HERE
                ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize,fontweight='bold')
        g.fig.subplots_adjust(wspace=0.3, hspace=0.3)
        ax = plt.gca()

        props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
        ax.text(1.3, 3, 'Date:  {}\n\nWells:     {}\nSamples used:     {}'.format(date, piv.shape[0] ,samples), transform=ax.transAxes, fontsize=20, fontweight='bold', verticalalignment='bottom', bbox=props)
        # Add titles to the diagonal axes/subplots
        for ax, col in zip(np.diag(g.axes), piv.columns):
            ax.set_title(col, y=0.82, fontsize=15)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        g.fig.savefig(save_dir + '/' + date + '.png', bbox_inches="tight")
        if(returnData):
            return piv

    
    # Description: 
    #    Plots the correlations with the physical plots as well as the correlations of the important analytes for ALL the wells in specified year.
    # Parameters:
    #    year (int): year to be analyzed
    #    analytes (list of strings): list of analyte names to use
    #    remove_outliers (bool): choose whether or to remove the outliers.
    #    z_threshold (float): z_score threshold to eliminate outliers
    #    min_samples (int): minimum number of samples the result should contain in order to execute.
    #    save_dir (string): name of the directory you want to save the plot to
    def plot_corr_by_year(self, year, analytes, remove_outliers=True, z_threshold=4, min_samples=10, save_dir='plot_corr_by_year', log_transform=False, fontsize=20, returnData=False, no_log=None):
        data = self.data
        query = data
        query = self.simplify_data(data=query)
        query.COLLECTION_DATE = pd.to_datetime(query.COLLECTION_DATE)
        query = query[query.COLLECTION_DATE.dt.year == year]
        a = list(np.unique(query.ANALYTE_NAME.values))# get all analytes from dataset
        for value in analytes:
            if((value in a)==False):
                return 'ERROR: No analyte named "{}" in data.'.format(value)
        analytes = sorted(analytes)
        query = query.loc[query.ANALYTE_NAME.isin(analytes)]
        if(query.shape[0] == 0):
            return 'ERROR: {} has no data for the 6 analytes.'.format(year)
        samples = query[['COLLECTION_DATE', 'STATION_ID', 'ANALYTE_NAME']].duplicated().value_counts()[0]
        if(samples < min_samples):
            return 'ERROR: {} does not have at least {} samples.'.format(date, min_samples)
        else:
            piv = query.reset_index().pivot_table(index = 'STATION_ID', columns='ANALYTE_NAME', values='RESULT',aggfunc=np.mean)
            # return piv
            # Remove outliers
            if(remove_outliers):
                piv = self.remove_outliers(piv, z_threshold=z_threshold)
            samples = piv.shape[0] * piv.shape[1]

            title = str(year) + '_correlation'
            # scaler = StandardScaler()
            # pivScaled = scaler.fit_transform(piv)
            # pivScaled = pd.DataFrame(pivScaled, columns=piv.columns)
            # pivScaled.index = piv.index
            # piv = pivScaled

            if(log_transform):
                piv[piv <= 0] = 0.00000001
                temp = piv.copy()
                piv = np.log10(piv)
                if(no_log !=None):
                    for col in no_log:
                        piv[col] = temp[col]

            sns.set_style("white", {"axes.facecolor": "0.95"})
            g = sns.PairGrid(piv, aspect=1.2, diag_sharey=False, despine=False)
            g.fig.suptitle(title, fontweight='bold', y=1.08, fontsize=25)
            g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'red', 'lw': 3},
                                                            scatter_kws={'color': 'black', 's': 20})
            g.map_diag(sns.distplot, kde_kws={'color': 'black', 'lw': 3}, hist_kws={'histtype': 'bar', 'lw': 2, 'edgecolor': 'k', 'facecolor':'grey'})
            g.map_upper(self.__plotUpperHalf)
            for ax in g.axes.flat:
                ax.tick_params("y", labelrotation=0, labelsize=fontsize)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=fontsize)
                ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize, fontweight='bold') #HERE
                ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize,fontweight='bold')
            g.fig.subplots_adjust(wspace=0.3, hspace=0.3)
            ax = plt.gca()

            props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
            ax.text(1.3, 3, 'Date:  {}\n\nSamples used:     {}'.format(year, samples), transform=ax.transAxes, fontsize=20, fontweight='bold', verticalalignment='bottom', bbox=props)
            # Add titles to the diagonal axes/subplots
            for ax, col in zip(np.diag(g.axes), piv.columns):
                ax.set_title(col, y=0.82, fontsize=15)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            g.fig.savefig(save_dir + '/' + str(year) + '.png', bbox_inches="tight")
            if(returnData):
                return piv
            
    # Description: 
    #    Plots the linear regression line of data given the analyte_name and well_name. The plot includes the prediction where the line of best fit intersects with the Maximum Concentration Limit (MCL).
    # Parameters:
    #    well_name (string): name of the well to be processed
    #    analyte_name (string): name of the analyte to be processed
    #    year_interval (int): plot by how many years to appear in the axis e.g.(1 = every year, 5 = every 5 years, ...)
    #    save_dir (string): name of the directory you want to save the plot to     
    def plot_MCL(self, well_name, analyte_name, year_interval=5, save_dir='plot_MCL'):
        data = self.data
        # finds the intersection point of 2 lines given the slopes and y-intercepts
        def line_intersect(m1, b1, m2, b2):
            if m1 == m2:
                print ('The lines are parallel')
                return None
            x = (b2 - b1) / (m1 - m2)
            y = m1 * x + b1
            return x,y

        # Gets appropriate data (well_name and analyte_name)
        query = self.query_data(well_name, analyte_name)

        if(type(query)==int and query == 0):
            return 'No results found for {} and {}'.format(well_name, analyte_name)
        else:   

            test = query.groupby(['COLLECTION_DATE']).mean()
            test.index = pd.to_datetime(test.index)

            x = date2num(test.index)
            y = np.log10(test.RESULT)
            ylabel = 'log-Concentration (' + self.get_unit(analyte_name) + ')'
            y = y.rename(ylabel)

            p, cov = np.polyfit(x, y, 1, cov=True)  # parameters and covariance from of the fit of 1-D polynom.

            m_unc = np.sqrt(cov[0][0])
            b_unc = np.sqrt(cov[1][1])

            f = np.poly1d(p)

            try:
                MCL = self.get_MCL(analyte_name)
                m1, b1 = f # line of best fit
                m2, b2 = 0, MCL # MCL constant

                intersection = line_intersect(m1, b1, m2, b2)

                ## Get confidence interval intersection points with MCL
                data = list(zip(x,y))
                n = len(data)
                list_slopes = []
                list_intercepts = []
                random.seed(50)
                for _ in range(80):
                    sampled_data = [ random.choice(data) for _ in range(n) ]
                    x_s, y_s = zip(*sampled_data)
                    x_s = np.array(x_s)
                    y_s = np.array(y_s)

                    m_s, b_s, r, p, err = scipy.stats.linregress(x_s,y_s)
                    ymodel = m_s*x_s + b_s
                    list_slopes.append(m_s)
                    list_intercepts.append(b_s)

                max_index = list_slopes.index(max(list_slopes))
                min_index = list_slopes.index(min(list_slopes))
                intersection_left = line_intersect(list_slopes[min_index], list_intercepts[min_index], m2, b2)
                intersection_right = line_intersect(list_slopes[max_index], list_intercepts[max_index], m2, b2)
                ##

                fig, ax = plt.subplots(figsize=(10, 6))

                ax.set_title(well_name + ' - ' + analyte_name, fontweight='bold')
                ttl = ax.title
                ttl.set_position([.5, 1.05])
                years = mdates.YearLocator(year_interval)  # every year
                months = mdates.MonthLocator()  # every month
                yearsFmt = mdates.DateFormatter('%Y') 

                ax.xaxis.set_major_locator(years)
                ax = plt.gca()
                ax.xaxis.set_major_locator(years)
                ax.xaxis.set_major_formatter(yearsFmt)
                ax.autoscale_view()
                ax.grid(True, alpha=0.4)
                small_fontSize = 15
                large_fontSize = 20
                plt.rc('axes', titlesize=large_fontSize)
                plt.rc('axes', labelsize=large_fontSize)
                plt.rc('legend', fontsize=small_fontSize)
                plt.rc('xtick', labelsize=small_fontSize)
                plt.rc('ytick', labelsize=small_fontSize)

                ax.set_xlabel('Years')
                ax.set_ylabel('log-Concentration (' + self.get_unit(analyte_name) + ')')

                if(intersection[0] < min(x)):
                    temp = intersection_left
                    intersection_left = intersection_right
                    intersection_right = temp
                    ax.set_ylim([0, max(y)+1])
                    ax.set_xlim([intersection_left[0]-1000, max(x)+1000])
                elif(intersection[0] < max(x) and intersection[0] > min(x)):
                    ax.set_ylim([0, max(y)+1])
                    ax.set_xlim(min(x)-1000, max(x)+1000)

                else:
                    ax.set_ylim([0, max(y)+1])
                    ax.set_xlim([min(x)-1000, intersection_right[0]+1000])

                ax = sns.regplot(x, y, logx=True, truncate=False, seed=42, n_boot=1000, ci=95) # Line of best fit
                ax.plot(x, y, ls='', marker='o', ms=5, color='black', alpha=1) # Data
                ax.axhline(y=MCL, color='r', linestyle='--') # MCL
                ax.plot(intersection[0], intersection[1], color='blue', marker='o', ms=10)
                ax.plot(intersection_left[0], intersection_left[1], color='green', marker='o', ms=5)
                ax.plot(intersection_right[0], intersection_right[1], color='green', marker='o', ms=5)

                predict = num2date(intersection[0]).date()
                l_predict = num2date(intersection_left[0]).date()
                u_predict = num2date(intersection_right[0]).date()
                ax.annotate(predict, (intersection[0], intersection[1]), xytext=(intersection[0], intersection[1]+1), 
                            bbox=dict(boxstyle="round", alpha=0.1),ha='center', arrowprops=dict(arrowstyle="->", color='blue'), fontsize=small_fontSize, fontweight='bold')
                props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
                ax.text(1.1, 0.5, 'Lower confidence:  {}\n            Prediction:  {}\nUpper confidence:  {}'.format(l_predict, predict, u_predict), transform=ax.transAxes, fontsize=small_fontSize, fontweight='bold', verticalalignment='bottom', bbox=props)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.savefig(save_dir + '/' + well_name + '-' + analyte_name +'.png', bbox_inches="tight")

            except:
                print('ERROR: Something went wrong')
                return None
    

    # Description: 
    #    Gernates a PCA biplot (PCA score plot + loading plot) of the data given a date in the dataset. The data is also clustered into n_clusters.
    # Parameters:
    #    date (string): date to be analyzed
    #    analytes (list of strings): list of analyte names to use
    #    lag (int): number of days to look ahead and behind the specified date (+/-)
    #    n_clusters (int): number of clusters to split the data into.
    #    filter (bool): Flag to indicate well filtering.
    #    col (string): column name from the construction dataset that you want to filter by
    #    equals (list of strings): value(s) to filter by in column col
    #    return_clusters (bool): Flag to return the cluster data to be used for spatial plotting.
    #    min_samples (int): minimum number of samples the result should contain in order to execute.
    #    show_labels (bool): choose whether or not to show the name of the wells.
    #    save_dir (string): name of the directory you want to save the plot to
    def plot_PCA_by_date(self, date, analytes, lag=0, n_clusters=4, return_clusters=False, min_samples=3, show_labels=True, save_dir='plot_PCA_by_date', filter=False, col=None, equals=[]):
        if(lag==0):
            data = self.data
            data = self.simplify_data(data=data)
            query = data[data.COLLECTION_DATE == date]
            if(filter):
                filter_res = self.filter_by_column(data=self.construction_data, col=col, equals=equals)
                if('ERROR:' in str(filter_res)):
                    return filter_res
                query_wells = list(query.STATION_ID.unique())
                filter_wells = list(filter_res.index.unique())
                intersect_wells = list(set(query_wells) & set(filter_wells))
                if(len(intersect_wells)<=0):
                    return 'ERROR: No results for this query with the specifed filter parameters.'
                query = query[query['STATION_ID'].isin(intersect_wells)]
            a = list(np.unique(query.ANALYTE_NAME.values))# get all analytes from dataset
            for value in analytes:
                if((value in a)==False):
                    return 'ERROR: No analyte named "{}" in data.'.format(value)
            analytes = sorted(analytes)
            query = query.loc[query.ANALYTE_NAME.isin(analytes)]

            if(query.shape[0] == 0):
                return 'ERROR: {} has no data for the 6 analytes.'.format(date)
            samples = query[['COLLECTION_DATE', 'STATION_ID', 'ANALYTE_NAME']].duplicated().value_counts()[0]
            if(samples < min_samples):
                return 'ERROR: {} does not have at least {} samples.'.format(date, min_samples)
            # if(len(np.unique(query.ANALYTE_NAME.values)) < 6):
            #     return 'ERROR: {} has less than the 6 analytes we want to analyze.'.format(date)
            else:
                # analytes = self.__custom_analyte_sort(np.unique(query.ANALYTE_NAME.values))
                analytes = sorted(analytes)
                piv = query.reset_index().pivot_table(index = 'STATION_ID', columns='ANALYTE_NAME', values='RESULT',aggfunc=np.mean)
                
                # return piv
        else:
            # If the data has already been calculated with the lag specified, retrieve it
            if(self.jointData_is_set(lag=lag)==True): 
                data = self.__jointData[0]
            # Otherwise, calculate it
            else:
                data = self.getJointData(analytes, lag=lag)
                self.set_jointData(data=data, lag=lag)
            # get new range based on the lag and create the pivor table to be able to do the correlation
            dateStart, dateEnd = self.__getLagDate(date, lagDays=lag)
            dateRange_key = str(dateStart.date()) + " - " + str(dateEnd.date())
            piv = pd.DataFrame(data.loc[dateRange_key]).unstack().T
            piv.index = piv.index.droplevel()
            piv = pd.DataFrame(piv).dropna(axis=0, how='all')
            num_NaNs = int(piv.isnull().sum().sum())
            samples = (piv.shape[0]*piv.shape[1])-num_NaNs
            for col in piv.columns:
                piv[col] = piv[col].astype('float64', errors = 'raise')
            if(lag>0):
                date = dateRange_key
            # return piv

            
        main_data = piv.dropna()
        
        scaler = StandardScaler()
        X = scaler.fit_transform(main_data)
        pca = PCA(n_components=2)
        x_new = pca.fit_transform(X)
        
        pca_points = pd.DataFrame(x_new, columns=["x1", "x2"])
        k_Means = KMeans(n_clusters=n_clusters, random_state=42)
        model = k_Means.fit(pca_points[['x1', 'x2']])
        predict = model.predict(pca_points[['x1', 'x2']])
        # attach predicted cluster to original points
        pca_points['predicted'] = model.labels_
        # Create a dataframe for cluster_centers (centroids)
        centroids = pd.DataFrame(model.cluster_centers_, columns=["x1", "x2"])
        colors = ['red', 'blue', 'orange', 'purple', 'green', 'beige', 'pink', 'black', 'cadetblue', 'lightgreen']
        pca_points['color'] = pca_points['predicted'].map(lambda p: colors[p])

        fig, ax = plt.subplots(figsize=(10,10))
        ax = plt.axes()

        small_fontSize = 15
        large_fontSize = 20
        plt.rc('axes', titlesize=large_fontSize)
        plt.rc('axes', labelsize=large_fontSize)
        plt.rc('legend', fontsize=small_fontSize)
        plt.rc('xtick', labelsize=small_fontSize)
        plt.rc('ytick', labelsize=small_fontSize)

        def myplot(score,coeff,labels=None,c='r', centroids=None):
            xs = score.iloc[:,0]
            ys = score.iloc[:,1]
            n = coeff.shape[0]
            scalex = 1.0/(xs.max() - xs.min())
            scaley = 1.0/(ys.max() - ys.min())
            scatt_X = xs * scalex
            scatt_Y = ys * scaley
            scatter = plt.scatter(scatt_X, scatt_Y, alpha=0.8, label='Wells', c=c)
            centers = plt.scatter(centroids.iloc[:,0]* scalex, centroids.iloc[:,1]* scaley,
                                    c = colors[0:n_clusters],
                                    marker='X', s=550)

            for i in range(n):
                arrow = plt.arrow(0, 0, coeff[i,0], coeff[i,1], color = 'r', alpha = 0.9, head_width=0.05, head_length=0.05, label='Loadings')
                if labels is None:
                    plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
                else:
                    plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'bottom')

            if(show_labels):
                for x_pos, y_pos, label in zip(scatt_X, scatt_Y, main_data.index):
                    ax.annotate(label, # The label for this point
                    xy=(x_pos, y_pos), # Position of the corresponding point
                    xytext=(7, 0),     # Offset text by 7 points to the right
                    textcoords='offset points', # tell it to use offset points
                    ha='left',         # Horizontally aligned to the left
                    va='center',       # Vertical alignment is centered
                    color='black', alpha=0.8)
            plt.legend( [scatter, centers, arrow], ['Wells', 'Well centroids','Loadings'])

        samples = x_new.shape[0]*piv.shape[1]
        props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
        ax.text(1.1, 0.5, 'Date:  {}\n\nSamples:          {}\nWells:               {}'.format(date, samples, x_new.shape[0]), 
                    transform=ax.transAxes, fontsize=20, fontweight='bold', verticalalignment='bottom', bbox=props)

        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.xlabel("PC{}".format(1))
        plt.ylabel("PC{}".format(2))
        ax.set_title('PCA Biplot - ' + date, fontweight='bold')
        plt.grid(alpha=0.5)

        #Call the function. Use only the 2 PCs.
        myplot(pca_points,np.transpose(pca.components_[0:2, :]), labels=piv.columns, c=pca_points['color'], centroids=centroids)
        plt.show()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(save_dir + '/' + 'PCA Biplot - '+ date +'.png', bbox_inches="tight")
        
        if(return_clusters):
            stations = list(main_data.index)
            color_wells = list(pca_points.color)
            def merge(list1, list2): 
                merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 
                return merged_list
            color_df = pd.DataFrame(merge(stations, color_wells), columns=['STATION_ID', 'color'])
            if(self.get_Construction_Data==None):
                print('You need to set the GPS data first using the getConstructionData function.')
                return None
            else:
                gps_color = pd.merge(self.get_Construction_Data(), color_df, on=['STATION_ID'])
                return gps_color
    
    
    # Description: 
    #    Gernates a PCA biplot (PCA score plot + loading plot) of the data given a year in the dataset. The data is also clustered into n_clusters.
    # Parameters:
    #    year (int): year to be analyzed
    #    analytes (list of strings): list of analyte names to use
    #    lag (int): number of days to look ahead and behind the specified date (+/-)
    #    n_clusters (int): number of clusters to split the data into.
    #    filter (bool): Flag to indicate well filtering.
    #    col (string): column name from the construction dataset that you want to filter by
    #    equals (list of strings): value(s) to filter by in column col
    #    return_clusters (bool): Flag to return the cluster data to be used for spatial plotting.
    #    min_samples (int): minimum number of samples the result should contain in order to execute.
    #    show_labels (bool): choose whether or not to show the name of the wells.
    #    save_dir (string): name of the directory you want to save the plot to
    def plot_PCA_by_year(self, year, analytes, n_clusters=4, return_clusters=False, min_samples=10, show_labels=True, save_dir='plot_PCA_by_year', filter=False, col=None, equals=[]):
        data = self.data
        query = self.simplify_data(data=data)
        query.COLLECTION_DATE = pd.to_datetime(query.COLLECTION_DATE)
        query = query[query.COLLECTION_DATE.dt.year == year]
        if(filter):
            filter_res = self.filter_by_column(data=self.construction_data, col=col, equals=equals)
            if('ERROR:' in str(filter_res)):
                return filter_res
            query_wells = list(query.STATION_ID.unique())
            filter_wells = list(filter_res.index.unique())
            intersect_wells = list(set(query_wells) & set(filter_wells))
            if(len(intersect_wells)<=0):
                return 'ERROR: No results for this query with the specifed filter parameters.'
            query = query[query['STATION_ID'].isin(intersect_wells)]
        a = list(np.unique(query.ANALYTE_NAME.values))# get all analytes from dataset
        for value in analytes:
            if((value in a)==False):
                return 'ERROR: No analyte named "{}" in data.'.format(value)
        analytes = sorted(analytes)
        query = query.loc[query.ANALYTE_NAME.isin(analytes)]

        if(query.shape[0] == 0):
            return 'ERROR: {} has no data for the 6 analytes.'.format(year)
        samples = query[['COLLECTION_DATE', 'STATION_ID', 'ANALYTE_NAME']].duplicated().value_counts()[0]
        if(samples < min_samples):
            return 'ERROR: {} does not have at least {} samples.'.format(year, min_samples)
        # if(len(np.unique(query.ANALYTE_NAME.values)) < 6):
        #     return 'ERROR: {} has less than the 6 analytes we want to analyze.'.format(year)
        else:
            # analytes = self.__custom_analyte_sort(np.unique(query.ANALYTE_NAME.values))
            analytes = sorted(analytes)
            piv = query.reset_index().pivot_table(index = 'STATION_ID', columns='ANALYTE_NAME', values='RESULT',aggfunc=np.mean)
            
            main_data = piv.dropna()
            # # FILTERING CODE
            # if(filter):
            #     res_wells = self.filter_wells(filter_well_by)
            #     main_data = main_data.loc[main_data.index.isin(res_wells)]
            
            scaler = StandardScaler()
            X = scaler.fit_transform(main_data)
            pca = PCA(n_components=2)
            x_new = pca.fit_transform(X)

            pca_points = pd.DataFrame(x_new, columns=["x1", "x2"])
            k_Means = KMeans(n_clusters=n_clusters, random_state=42)
            model = k_Means.fit(pca_points[['x1', 'x2']])
            predict = model.predict(pca_points[['x1', 'x2']])
            # attach predicted cluster to original points
            pca_points['predicted'] = model.labels_
            # Create a dataframe for cluster_centers (centroids)
            centroids = pd.DataFrame(model.cluster_centers_, columns=["x1", "x2"])
            colors = ['red', 'blue', 'orange', 'purple', 'green', 'beige', 'pink', 'black', 'cadetblue', 'lightgreen']
            pca_points['color'] = pca_points['predicted'].map(lambda p: colors[p])
            
            fig, ax = plt.subplots(figsize=(15,15))
            ax = plt.axes()

            small_fontSize = 15
            large_fontSize = 20
            plt.rc('axes', titlesize=large_fontSize)
            plt.rc('axes', labelsize=large_fontSize)
            plt.rc('legend', fontsize=small_fontSize)
            plt.rc('xtick', labelsize=small_fontSize)
            plt.rc('ytick', labelsize=small_fontSize) 

            def myplot(score,coeff,labels=None,c='r', centroids=None):
                xs = score[:,0]
                ys = score[:,1]
                n = coeff.shape[0]
                scalex = 1.0/(xs.max() - xs.min())
                scaley = 1.0/(ys.max() - ys.min())
                scatt_X = xs * scalex
                scatt_Y = ys * scaley
                scatter = plt.scatter(scatt_X, scatt_Y, alpha=0.8, label='Wells', c=c)
                centers = plt.scatter(centroids.iloc[:,0]* scalex, centroids.iloc[:,1]* scaley,
                                      c = colors[0:n_clusters],
                                      marker='X', s=550)
                for i in range(n):
                    arrow = plt.arrow(0, 0, coeff[i,0], coeff[i,1], color = 'r', alpha = 0.9, head_width=0.05, head_length=0.05)
                    if labels is None:
                        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
                    else:
                        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'bottom')

                if(show_labels):
                    for x_pos, y_pos, label in zip(scatt_X, scatt_Y, main_data.index):
                        ax.annotate(label, # The label for this point
                        xy=(x_pos, y_pos), # Position of the corresponding point
                        xytext=(7, 0),     # Offset text by 7 points to the right
                        textcoords='offset points', # tell it to use offset points
                        ha='left',         # Horizontally aligned to the left
                        va='center', color='black', alpha=0.8)       # Vertical alignment is centered
                plt.legend( [scatter, centers, arrow], ['Wells', 'Well centroids','Loadings'])

            samples = x_new.shape[0]*piv.shape[1]    
            props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
            ax.text(1.1, 0.5, 'Date:  {}\n\nSamples:          {}\nWells:               {}'.format(year,samples, x_new.shape[0]), 
                        transform=ax.transAxes, fontsize=20, fontweight='bold', verticalalignment='bottom', bbox=props)

            plt.xlim(-1,1)
            plt.ylim(-1,1)
            plt.xlabel("PC{}".format(1))
            plt.ylabel("PC{}".format(2))
            ax.set_title('PCA Biplot - ' + str(year), fontweight='bold')
            plt.grid(alpha=0.5)

            #Call the function. Use only the 2 PCs.
            myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]), labels=piv.columns, c=pca_points['color'], centroids=centroids)

            plt.show()
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fig.savefig(save_dir + '/' + 'PCA Biplot - '+ str(year) +'.png', bbox_inches="tight")
            
            if(return_clusters):
                stations = list(main_data.index)
                color_wells = list(pca_points.color)
                def merge(list1, list2): 
                    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 
                    return merged_list
                color_df = pd.DataFrame(merge(stations, color_wells), columns=['STATION_ID', 'color'])
                if(self.get_Construction_Data==None):
                    print('You need to set the GPS data first using the setConstructionData function.')
                    return None
                else:
                    gps_color = pd.merge(self.get_Construction_Data(), color_df, on=['STATION_ID'])
                    return gps_color
    
    # Description: 
    #    Gernates a PCA biplot (PCA score plot + loading plot) of the data given a well_name in the dataset. Only uses the 6 important analytes.
    # Parameters:
    #    well_name (string): name of the well to be processed
    #    interpolate (bool): choose whether or to interpolate the data
    #    frequency (string): {‘D’, ‘W’, ‘M’, ‘Y’} frequency to interpolate. See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html for valid frequency inputs. (e.g. ‘W’ = every week, ‘D ’= every day, ‘2W’ = every 2 weeks)
    #    min_samples (int): minimum number of samples the result should contain in order to execute.
    #    show_labels (bool): choose whether or not to show the name of the wells.
    #    save_dir (string): name of the directory you want to save the plot to
    def plot_PCA_by_well(self, well_name, analytes, interpolate=False, frequency='2W', min_samples=10, show_labels=True, save_dir='plot_PCA_by_well'):
        data = self.data
        query = data[data.STATION_ID == well_name]
        a = list(np.unique(query.ANALYTE_NAME.values))# get all analytes from dataset
        for value in analytes:
            if((value in a)==False):
                return 'ERROR: No analyte named "{}" in data.'.format(value)
        analytes = sorted(analytes)
        query = query.loc[query.ANALYTE_NAME.isin(analytes)]
        x = query[['COLLECTION_DATE', 'ANALYTE_NAME']]
        unique = ~x.duplicated()
        query = query[unique]
        piv = query.reset_index().pivot(index='COLLECTION_DATE',columns='ANALYTE_NAME', values='RESULT')
        piv = piv[analytes]
        piv.index = pd.to_datetime(piv.index)
        totalSamples = piv.stack().shape[0]
        piv = piv.dropna()
        if(interpolate):
            piv = self.interpolate_well_data(well_name, analytes, frequency=frequency)
            title = 'PCA Biplot - ' + well_name + ' - interpolated every ' + frequency
        else:
            title = 'PCA Biplot - ' + well_name

        if(query.shape[0] == 0):
            return 'ERROR: {} has no data for the 6 analytes.'.format(date)
        samples = query[['COLLECTION_DATE', 'STATION_ID', 'ANALYTE_NAME']].duplicated().value_counts()[0]
        if(samples < min_samples):
            return 'ERROR: {} does not have at least {} samples.'.format(date, min_samples)
        # if(len(np.unique(query.ANALYTE_NAME.values)) < 6):
        #     return 'ERROR: {} has less than the 6 analytes we want to analyze.'.format(well_name)
        else:
            scaler = StandardScaler()
            X = scaler.fit_transform(piv.dropna())
            pca = PCA(n_components=2)
            x_new = pca.fit_transform(X)

            fig, ax = plt.subplots(figsize=(15,15))
            ax = plt.axes()

            small_fontSize = 15
            large_fontSize = 20
            plt.rc('axes', titlesize=large_fontSize)
            plt.rc('axes', labelsize=large_fontSize)
            plt.rc('legend', fontsize=small_fontSize)
            plt.rc('xtick', labelsize=small_fontSize)
            plt.rc('ytick', labelsize=small_fontSize) 

            def myplot(score,coeff,labels=None):
                xs = score[:,0]
                ys = score[:,1]
                n = coeff.shape[0]
                scalex = 1.0/(xs.max() - xs.min())
                scaley = 1.0/(ys.max() - ys.min())
                scatt_X = xs * scalex
                scatt_Y = ys * scaley
                scatter = plt.scatter(scatt_X, scatt_Y, alpha=0.8, label='Date samples')

                for i in range(n):
                    arrow = plt.arrow(0, 0, coeff[i,0], coeff[i,1], color = 'r', alpha = 0.9, head_width=0.05, head_length=0.05, label='Loadings')
                    if labels is None:
                        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
                    else:
                        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'bottom')

                if(show_labels):
                    for x_pos, y_pos, label in zip(scatt_X, scatt_Y, piv.dropna().index.date):
                        ax.annotate(label, # The label for this point
                        xy=(x_pos, y_pos), # Position of the corresponding point
                        xytext=(7, 0),     # Offset text by 7 points to the right
                        textcoords='offset points', # tell it to use offset points
                        ha='left',         # Horizontally aligned to the left
                        va='center',       # Vertical alignment is centered
                        color='black', alpha=0.8)
                plt.legend( [scatter, arrow], ['Date samples', 'Loadings'])

            samples = x_new.shape[0]*piv.shape[1]
            props = dict(boxstyle='round', facecolor='grey', alpha=0.15)      
            ax.text(1.1, 0.5, 'Start date:  {}\nEnd date:    {}\n\nOriginal samples:     {}\nSamples used:     {}\nDate samples:               {}'
                        .format(piv.index[0].date(), piv.index[-1].date(), totalSamples, samples, x_new.shape[0]), 
                        transform=ax.transAxes, fontsize=20, fontweight='bold', verticalalignment='bottom', bbox=props)

            plt.xlim(-1,1)
            plt.ylim(-1,1)
            plt.xlabel("PC{}".format(1))
            plt.ylabel("PC{}".format(2))
            ax.set_title(title, fontweight='bold')
            plt.grid(alpha=0.5)

            #Call the function. Use only the 2 PCs.
            myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]), labels=piv.columns)
            plt.show()
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fig.savefig(save_dir + '/' + title +'.png', bbox_inches="tight")
            
    # Description: 
    #    Plots the well locations on an interactive map given coordinates.
    # Parameters:
    #    gps_data (dataframe): Data frame with the following column names: station_id, latitude, longitude, color. If the color column is not passed, the default color will be blue.
    #    center (list with 2 floats): latitude and longitude coordinates to center the map view.
    #    zoom (int): value to determine the initial scale of the map
    def plot_coordinates_to_map(self, gps_data, center=[33.271459, -81.675873], zoom=14):
        center = center
        zoom = 14
        m = Map(basemap=basemaps.Esri.WorldImagery, center=center, zoom=zoom)

        m.add_control(FullScreenControl())
        for (index,row) in gps_data.iterrows():

            if('color' in gps_data.columns):
                icon = AwesomeIcon(
                    name='tint',
                    marker_color=row.loc['color'],
                    icon_color='black',
                    spin=False
                )
            else:
                icon = AwesomeIcon(
                    name='tint',
                    marker_color='blue',
                    icon_color='black',
                    spin=False
                )

            loc = [row.loc['LATITUDE'],row.loc['LONGITUDE']]
            station = HTML(value=row.loc['STATION_ID'])

            marker = Marker(location=loc,
                            icon=icon,
                            draggable=False,
                       )

            m.add_layer(marker)

            popup = Popup(child=station,
                          max_height=1)

            marker.popup = popup

        return m


    # Description: 
    #    Resamples analyte data based on the frequency specified and interpolates the values in between. NaN values are replaced with the average value per well.
    # Parameters:
    #    analyte (string): analyte name for interpolation of all present wells.
    #    frequency (string): {‘D’, ‘W’, ‘M’, ‘Y’} frequency to interpolate. See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html for valid frequency inputs. (e.g. ‘W’ = every week, ‘D ’= every day, ‘2W’ = every 2 weeks)
    #    rm_outliers (bool): flag to remove outliers in the data
    #    z_threshold (int): z_score threshold to eliminate outliers
    def interpolate_wells_by_analyte(self, analyte, frequency='2W', rm_outliers=True, z_threshold=3):
        data = self.data
        df_t, dates = self.transform_time_series( 
                                                 analytes=[analyte], 
                                                 resample=frequency, 
                                                 rm_outliers=True, 
                                                 z_threshold=z_threshold)
        res_interp = self.get_individual_analyte_df(data=df_t, dates=dates, analyte=analyte)
        res_interp = res_interp.dropna(axis=1, how='all')
        return res_interp
    
    # IN THE WORKS
    def transform_time_series(self, analytes=[], resample='2W', rm_outliers=False, z_threshold=4):
        data = self.data
        def transform_time_series_by_analyte(data, analyte_name):
            wells_analyte = np.unique(data[data.ANALYTE_NAME == analyte_name].STATION_ID)
            condensed = data[data.ANALYTE_NAME == analyte_name].groupby(['STATION_ID','COLLECTION_DATE']).mean()
            analyte_df_resample = pd.DataFrame(index=wells_analyte, columns=t)
            analyte_df_resample.sort_index(inplace=True)
            for well in wells_analyte:
                for date in condensed.loc[well].index:
                    analyte_df_resample.at[well, date] = condensed.loc[well,date].RESULT
            analyte_df_resample = analyte_df_resample.astype('float').T
            analyte_df_resample = analyte_df_resample.interpolate(method='linear')
            return analyte_df_resample

        all_dates = np.unique(data.COLLECTION_DATE)
        # Create array of equally spaced dates
        start = pd.Timestamp(all_dates.min())
        end = pd.Timestamp(all_dates.max())
        delta = end - start
        t = np.linspace(start.value, end.value, delta.days)
        t = pd.to_datetime(t)
        t = pd.Series(t)
        t = t.apply(lambda x: x.replace(minute=0, hour=0, second=0, microsecond=0, nanosecond=0))

        cutoff_dates = []
        # Save each analyte data
        analyte_data = []
        for analyte in analytes:
            ana_data = transform_time_series_by_analyte(data, analyte)
            if(rm_outliers):
                col_num = ana_data.shape[1]
                for col in range(col_num):
                    ana_data.iloc[:,col] = self.remove_outliers(ana_data.iloc[:,col].dropna(), z_threshold=z_threshold)
                ana_data = ana_data.interpolate(method='linear')
            ana_data.index = pd.to_datetime(ana_data.index)
            # Resample
            ana_data_resample = ana_data.resample(resample).mean()
            # Save data
            analyte_data.append(ana_data_resample)
            # Determine cuttoff point for number of NaNs in dataset
            passes_limit = []
            for date in ana_data_resample.index:
                limit = 0.7 * ana_data_resample.shape[1]
                curr = ana_data_resample.isna().loc[date,:].value_counts()
                if('False' in str(curr)):
                    curr_total = ana_data_resample.isna().loc[date,:].value_counts()[0]
                    if curr_total > limit:
                        passes_limit.append(date)
            passes_limit = pd.to_datetime(passes_limit)
            cutoff_dates.append(passes_limit.min())
        start_index = pd.Series(cutoff_dates).max()

        # Get list of shared wells amongst all the listed analytes
        combined_well_list = []
        for x in range(len(analytes)):
            combined_well_list = combined_well_list + list(analyte_data[x].columns)
        combined_count = pd.Series(combined_well_list).value_counts()
        shared_wells = list(combined_count[list(pd.Series(combined_well_list).value_counts()==len(analytes))].index)

        # Vectorize data
        vectorized_df = pd.DataFrame(columns=analytes, index = shared_wells)

        for analyte, num in zip(analytes, range(len(analytes))):
            for well in shared_wells:
                analyte_data_full = analyte_data[num][well].fillna(analyte_data[num][well].mean())
                vectorized_df.at[well, analyte] = analyte_data_full[start_index:].values

        dates = ana_data_resample[start_index:].index
        return vectorized_df, dates
    
    def get_individual_analyte_df(self, data, dates, analyte):
        sample = data[analyte]
        sample_analyte = pd.DataFrame(sample, index=dates, columns=sample.index)
        for well in sample.index:
            sample_analyte[well] = sample[well]
        return sample_analyte
    
    def cluster_data_OLD(self, data, n_clusters=4, log_transform=False, filter=False, filter_well_by=['D'], return_clusters=False):
        if(filter):
            res_wells = self.filter_wells(filter_well_by)
            data = data.T
            data = data.loc[data.index.isin(res_wells)]
            data = data.T
        if(log_transform):
            data = np.log10(data)
            data = data.dropna(axis=1)
        temp = data.T
        k_Means = KMeans(n_clusters=n_clusters, random_state=42)
        km = k_Means.fit(temp)
        predict = km.predict(temp)
        temp['predicted'] = km.labels_
        colors = ['red', 'blue', 'orange', 'purple', 'green', 'beige', 'pink', 'black', 'cadetblue', 'lightgreen']
        temp['color'] = temp['predicted'].map(lambda p: colors[p])

        fig, ax = plt.subplots(figsize=(20,10))
        ax = plt.axes()

        color = temp['color']
        for x in range(temp.shape[0]):
            curr = data.iloc[:,x]
            ax.plot(curr, label=curr.name, color=color[x])
            ax.legend()
        
        if(return_clusters):
            color_df = pd.DataFrame(temp['color'])
            color_df['STATION_ID'] = color_df.index
            if(self.get_Construction_Data==None):
                print('You need to set the GPS data first using the setConstructionData function.')
                return None
            else:
                gps_color = pd.merge(self.get_Construction_Data(), color_df, on=['STATION_ID'])
                return gps_color

    def cluster_data(self, data, analyte_name=["ANALYTE_NAME"], n_clusters=4, filter=False, col=None, equals=[], year_interval=5, y_label = 'Concentration', return_clusters=False ):
        data = data.copy()
        if(filter):
            filter_res = self.filter_by_column(data=self.get_Construction_Data(), col=col, equals=equals)
            if('ERROR:' in str(filter_res)):
                return filter_res
            query_wells = list(data.columns)
            filter_wells = list(filter_res.index.unique())
            intersect_wells = list(set(query_wells) & set(filter_wells))
            print(intersect_wells)
            if(len(intersect_wells)<=0):
                return 'ERROR: No results for this query with the specifed filter parameters.'
            data = data[intersect_wells]
        data.index = date2num(data.index)
        temp = data.T
        k_Means = KMeans(n_clusters=n_clusters, random_state=43)
        km = k_Means.fit(temp)
        predict = km.predict(temp)
        temp['predicted'] = km.labels_
        colors = ['red', 'blue', 'orange', 'purple', 'green', 'pink', 'black', 'cadetblue', 'lightgreen','beige']
        temp['color'] = temp['predicted'].map(lambda p: colors[p])
        
        fig, ax = plt.subplots(figsize=(10,10), dpi=100)
        ax.minorticks_off()
        ax = plt.axes()

        color = temp['color']
        for x in range(temp.shape[0]):
            curr = data.iloc[:,x]
            ax.plot(curr, label=curr.name, color=color[x])
            
        years = mdates.YearLocator(year_interval)  # every year
        months = mdates.MonthLocator()  # every month
        yearsFmt = mdates.DateFormatter('%Y') 
        ax = plt.gca()
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.autoscale_view()
        ax.set_xlabel("Years", fontsize=20)
        plt.xticks(fontsize=20)
        ax.set_ylabel(y_label, fontsize=20)
        plt.yticks(fontsize=20)
        ax.set_title("{}: {} clusters".format(analyte_name, n_clusters), fontsize=20)
        if(return_clusters):
            color_df = pd.DataFrame(temp['color'])
            color_df['STATION_ID'] = color_df.index
            if(self.get_Construction_Data==None):
                print('You need to set the GPS data first using the setConstructionData function.')
                return None
            else:
                gps_color = pd.merge(self.get_Construction_Data(), color_df, on=['STATION_ID'])
                return gps_color
        # if(return_data):
        #     return color, temp

    def plot_all_time_series_simple(self, analyte_name=None, start_date=None, end_date=None, title='Dataset: Time ranges', x_label='Well', y_label='Year',
                             min_days=10, x_min_lim=-5, x_max_lim = 170, y_min_date='1988-01-01', y_max_date='2020-01-01', return_data=False, filter=False, col=None, equals=[]):
        data = self.simplify_data()
        if(filter):
            filter_res = self.filter_by_column(data=self.construction_data, col=col, equals=equals)
            if('ERROR:' in str(filter_res)):
                return filter_res
            query_wells = list(data.STATION_ID.unique())
            filter_wells = list(filter_res.index.unique())
            intersect_wells = list(set(query_wells) & set(filter_wells))
            if(len(intersect_wells)<=0):
                return 'ERROR: No results for this query with the specifed filter parameters.'
            data = data[data['STATION_ID'].isin(intersect_wells)]

        if(analyte_name!=None):
            data = data[data.ANALYTE_NAME == analyte_name]
        wells = data.STATION_ID.unique()
        wells_dateRange=pd.DataFrame(columns=['STATION_ID','START_DATE','END_DATE'])
        for i in range(len(wells)):
            wellName=wells[i]
            wellNamedData=data[data['STATION_ID']==wells[i]]
            minDate=min(wellNamedData['COLLECTION_DATE'])
            maxDate=max(wellNamedData['COLLECTION_DATE'])
            wells_dateRange.loc[wells_dateRange.shape[0]]=[wellName,minDate,maxDate]

        wells_dateRange["RANGE"] = wells_dateRange.END_DATE - wells_dateRange.START_DATE
        wells_dateRange.RANGE = wells_dateRange.RANGE.astype('timedelta64[D]').astype('int')
        wells_dateRange = wells_dateRange[wells_dateRange.RANGE>min_days]
        wells_dateRange.sort_values(by=["RANGE","END_DATE","START_DATE"], ascending = (False, False, True), inplace=True)
        wells_dateRange.reset_index(inplace=True)
        wells_dateRange.drop('index', axis=1, inplace=True)
        wells = np.array(wells_dateRange.STATION_ID)

        fig, ax = plt.subplots(1, 1, sharex=False,figsize=(20,6),dpi=300)

        ax.set_xticks(range(len(wells)))
        ax.set_xticklabels(wells, rotation='vertical', fontsize=6)

        ax.plot(wells_dateRange['START_DATE'], c='blue', marker='o',lw=0, label='Start date')
        ax.plot(wells_dateRange['END_DATE'], c='red', marker='o',lw=0, label='End date')

        ax.hlines([max(wells_dateRange['END_DATE'])], x_min_lim, x_max_lim, colors='purple', label='Selected end date')
        if(start_date==None):
            ax.hlines([min(wells_dateRange['START_DATE'])], x_min_lim, x_max_lim, colors='green', label='Selected start date')
        else:
            ax.hlines([pd.to_datetime(start_date)], x_min_lim, x_max_lim, colors='green', label='Selected start date')

        x_label = x_label + ' (count: ' + str(wells_dateRange.shape[0])+ ')'
        ax.set_xlabel(x_label, fontsize=20)
        ax.set_ylabel(y_label, fontsize=20)   
        ax.set_xlim([x_min_lim, x_max_lim])
        ax.set_ylim([pd.to_datetime(y_min_date), pd.to_datetime(y_max_date)]) 
        ax.plot([], [], ' ', label="Time series with at least {} days".format(min_days))
        ax.legend()
        if(analyte_name!=None):
            title = title + ' (' + analyte_name + ')'
        fig.suptitle(title, fontsize=20)
        for i in range(wells_dateRange.shape[0]):
            ax.vlines(i,wells_dateRange.loc[i,'START_DATE'],wells_dateRange.loc[i,'END_DATE'],colors='k')
        if(return_data):
            return wells_dateRange


    def plot_all_time_series(self, analyte_name=None, title='Dataset: Time ranges', x_label='Well', y_label='Year', x_label_size=8, marker_size=30,
                            min_days=10, x_min_lim=-5, x_max_lim = 170, y_min_date='1988-01-01', y_max_date='2020-01-01', sort_by_distance=True, basin_coordinate=[436642.70,3681927.09], log_transform=False, cmap=mpl.cm.rainbow, 
                            drop_cols=[], return_data=False, filter=False, col=None, equals=[], cbar_min=None, cbar_max=None, reverse_y_axis=False, fontsize = 20, figsize=(20,6), dpi=300, y_2nd_label=None):
        dt = self.getCleanData([analyte_name])
        dt = dt[analyte_name] 
        if(filter):
            filter_res = self.filter_by_column(data=self.get_Construction_Data(), col=col, equals=equals)
            if('ERROR:' in str(filter_res)):
                return filter_res
            query_wells = list(dt.columns.unique())
            filter_wells = list(filter_res.index.unique())
            intersect_wells = list(set(query_wells) & set(filter_wells) & set(dt.columns))
            if(len(intersect_wells)<=0):
                return 'ERROR: No results for this query with the specifed filter parameters.'
            dt = dt[intersect_wells]
        
        dt = dt.interpolate()
        well_info = self.get_Construction_Data()
        shared_wells = list(set(well_info.index) & set(dt.columns))
        dt = dt[shared_wells]
        well_info = well_info.T[shared_wells]
        dt = dt.reindex(sorted(dt.columns), axis=1)
        well_info = well_info.reindex(sorted(well_info.columns), axis=1)
        well_info = well_info.T
        transformer = Transformer.from_crs("epsg:4326", "epsg:26917") # Latitude/Longitude to UTM
        UTM_x, UTM_y = transformer.transform(well_info.LATITUDE, well_info.LONGITUDE)
        X = np.vstack((UTM_x,UTM_y)).T
        well_info = pd.DataFrame(X, index=list(well_info.index),columns=['Easting', 'Northing'])
        well_info = self.add_dist_to_basin(well_info, basin_coordinate=basin_coordinate)
        if(sort_by_distance):
            well_info.sort_values(by=['dist_to_basin'], ascending = True, inplace=True)
        dt = dt[well_info.index]
        dt = dt.drop(drop_cols, axis=1) # DROP BAD ONES 
        
        if(log_transform):
            dt[dt <= 0] = 0.00000001
            dt = np.log10(dt)
        wells = dt.columns
        
        if(cbar_min==None):
            cbar_min = dt.min().min()
        if(cbar_max==None):
            cbar_max = dt.max().max() 
        norm = mpl.colors.Normalize(vmin=cbar_min, vmax=cbar_max)

        fig, ax = plt.subplots(1, 2, sharex=False, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': [40, 1]})
        ax[0].set_xticks(range(len(wells)))
        ax[0].set_xticklabels(wells, rotation='vertical', fontsize=x_label_size)

        for col in wells:
            curr_start = dt[col].first_valid_index()
            curr_end =  dt[col].last_valid_index()
            length = len(list(dt[col].loc[curr_start:curr_end].index))
            color_vals = list(dt[col].loc[curr_start:curr_end])
            color_vals = [cmap(norm(x)) for x in color_vals]
            x = col
            ys = list(dt[col].loc[curr_start:curr_end].index)
            ax[0].scatter([x]*length, ys, c=color_vals, marker='o',lw=0,s=marker_size, alpha=0.75)
            ax[0].scatter

        x_label = x_label + ' (count: ' + str(dt.shape[1])+ ')'
        ax[0].set_xlabel(x_label, fontsize=fontsize)
        ax[0].set_ylabel(y_label, fontsize=fontsize)
        ax[0].set_xlim([x_min_lim, x_max_lim])
        ax[0].set_ylim([pd.to_datetime(y_min_date), pd.to_datetime(y_max_date)]) 
        ax[0].plot([], [], ' ', label="Time series with at least {} days".format(min_days))
        ax[0].set_facecolor((0, 0, 0,0.1))
            
        # COLORBAR
        label_cb = "Concentration ({})".format(self.get_unit(analyte_name))
        if(log_transform):
            label_cb = "Log " + label_cb
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=ax[1], orientation='vertical')
        if(y_2nd_label!=None):
            label_cb = y_2nd_label
        cbar.set_label(label=label_cb,size=fontsize)

        ax[0].tick_params(axis='y', labelsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        plt.tight_layout()
        if(reverse_y_axis):
            ax[0].invert_yaxis()
        if(analyte_name!=None):
            title = title + ' (' + analyte_name + ')'
        fig.suptitle(title, fontsize=fontsize, y=1.05)
        if(return_data):
            return dt

    
    # Helper function to return start and end date for a date and a lag (+/- days)
    def __getLagDate(self, date, lagDays=7):
        date = pd.to_datetime(date)
        dateStart = date - pd.DateOffset(days=lagDays)
        dateEnd = date + pd.DateOffset(days=lagDays)
        return dateStart, dateEnd

    # Description: 
    #    Creates a table filling the data from the concentration dataset for a given analyte list where the columns are multi-indexed as follows [analytes, well names] and the index is all of the dates in the dataset. Many NaN should be expected.
    # Parameters:
    #    analytes (list of strings): list of analyte names to use
    def getCleanData(self, analytes):
        curr = self.data[['STATION_ID', 'COLLECTION_DATE', 'ANALYTE_NAME', 'RESULT']]
        main = pd.DataFrame()
        for ana in analytes:
            main = pd.concat([main, curr[curr.ANALYTE_NAME==ana]])
        piv = main.pivot_table(index=['COLLECTION_DATE'],columns=['ANALYTE_NAME', 'STATION_ID'], values='RESULT', aggfunc=np.mean)
        piv.index = pd.to_datetime(piv.index)
        piv.sort_index(inplace=True)
        return piv

    # Description: 
    #    Creates a table which counts the number of wells within a range specified by a list of lag days.
    # Parameters:
    #    analytes (list of strings): list of analyte names to use
    #    lag (list of ints): list of days to look ahead and behind the specified date (+/-)
    def getCommonDates(self, analytes, lag=[3,7,10]):
        piv = self.getCleanData(analytes)
        dates = piv.index
        names=['Dates', 'Lag']
        tuples = [dates, lag]
        finalData = pd.DataFrame(index=pd.MultiIndex.from_product(tuples, names=names), columns=['Date Ranges', 'Number of wells'])
        for date in dates:
            for i in lag:
                dateStart, dateEnd = self.__getLagDate(date, lagDays=i)
                mask = (piv.index > dateStart) & (piv.index <= dateEnd)
                result = piv[mask].dropna(axis=1, how='all')
                numWells = len(list(result.columns.get_level_values(1).unique()))
                dateRange = str(dateStart.date()) + " - " + str(dateEnd.date())
                finalData.loc[date, i]['Date Ranges'] = dateRange
                finalData.loc[date, i]['Number of wells'] = numWells
        return finalData
    
    # Description: 
    #    Creates a table filling the data from the concentration dataset for a given analyte list where the columns are multi-indexed as follows [analytes, well names] and the index is the date ranges secified by the lag.
    # Parameters:
    #    analytes (list of strings): list of analyte names to use
    #    lag (int): number of days to look ahead and behind the specified date (+/-)
    def getJointData(self, analytes, lag=3):
        if(self.jointData_is_set(lag=lag)==True):
            finalData = self.__jointData[0]
            return finalData
        piv = self.getCleanData(analytes)
        dates = piv.index
        dateRanges = []
        for date in dates:
            dateStart, dateEnd = self.__getLagDate(date, lagDays=lag)
            dateRange = str(dateStart.date()) + " - " + str(dateEnd.date())
            dateRanges.append(dateRange)
        finalData = pd.DataFrame(columns=piv.columns, index=dateRanges)
        numLoops = len(dates)
        everySomePercent = []
        print("Generating data with a lag of {}.".format(lag).upper())
        print("Progress:")
        for x in list(np.arange(1, 100, 1)):
            everySomePercent.append(round((x/100)*numLoops))
        for date, iteration in zip(dates, range(numLoops)):
            if(iteration in everySomePercent):
                print(str(round(iteration/numLoops*100)) + "%", end=', ')
            dateStart, dateEnd = self.__getLagDate(date, lagDays=lag)
            dateRange = str(dateStart.date()) + " - " + str(dateEnd.date())
            mask = (piv.index > dateStart) & (piv.index <= dateEnd)
            result = piv[mask].dropna(axis=1, how='all')
            resultCollapse = pd.concat([result[col].dropna().reset_index(drop=True) for col in result], axis=1)
            # HANDLE MULTIPLE VALUES
            if(resultCollapse.shape[0]>1):
                resultCollapse = pd.DataFrame(resultCollapse.mean()).T
            resultCollapse = resultCollapse.rename(index={0: dateRange})
            for ana_well in resultCollapse.columns:
                finalData.loc[dateRange, ana_well] =  resultCollapse.loc[dateRange, ana_well]
            # Save data to the pylenm global variable
            self.set_jointData(data=finalData, lag=lag)
        for col in finalData.columns:
            finalData[col] = finalData[col].astype('float64')
        print("Completed")
        return finalData
    
    # Error Metric: Mean Squared Error 
    def mse(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    # Description: 
    #    Returns the best Gaussian Process model for a given X and y.
    # Parameters:
    #    X (array): array of dimension (number of wells, 2) where each element is a pair of UTM coordinates.
    #    y (array of floats): array of size (number of wells) where each value corresponds to a concentration value at a well.
    #    smooth (bool): flag to toggle WhiteKernel on and off
    def get_Best_GP(self, X, y, smooth=True, seed = 42):
        gp = GaussianProcessRegressor(normalize_y=True, random_state=seed)
        # Kernel models
        if(smooth):
            k1 = Matern(length_scale=400, nu=1.5, length_scale_bounds=(100.0, 5000.0))+WhiteKernel()
            k2 = Matern(length_scale=800, nu=1.5, length_scale_bounds=(100.0, 5000.0))+WhiteKernel()
            k3 = Matern(length_scale=1200, nu=1.5, length_scale_bounds=(100.0, 5000.0))+WhiteKernel()
            k4 = Matern(length_scale=400, nu=1.5, length_scale_bounds=(100.0, 5000.0))+WhiteKernel()
            k5 = Matern(length_scale=800, nu=1.5, length_scale_bounds=(100.0, 5000.0))+WhiteKernel()
            k6 = Matern(length_scale=1200, nu=2.5, length_scale_bounds=(100.0, 5000.0))+WhiteKernel()
            k7 = Matern(length_scale=400, nu=2.5, length_scale_bounds=(100.0, 5000.0))+WhiteKernel()
            k8 = Matern(length_scale=800, nu=2.5, length_scale_bounds=(100.0, 5000.0))+WhiteKernel()
            k9 = Matern(length_scale=1200, nu=2.5, length_scale_bounds=(100.0, 5000.0))+WhiteKernel()
            k10 = Matern(length_scale=400, nu=np.inf, length_scale_bounds=(100.0, 5000.0))+WhiteKernel()
            k11 = Matern(length_scale=800, nu=np.inf, length_scale_bounds=(100.0, 5000.0))+WhiteKernel()
            k12 = Matern(length_scale=1200, nu=np.inf, length_scale_bounds=(100.0, 5000.0))+WhiteKernel()
        else:
            k1 = Matern(length_scale=400, nu=1.5, length_scale_bounds=(100.0, 5000.0))
            k2 = Matern(length_scale=800, nu=1.5, length_scale_bounds=(100.0, 5000.0))
            k3 = Matern(length_scale=1200, nu=1.5, length_scale_bounds=(100.0, 5000.0))
            k4 = Matern(length_scale=400, nu=1.5, length_scale_bounds=(100.0, 5000.0))
            k5 = Matern(length_scale=800, nu=1.5, length_scale_bounds=(100.0, 5000.0))
            k6 = Matern(length_scale=1200, nu=2.5, length_scale_bounds=(100.0, 5000.0))
            k7 = Matern(length_scale=400, nu=2.5, length_scale_bounds=(100.0, 5000.0))
            k8 = Matern(length_scale=800, nu=2.5, length_scale_bounds=(100.0, 5000.0))
            k9 = Matern(length_scale=1200, nu=2.5, length_scale_bounds=(100.0, 5000.0))
            k10 = Matern(length_scale=400, nu=np.inf, length_scale_bounds=(100.0, 5000.0))
            k11 = Matern(length_scale=800, nu=np.inf, length_scale_bounds=(100.0, 5000.0))
            k12 = Matern(length_scale=1200, nu=np.inf, length_scale_bounds=(100.0, 5000.0))
        parameters = {'kernel': [k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12]}
        model = GridSearchCV(gp, parameters)
        model.fit(X, y)
        return model

    # Description: 
    #    Fits Gaussian Process for X and y and returns both the GP model and the predicted values
    # Parameters:
    #    X (array): array of dimension (number of wells, 2) where each element is a pair of UTM coordinates.
    #    y (array of floats): array of size (number of wells) where each value corresponds to a concentration value at a well.
    #    xx (array floats): prediction locations
    #    model (GP model): model to fit
    #    smooth (bool): flag to toggle WhiteKernel on and off
    def fit_gp(self, X, y, xx, model=None, smooth=True):
        if(model==None):
            gp = self.get_Best_GP(X, y, smooth) # selects best kernel params to fit
        else:
            gp = model
        gp.fit(X, y)
        y_pred = gp.predict(xx)
        return gp, y_pred

    # Description: 
    #    Interpolate the water table as a function of topographic metrics using Gaussian Process. Uses regression to generate trendline adds the values to the GP map.
    # Parameters:
    #    X (dataframe): training values. Must include "Easting" and "Northing" columns.
    #    y (array of floats): array of size (number of wells) where each value corresponds to a concentration value at a well.
    #    xx (array floats): prediction locations
    #    ft (list of stings): feature names to train on
    #    regression (string): choice between 'linear' for linear regression, 'rf' for random forest regression, 'ridge' for ridge regression, or 'lasso' for lasso regression.
    #    model (GP model): model to fit
    #    smooth (bool): flag to toggle WhiteKernel on and off
    def interpolate_topo(self, X, y, xx, ft=['Elevation'], model=None, smooth=True, regression='linear', seed = 42):
        alpha_Values = [1e-5, 5e-5, 0.0001, 0.0005, 0.005, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 80]
        if(regression.lower()=='linear'):
            reg = LinearRegression()
        if(regression.lower()=='rf'):
            reg = RandomForestRegressor(n_estimators=200, random_state=seed)
        if(regression.lower()=='ridge'):
            # reg = make_pipeline(PolynomialFeatures(3), Ridge())
            reg = RidgeCV(alphas=alpha_Values)
        if(regression.lower()=='lasso'):
            # reg = make_pipeline(PolynomialFeatures(3), Lasso())
            reg = LassoCV(alphas=alpha_Values)
        if(all(elem in list(xx.columns) for elem in ft)):
            reg.fit(X[ft], y)
            y_est = reg.predict(X[ft])
            residuals = y - y_est
            if(model==None):
                model = self.get_Best_GP(X[['Easting','Northing']], residuals, smooth=smooth, seed=seed)
            else:
                model = model
            reg_trend = reg.predict(xx[ft])
        else:
            reg.fit(X[['Easting','Northing']], y)
            y_est = reg.predict(X[['Easting','Northing']])
            residuals = y - y_est
            if(model==None):
                model = self.get_Best_GP(X[['Easting','Northing']], residuals, smooth=smooth, seed=seed)
            else:
                model = model
            reg_trend = reg.predict(xx)
        r_map = model.predict(xx[['Easting','Northing']])
        y_map = reg_trend + r_map
        return y_map, r_map, residuals, reg_trend

    # Helper fucntion for get_Best_Wells
    def __get_Best_Well(self, X, y, xx, ref, selected, leftover, ft=['Elevation'], regression='linear', verbose=True, smooth=True, model=None):
        num_selected=len(selected)
        errors = []
        if(model==None):
            if(len(selected)<5):
                model, pred = self.fit_gp(X, y, xx)
            else:
                model = None
        else:
            model=model
        if(verbose):  
            print("# of wells to choose from: ", len(leftover))
        if(num_selected==0):
            if(verbose): 
                print("Selecting first well")
            for ix in leftover:
                y_pred, r_map, residuals, lr_trend = self.interpolate_topo(X=X.iloc[ix:ix+1,:], y=y[ix:ix+1], xx=xx, ft=ft, regression=regression, model=model, smooth=smooth)
                y_err = self.mse(ref, y_pred)
                errors.append((ix, y_err))
        
        if(num_selected > 0):
            for ix in leftover:
                joined = selected + [ix]
                y_pred, r_map, residuals, lr_trend = self.interpolate_topo(X=X.iloc[joined,:], y=y[joined], xx=xx, ft=ft, regression=regression, model=model, smooth=smooth)
                y_err = self.mse(ref, y_pred)
                errors.append((ix, y_err))
            
        err_ix = [x[0] for x in errors]
        err_vals = [x[1] for x in errors]
        min_val = min(err_vals)
        min_ix = err_ix[err_vals.index(min(err_vals))]
        if(verbose):
            print("Selected well: {} with a MSE error of {}\n".format(min_ix, min_val))
        return min_ix, min_val

    # Description: 
    #    Optimization function to select a subset of wells as to minimizes the MSE from a reference map
    # Parameters:
    #    X (array): array of dimension (number of wells, 2) where each element is a pair of UTM coordinates.
    #    y (array of floats): array of size (number of wells) where each value corresponds to a concentration value at a well.
    #    xx (array floats): prediction locations
    #    ref (array): reference values for xx locations
    #    max_wells (int):{} number of wells to optimize for
    #    ft (list of stings): feature names to train on
    #    regression (string): choice between 'linear' for linear regression, 'rf' for random forest regression, 'ridge' for ridge regression, or 'lasso' for lasso regression.
    #    initial (list of ints): indices of wells as the starting wells for optimization
    #    verbose (bool): flag to toggle details of the well selection process
    #    model (GP model): model to fit
    #    smooth (bool): flag to toggle WhiteKernel on and off
    def get_Best_Wells(self, X, y, xx, ref, initial, max_wells, ft=['Elevation'], regression='linear', verbose=True, smooth=True, model=None):
        tot_err = []
        selected = initial
        leftover = list(range(0, X.shape[0])) # all indexes from 0 to number of well
        
        # Remove the initial set of wells from pool of well indices to choose from
        for i in initial:
            leftover.remove(i)

        for i in range(max_wells-len(selected)):
            if(i==0): # select first well will min error
                well_ix, err = self.__get_Best_Well(X=X,y=y, xx=xx, ref=ref, selected=selected, leftover=leftover, ft=ft, regression=regression, verbose=verbose, smooth=smooth, model=model)
                selected.append(well_ix)
                leftover.remove(well_ix)
                tot_err.append(err)
            else:
                well_ix, err = self.__get_Best_Well(X=X,y=y, xx=xx, ref=ref, selected=selected, leftover=leftover, ft=ft, regression=regression, verbose=verbose, smooth=smooth, model=model)
                selected.append(well_ix)
                leftover.remove(well_ix)
                tot_err.append(err)
        print(selected)
        return selected, tot_err

    
    def dist(self, p1, p2):
        return sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))

    def add_dist_to_basin(self, XX, basin_coordinate=[436642.70,3681927.09], col_name='dist_to_basin'):
        x1,y1 = basin_coordinate
        distances = []
        for i in range(XX.shape[0]):
            x2,y2 = XX.iloc[i][0], XX.iloc[i][1]
            distances.append(self.dist([x1,y1],[x2,y2]))
        XX[col_name] = distances
        return XX