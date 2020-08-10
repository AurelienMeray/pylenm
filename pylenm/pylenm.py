# Required imports
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
import scipy
import random
import datetime
import re
import matplotlib.dates as mdates
from matplotlib.dates import date2num, num2date
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn import preprocessing
pd.set_option('display.max_columns', None) # to view all columns
from scipy.optimize import curve_fit
from supersmoother import SuperSmoother
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")
from ipyleaflet import Map 
from ipyleaflet import basemaps
from ipyleaflet import (Map, basemaps, WidgetControl, GeoJSON, 
                        LayersControl, Icon, Marker,FullScreenControl,
                        CircleMarker, Popup, AwesomeIcon) 
from ipywidgets import HTML


class functions:
    
    def __init__(self, data):
        self.setData(data)
    
    def __isValid_Data(self, data):
        if(str(type(data)).lower().find('dataframe') == -1):
            return (False, 'Make sure the data is a pandas DataFrame.\n')
        if(not self.__hasColumns_Data(data)):
            return (False, 'Make sure that ALL of the columns specified in the REQUIREMENTS are present.\n')
        else:
            return (True, None)
        
    def __isValid_GPS(self, data):
        if(str(type(data)).lower().find('dataframe') == -1):
            return (False, 'Make sure the data is a pandas DataFrame.\n')
        if(not self.__hasColumns_GPS(data)):
            return (False, 'Make sure that ALL of the columns specified in the REQUIREMENTS are present.\n')
        else:
            return (True, None)
    
    def __hasColumns_Data(self, data):
        find = ['COLLECTION_DATE','STATION_ID','ANALYTE_NAME','RESULT','RESULT_UNITS']
        cols = list(data.columns)
        hasCols =  all(item in cols for item in find)
        return hasCols
    
    def __hasColumns_GPS(self, data):
        find = ['station_id', 'latitude', 'longitude']
        cols = list(data.columns)
        hasCols =  all(item in cols for item in find)
        return hasCols
    
    def setData(self, data):
        validation = self.__isValid_Data(data)
        if(validation[0]):
            self.data = data
            print('Successfully stored the data!\n')

        else:
            print('ERROR: {}'.format(validation[1]))
            return self.REQUIREMENTS_DATA()
    
    def set_GPS_Data(self, GPS_Data):
        validation = self.__isValid_GPS(GPS_Data)
        if(validation[0]):
            self.GPS_Data = GPS_Data
            print('Successfully stored the GPS data!\n')

        else:
            print('ERROR: {}'.format(validation[1]))
            return self.REQUIREMENTS_GPS()
    
    def getData(self):
        return self.data
        
    def get_GPS_Data(self):
        return self.GPS_Data
        
    def REQUIREMENTS_DATA(self):
        print('PYLENM DATA REQUIREMENTS:\nThe imported data needs to meet ALL of the following conditions to have a successful import:')
        print('   1) Data should be a pandas dataframe.')
        print("   2) Data must have these column names (Case sensitive): \n      ['COLLECTION_DATE','STATION_ID','ANALYTE_NAME','RESULT','RESULT_UNITS']")
        
    def REQUIREMENTS_GPS(self):
        print('PYLENM GPS REQUIREMENTS:\nThe imported gps data needs to meet ALL of the following conditions to have a successful import:')
        print('   1) Data should be a pandas dataframe.')
        print("   2) Data must have these column names (Case sensitive): \n      ['station_id', 'latitude', 'longitude']")
    
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
            self.setData(data)
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
    
    # Description:
    #    Returns the unit of the analyte you specify.
    #    Example: 'DEPTH_TO_WATER' returns 'ft'
    # Parameters:
    #    analyte_name (string): name of the analyte to be processed
    def get_unit(self, analyte_name):
        unit_dictionary = {"1,1'-BIPHENYL": 'ug/L', '1,1,1,2-TETRACHLOROETHANE': 'ug/L', '1,1,1-TRICHLOROETHANE': 'ug/L', '1,1,2,2-TETRACHLOROETHANE': 'ug/L', '1,1,2-TRICHLORO-1,2,2-TRIFLUOROETHANE': 'ug/L', '1,1,2-TRICHLOROETHANE': 'ug/L', '1,1-DICHLOROETHANE': 'ug/L', '1,1-DICHLOROETHYLENE': 'ug/L', '1,1-DICHLOROPROPENE': 'ug/L', '1,2,3,4,6,7,8-HPCDD': 'ng/L', '1,2,3,4,6,7,8-HPCDF': 'ng/L', '1,2,3,4,7,8-HXCDD': 'ng/L', '1,2,3,4,7,8-HXCDF': 'ng/L', '1,2,3,7,8-PCDF': 'ng/L', '1,2,3-TRICHLOROBENZENE': 'ug/L', '1,2,3-TRICHLOROPROPANE': 'ug/L', '1,2,4,5-TETRACHLOROBENZENE': 'ug/L', '1,2,4-TRICHLOROBENZENE': 'ug/L', '1,2-DIBROMO-3-CHLOROPROPANE': 'ug/L', '1,2-DIBROMOETHANE': 'ug/L', '1,2-DICHLOROBENZENE': 'ug/L', '1,2-DICHLOROETHANE (EDC)': 'ug/L', '1,2-DICHLOROETHYLENE': 'ug/L', '1,2-DICHLOROPROPANE': 'ug/L', '1,2-DIPHENYLHYDRAZINE': 'ug/L', '1,3,5-TRIMETHYLBENZENE': 'ug/L', '1,3,5-TRINITROBENZENE': 'ug/L', '1,3-DICHLOROBENZENE': 'ug/L', '1,3-DICHLOROPROPANE': 'ug/L', '1,3-DINITROBENZENE': 'ug/L', '1,4-DICHLOROBENZENE': 'ug/L', '1,4-DIOXANE': 'ug/L', '1,4-NAPHTHOQUINONE': 'ug/L', '1-NAPHTHYLAMINE': 'ug/L', '2,2-DICHLOROPROPANE': 'ug/L', '2,3,4,6-TETRACHLOROPHENOL': 'ug/L', '2,3,7,8-TCDD': 'ng/L', '2,3,7,8-TCDF': 'ng/L', '2,4,5-T': 'ug/L', '2,4,5-TP (SILVEX)': 'ug/L', '2,4,5-TRICHLOROPHENOL': 'ug/L', '2,4,6-TRICHLOROPHENOL': 'ug/L', '2,4-DICHLOROPHENOL': 'ug/L', '2,4-DICHLOROPHENOXYACETIC ACID (2,4-D)': 'ug/L', '2,4-DIMETHYLPHENOL': 'ug/L', '2,4-DINITROPHENOL': 'ug/L', '2,4-DINITROTOLUENE': 'ug/L', '2,6-DICHLOROPHENOL': 'ug/L', '2,6-DINITROTOLUENE': 'ug/L', '2-ACETYLAMINOFLUORENE': 'ug/L', '2-CHLOROETHYL VINYL ETHER': 'ug/L', '2-CHLORONAPHTHALENE': 'ug/L', '2-CHLOROPHENOL': 'ug/L', '2-HEXANONE': 'ug/L', '2-METHYLANILINE (O-TOLUIDINE)': 'ug/L', '2-METHYLNAPHTHALENE': 'ug/L', '2-NAPHTHYLAMINE': 'ug/L', '2-NITROANILINE': 'ug/L', '2-NITROPHENOL': 'ug/L', '2-NITROPROPANE': 'ug/L', '2-PICOLINE': 'ug/L', "3,3'-DIMETHYLBENZIDINE": 'ug/L', '3,3-DICHLOROBENZIDINE': 'ug/L', '3-METHYLCHOLANTHRENE': 'ug/L', '4-AMINOBIPHENYL': 'ug/L', '4-BROMOPHENYL PHENYL ETHER': 'ug/L', '4-CHLOROANILINE': 'ug/L', '4-CHLOROPHENYL PHENYL ETHER': 'ug/L', '4-CHLOROTOLUENE': 'ug/L', '4-NITROPHENOL': 'ug/L', '4-NITROQUINOLINE-1-OXIDE': 'ug/L', '5-NITRO-O-TOLUIDINE': 'ug/L', '7,12-DIMETHYLBENZ(A)ANTHRACENE': 'ug/L', 'A,A-DIMETHYLPHENETHYLAMINE': 'ug/L', 'ACENAPHTHENE': 'ug/L', 'ACENAPHTHYLENE': 'ug/L', 'ACETONE': 'ug/L', 'ACETONITRILE (METHYL CYANIDE)': 'ug/L', 'ACETOPHENONE': 'ug/L', 'ACROLEIN': 'ug/L', 'ACRYLONITRILE': 'ug/L', 'ACTINIUM-228': 'pCi/L', 'AIR TEMPERATURE': 'degC', 'ALDRIN': 'ug/L', 'ALLYL CHLORIDE': 'ug/L', 'ALPHA-BENZENE HEXACHLORIDE': 'ug/L', 'ALPHA-CHLORDANE': 'ug/L', 'ALUMINUM': 'ug/L', 'AMERICIUM-241': 'pCi/L', 'AMERICIUM-241/CURIUM-246': 'pCi/L', 'AMERICIUM-243': 'pCi/L', 'AMMONIA': 'mg/L', 'ANILINE': 'ug/L', 'ANTHRACENE': 'ug/L', 'ANTIMONY': 'ug/L', 'ANTIMONY-124': 'pCi/L', 'ANTIMONY-125': 'pCi/L', 'ARAMITE': 'ug/L', 'AROCLOR 1016': 'ug/L', 'AROCLOR 1221': 'ug/L', 'AROCLOR 1232': 'ug/L', 'AROCLOR 1242': 'ug/L', 'AROCLOR 1248': 'ug/L', 'AROCLOR 1254': 'ug/L', 'AROCLOR 1260': 'ug/L', 'ARSENIC': 'ug/L', 'ATRAZINE': 'ug/L', 'BARIUM': 'ug/L', 'BARIUM-133': 'pCi/L', 'BARIUM-140': 'pCi/L', 'BENZALDEHYDE': 'ug/L', 'BENZENE': 'ug/L', 'BENZIDINE': 'ug/L', 'BENZO(G,H,I)PERYLENE': 'ug/L', 'BENZOIC ACID': 'ug/L', 'BENZO[A]ANTHRACENE': 'ug/L', 'BENZO[A]PYRENE': 'ug/L', 'BENZO[B]FLUORANTHENE': 'ug/L', 'BENZO[K]FLUORANTHENE': 'ug/L', 'BENZYL ALCOHOL': 'ug/L', 'BERYLLIUM': 'ug/L', 'BERYLLIUM-7': 'pCi/L', 'BETA-BENZENE HEXACHLORIDE': 'ug/L', 'BIS(2-CHLORO-1-METHYLETHYL)ETHER': 'ug/L', 'BIS(2-CHLOROETHOXY)METHANE': 'ug/L', 'BIS(2-CHLOROETHYL)ETHER': 'ug/L', 'BIS(2-ETHYLHEXYL)PHTHALATE (DEHP)': 'ug/L', 'BIS(CHLOROMETHYL)ETHER': 'ug/L', 'BISMUTH-212': 'pCi/L', 'BISMUTH-214': 'pCi/L', 'BORON': 'ug/L', 'BROMIDE': 'mg/L', 'BROMOBENZENE': 'ug/L', 'BROMOCHLOROMETHANE': 'ug/L', 'BROMODICHLOROMETHANE': 'ug/L', 'BROMOFORM (TRIBROMOMETHANE)': 'ug/L', 'BROMOMETHANE (METHYL BROMIDE)': 'ug/L', 'BUTYL BENZYL PHTHALATE': 'ug/L', 'CADMIUM': 'ug/L', 'CALCIUM': 'ug/L', 'CAPROLACTAM': 'ug/L', 'CARBAZOLE': 'ug/L', 'CARBON 13-LABELED 2,3,7,8-TCDD': 'ng/L', 'CARBON 13-LABELED 2,3,7,8-TCDF': 'ng/L', 'CARBON DISULFIDE': 'ug/L', 'CARBON TETRACHLORIDE': 'ug/L', 'CARBON-14': 'pCi/L', 'CARBONATE': 'mg/L', 'CERIUM-141': 'pCi/L', 'CERIUM-144': 'pCi/L', 'CESIUM': 'ug/L', 'CESIUM-134': 'pCi/L', 'CESIUM-137': 'pCi/L', 'CHLORIDE': 'mg/L', 'CHLOROBENZENE': 'ug/L', 'CHLOROBENZILATE': 'ug/L', 'CHLOROETHANE (ETHYL CHLORIDE)': 'ug/L', 'CHLOROETHENE (VINYL CHLORIDE)': 'ug/L', 'CHLOROFORM': 'ug/L', 'CHLOROMETHANE (METHYL CHLORIDE)': 'ug/L', 'CHLOROPRENE': 'ug/L', 'CHROMIUM': 'ug/L', 'CHROMIUM-51': 'pCi/L', 'CHRYSENE': 'ug/L', 'CIS-1,2-DICHLOROETHYLENE': 'ug/L', 'CIS-1,3-DICHLOROPROPENE': 'ug/L', 'COBALT': 'ug/L', 'COBALT-57': 'pCi/L', 'COBALT-58': 'pCi/L', 'COBALT-60': 'pCi/L', 'COPPER': 'ug/L', 'CUMENE (ISOPROPYLBENZENE)': 'ug/L', 'CURIUM-242': 'pCi/L', 'CURIUM-243': 'pCi/L', 'CURIUM-243/244': 'pCi/L', 'CURIUM-244': 'pCi/L', 'CURIUM-245/246': 'pCi/L', 'CURIUM-246': 'pCi/L', 'CYANIDE': 'ug/L', 'CYCLOHEXANE': 'ug/L', 'CYCLOHEXANONE': 'ug/L', 'DDD': 'ug/L', 'DDE': 'ug/L', 'DDT': 'ug/L', 'DELTA-BENZENE HEXACHLORIDE': 'ug/L', 'DEPTH_TO_WATER': 'ft', 'DI-N-BUTYL PHTHALATE': 'ug/L', 'DIALLATE': 'ug/L', 'DIBENZOFURAN': 'ug/L', 'DIBENZ[AH]ANTHRACENE': 'ug/L', 'DIBROMOCHLOROMETHANE': 'ug/L', 'DIBROMOMETHANE (METHYLENE BROMIDE)': 'ug/L', 'DICHLORODIFLUOROMETHANE': 'ug/L', 'DICHLOROMETHANE (METHYLENE CHLORIDE)': 'ug/L', 'DIELDRIN': 'ug/L', 'DIETHYL PHTHALATE': 'ug/L', 'DIMETHOATE': 'ug/L', 'DIMETHYL PHTHALATE': 'ug/L', 'DINITRO-O-CRESOL': 'ug/L', 'DINOSEB': 'ug/L', 'DIPHENYLAMINE': 'ug/L', 'DISULFOTON': 'ug/L', 'ENDOSULFAN I': 'ug/L', 'ENDOSULFAN II': 'ug/L', 'ENDOSULFAN SULFATE': 'ug/L', 'ENDRIN': 'ug/L', 'ENDRIN ALDEHYDE': 'ug/L', 'ENDRIN KETONE': 'ug/L', 'ETHANE': 'ug/L', 'ETHYL ACETATE': 'ug/L', 'ETHYL METHACRYLATE': 'ug/L', 'ETHYL METHANESULFONATE': 'ug/L', 'ETHYLBENZENE': 'ug/L', 'ETHYLENE': 'ug/L', 'EUROPIUM-152': 'pCi/L', 'EUROPIUM-154': 'pCi/L', 'EUROPIUM-155': 'pCi/L', 'FAMPHUR': 'ug/L', 'FLOW RATE': 'gpm', 'FLUORANTHENE': 'ug/L', 'FLUORENE': 'ug/L', 'FLUORIDE': 'mg/L', 'GAMMA-CHLORDANE': 'ug/L', 'GROSS ALPHA': 'pCi/L', 'HARDNESS AS CACO3': 'ug/L', 'HEPTACHLOR': 'ug/L', 'HEPTACHLOR EPOXIDE': 'ug/L', 'HEPTACHLORODIBENZO-P-DIOXINS': 'ng/L', 'HEPTACHLORODIBENZO-P-FURANS': 'ng/L', 'HEPTACHLORODIBENZOFURAN': 'ng/L', 'HEXACHLOROBENZENE': 'ug/L', 'HEXACHLOROBUTADIENE': 'ug/L', 'HEXACHLOROCYCLOPENTADIENE': 'ug/L', 'HEXACHLORODIBENZO-P-DIOXINS': 'ng/L', 'HEXACHLORODIBENZO-P-FURANS': 'ng/L', 'HEXACHLOROETHANE': 'ug/L', 'HEXACHLOROPHENE': 'ug/L', 'HEXACHLOROPROPENE': 'ug/L', 'HEXACHLORORDIBENZOFURAN': 'ng/L', 'HEXANE': 'ug/L', 'INDENO[1,2,3-CD]PYRENE': 'ug/L', 'IODINE-129': 'pCi/L', 'IODINE-131': 'pCi/L', 'IODOMETHANE (METHYL IODIDE)': 'ug/L', 'IRON': 'ug/L', 'IRON-55': 'pCi/L', 'IRON-59': 'pCi/L', 'ISOBUTANOL': 'ug/L', 'ISODRIN': 'ug/L', 'ISOPHORONE': 'ug/L', 'ISOSAFROLE': 'ug/L', 'KEPONE': 'ug/L', 'LEAD': 'ug/L', 'LEAD-212': 'pCi/L', 'LEAD-214': 'pCi/L', 'LINDANE': 'ug/L', 'LITHIUM': 'ug/L', 'M,P-XYLENE': 'ug/L', 'M-CRESOL': 'ug/L', 'M-NITROANILINE': 'ug/L', 'M/P-CRESOL': 'ug/L', 'MAGNESIUM': 'ug/L', 'MANGANESE': 'ug/L', 'MANGANESE-54': 'pCi/L', 'MERCURY': 'ug/L', 'METHACRYLONITRILE': 'ug/L', 'METHANE': 'ug/L', 'METHAPYRILENE': 'ug/L', 'METHOXYCHLOR': 'ug/L', 'METHYL ACETATE': 'ug/L', 'METHYL ETHYL KETONE': 'ug/L', 'METHYL ISOBUTYL KETONE': 'ug/L', 'METHYL METHACRYLATE': 'ug/L', 'METHYL METHANESULFONATE': 'ug/L', 'METHYL PARATHION': 'ug/L', 'METHYL TERTIARY BUTYL ETHER (MTBE)': 'ug/L', 'METHYLCYCLOHEXANE': 'ug/L', 'MOLYBDENUM': 'ug/L', 'N-BUTYLBENZENE': 'ug/L', 'N-DIOCTYL PHTHALATE': 'ug/L', 'N-NITROSO-N-METHYLETHYLAMINE': 'ug/L', 'N-NITROSODI-N-BUTYLAMINE': 'ug/L', 'N-NITROSODIETHYLAMINE': 'ug/L', 'N-NITROSODIMETHYLAMINE': 'ug/L', 'N-NITROSODIPHENYLAMINE': 'ug/L', 'N-NITROSODIPHENYLAMINE+DIPHENYLAMINE': 'ug/L', 'N-NITROSODIPROPYLAMINE': 'ug/L', 'N-NITROSOMORPHOLINE': 'ug/L', 'N-NITROSOPIPERIDINE': 'ug/L', 'N-NITROSOPYRROLIDINE': 'ug/L', 'N-PROPYLBENZENE': 'ug/L', 'NAPHTHALENE': 'ug/L', 'NEPTUNIUM-237': 'pCi/L', 'NEPTUNIUM-239': 'pCi/L', 'NICKEL': 'ug/L', 'NICKEL-59': 'pCi/L', 'NICKEL-63': 'pCi/L', 'NIOBIUM-95': 'pCi/L', 'NITRATE': 'mg/L', 'NITRATE-NITRITE AS NITROGEN': 'mg/L', 'NITRITES': 'mg/L', 'NITROBENZENE': 'ug/L', 'NONVOLATILE BETA': 'pCi/L', 'O,O,O-TRIETHYL PHOSPHOROTHIOATE': 'ug/L', 'O-CRESOL (2-METHYLPHENOL)': 'ug/L', 'O-XYLENE': 'ug/L', 'OCTACHLORODIBENZO-P-DIOXIN': 'ng/L', 'OCTACHLORODIBENZO-P-FURAN': 'ng/L', 'ORTHOCHLOROTOLUENE': 'ug/L', 'ORTHOPHOSPHATE': 'mg/L', 'OXALATE': 'mg/L', 'OXIDATION/REDUCTION POTENTIAL': 'mV', 'OXYGEN': 'mg/L', 'P-CHLORO-M-CRESOL': 'ug/L', 'P-CRESOL': 'ug/L', 'P-DIMETHYLAMINOAZOBENZENE': 'ug/L', 'P-NITROANILINE': 'ug/L', 'P-PHENYLENEDIAMINE': 'ug/L', 'PARACYMEN': 'ug/L', 'PARATHION': 'ug/L', 'PENTACHLOROBENZENE': 'ug/L', 'PENTACHLORODIBENZO-P-DIOXINS': 'ng/L', 'PENTACHLORODIBENZO-P-FURANS': 'ng/L', 'PENTACHLORODIBENZOFURAN': 'ng/L', 'PENTACHLOROETHANE': 'ug/L', 'PENTACHLORONITROBENZENE': 'ug/L', 'PENTACHLOROPHENOL': 'ug/L', 'PH': 'pH', 'PHENACETIN': 'ug/L', 'PHENANTHRENE': 'ug/L', 'PHENOL': 'ug/L', 'PHENOLPHTHALEIN ALKALINITY (AS CACO3)': 'mg/L', 'PHENOLS': 'mg/L', 'PHORATE': 'ug/L', 'PLUTONIUM-238': 'pCi/L', 'PLUTONIUM-239': 'pCi/L', 'PLUTONIUM-239/240': 'pCi/L', 'PLUTONIUM-242': 'pCi/L', 'POTASSIUM': 'ug/L', 'POTASSIUM-40': 'pCi/L', 'PROMETHIUM-144': 'pCi/L', 'PROMETHIUM-146': 'pCi/L', 'PRONAMIDE': 'ug/L', 'PROPIONITRILE': 'ug/L', 'PYRENE': 'ug/L', 'PYRIDINE': 'ug/L', 'RADIUM, TOTAL ALPHA-EMITTING': 'pCi/L', 'RADIUM-226': 'pCi/L', 'RADIUM-228': 'pCi/L', 'RADON-222': 'pCi/L', 'RUTHENIUM-103': 'pCi/L', 'RUTHENIUM-106': 'pCi/L', 'SAFROLE': 'ug/L', 'SEC-BUTYLBENZENE': 'ug/L', 'SELENIUM': 'ug/L', 'SILICA': 'ug/L', 'SILICON': 'ug/L', 'SILVER': 'ug/L', 'SODIUM': 'ug/L', 'SODIUM-22': 'pCi/L', 'SPECIFIC CONDUCTANCE': 'uS/cm', 'STRONTIUM': 'ug/L', 'STRONTIUM-89': 'pCi/L', 'STRONTIUM-89/90': 'pCi/L', 'STRONTIUM-90': 'pCi/L', 'STYRENE': 'ug/L', 'SULFATE': 'mg/L', 'SULFIDE': 'mg/L', 'SULFOTEPP': 'ug/L', 'SULFUR': 'ug/L', 'TECHNETIUM-99': 'pCi/L', 'TEMPERATURE': 'degC', 'TERT-BUTYLBENZENE': 'ug/L', 'TETRACHLORODIBENZO-P-DIOXIN': 'ng/L', 'TETRACHLORODIBENZO-P-FURANS': 'ng/L', 'TETRACHLORODIBENZOFURAN': 'ng/L', 'TETRACHLOROETHYLENE (PCE)': 'ug/L', 'THALLIUM': 'ug/L', 'THALLIUM-208': 'pCi/L', 'THIONAZIN': 'ug/L', 'THORIUM': 'ug/L', 'THORIUM-228': 'pCi/L', 'THORIUM-230': 'pCi/L', 'THORIUM-232': 'pCi/L', 'THORIUM-234': 'pCi/L', 'TIN': 'ug/L', 'TIN-113': 'pCi/L', 'TITANIUM': 'ug/L', 'TOLUENE': 'ug/L', 'TOTAL ACTIVITY': 'pCi/mL', 'TOTAL ALKALINITY (AS CACO3)': 'mg/L', 'TOTAL CHLORDANE': 'ug/L', 'TOTAL DISSOLVED SOLIDS': 'mg/L', 'TOTAL ORGANIC CARBON': 'mg/L', 'TOTAL ORGANIC HALOGENS': 'mg/L', 'TOTAL PHOSPHATES (AS  P)': 'ug/L', 'TOTAL SUSPENDED SOLIDS': 'mg/L', 'TOXAPHENE': 'ug/L', 'TRANS-1,2-DICHLOROETHYLENE': 'ug/L', 'TRANS-1,3-DICHLOROPROPENE': 'ug/L', 'TRANS-1,4-DICHLORO-2-BUTENE': 'ug/L', 'TRIBUTYL PHOSPHATE': 'ug/L', 'TRICHLOROETHYLENE (TCE)': 'ug/L', 'TRICHLOROFLUOROMETHANE': 'ug/L', 'TRICHLOROTRIFLUOROETHANE': 'ug/L', 'TRITIUM': 'pCi/mL', 'TURBIDITY': 'NTU', 'URANIUM': 'ug/L', 'URANIUM-233/234': 'pCi/L', 'URANIUM-234': 'pCi/L', 'URANIUM-235': 'pCi/L', 'URANIUM-235/236': 'pCi/L', 'URANIUM-238': 'pCi/L', 'VANADIUM': 'ug/L', 'VINYL ACETATE': 'ug/L', 'VOLUME PURGED': 'gal', 'WATER TEMPERATURE': 'degC', 'XYLENES': 'ug/L', 'YTTRIUM-88': 'pCi/L', 'ZINC': 'ug/L', 'ZINC-65': 'pCi/L', 'ZIRCONIUM-95': 'pCi/L'}
        return unit_dictionary[analyte_name]
    
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
            wells = pd.DataFrame(wells, columns=['station_id'])
            for index, row in wells.iterrows():
                mo = re.match('.+([0-9])[^0-9]*$', row.station_id)
                last_index = mo.start(1)
                wells.at[index, 'unit'] = row.station_id[last_index+1:]
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
        return list(res.station_id)

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
    def get_analyte_details(self, analyte_name, save_dir='analyte_details'):
        data = self.data
        data = data[data.ANALYTE_NAME == analyte_name].reset_index().drop('index', axis=1)
        data = data[~data.RESULT.isna()]
        data = data.drop(['ANALYTE_NAME', 'RESULT', 'RESULT_UNITS'], axis=1)
        data.COLLECTION_DATE = pd.to_datetime(data.COLLECTION_DATE)

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
            details['Cumulative samples'] = details['Unique samples'].cumsum()
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
    def get_data_summary(self, analytes=None, sort_by='date', ascending=False):
        data = self.data
        if(analytes == None):
            analytes = data.ANALYTE_NAME.unique()
        data = data.loc[data.ANALYTE_NAME.isin(analytes)].drop(['RESULT_UNITS'], axis=1)
        data = data[~data.duplicated()] # remove duplicates
        data.COLLECTION_DATE = pd.to_datetime(data.COLLECTION_DATE)
        data = data[~data.RESULT.isna()]

        info = []
        for analyte_name in analytes:
            query = data[data.ANALYTE_NAME == analyte_name]
            startDate = min(query.COLLECTION_DATE)
            endDate = max(query.COLLECTION_DATE)
            numSamples = query.shape[0]
            wellCount = len(query.STATION_ID.unique())
            stats = query.RESULT.describe().drop('count', axis=0)
            stats = pd.DataFrame(stats).T
            stats_col = ['Result '+ x for x in stats.columns]

            result = {'Analyte Name': analyte_name, 'Start Date': startDate, 'End Date': endDate,
                      'Date Range (days)':endDate-startDate, 'Unique wells': wellCount,'Samples': numSamples,
                      'Result unit': self.get_unit(analyte_name) }
            for num in range(len(stats_col)):
                result[stats_col[num]] = stats.iloc[0][num] 

            info.append(result)

            details = pd.DataFrame(info)
            details.index = details['Analyte Name']
            details = details.drop('Analyte Name', axis=1)
            if(sort_by.lower() == 'date'):
                details = details.sort_values(by=['Start Date', 'End Date', 'Date Range (days)'], ascending=ascending)
            elif(sort_by.lower() == 'samples'):
                details = details.sort_values(by=['Samples'], ascending=ascending)
            elif(sort_by.lower() == 'wells'):
                details = details.sort_values(by=['Unique wells'], ascending=ascending)

        return details
    
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
              plot_inline=True, year_interval=2, save_dir='plot_data'):
    
        # Gets appropriate data (well_name and analyte_name)
        query = self.query_data(well_name, analyte_name)
        query = self.simplify_data(data=query)

        if(type(query)==int and query == 0):
            return 'No results found for {} and {}'.format(well_name, analyte_name)
        else:   
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
                    if(log_transform):
                        ax.set_ylabel('log-Concentration (' + unit + ')')
                    else:
                        ax.set_ylabel('Concentration (' + unit + ')')
                    ax.set_xlabel('Years')
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
            ax.text(1.3, 1.05, 'Start date:  {}\nEnd date:    {}\n\nSamples:     {} of {}'.format(piv.index[0], piv.index[-1], samples, totalSamples), 
                    transform=ax.transAxes, fontsize=15, fontweight='bold', verticalalignment='bottom', bbox=props)
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
    #    remove_outliers (bool): choose whether or to remove the outliers.
    #    z_threshold (float): z_score threshold to eliminate outliers
    #    interpolate (bool): choose whether or to interpolate the data
    #    frequency (string): {‘D’, ‘W’, ‘M’, ‘Y’} frequency to interpolate. See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html for valid frequency inputs. (e.g. ‘W’ = every week, ‘D ’= every day, ‘2W’ = every 2 weeks)
    #    save_dir (string): name of the directory you want to save the plot to
    def plot_corr_by_well(self, well_name, remove_outliers=True, z_threshold=4, interpolate=False, frequency='2W', save_dir='plot_correlation'):
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
            scaler = StandardScaler()
            pivScaled = scaler.fit_transform(piv)
            pivScaled = pd.DataFrame(pivScaled, columns=piv.columns)
            pivScaled.index = piv.index
            piv = pivScaled
            # Remove outliers
            if(remove_outliers):
                piv = self.remove_outliers(piv, z_threshold=z_threshold)
            samples = piv.shape[0]
            

            sns.set_style("white", {"axes.facecolor": "0.95"})
            g = sns.PairGrid(piv, aspect=1.2, diag_sharey=False, despine=False)
            g.fig.suptitle(title, fontweight='bold', y=1.08, fontsize=25)
            g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'red', 'lw': 3},
                                                            scatter_kws={'color': 'black', 's': 20})
            g.map_diag(sns.distplot, kde_kws={'color': 'black', 'lw': 3},
                                     hist_kws={'histtype': 'bar', 'lw': 2, 'edgecolor': 'k', 'facecolor':'grey'})
            g.map_upper(self.__plotUpperHalf)
            for ax in g.axes.flat: 
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            g.fig.subplots_adjust(wspace=0.3, hspace=0.3)
            ax = plt.gca()

            props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
            ax.text(1.3, 6.2, 'Start date:  {}\nEnd date:    {}\n\nOriginal samples:     {}\nSamples used:     {}'
                        .format(piv.index[0].date(), piv.index[-1].date(), totalSamples, samples), 
                        transform=ax.transAxes, fontsize=20, fontweight='bold', verticalalignment='bottom', bbox=props)
            # Add titles to the diagonal axes/subplots
            for ax, col in zip(np.diag(g.axes), piv.columns):
                ax.set_title(col, y=0.82, fontsize=15)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            g.fig.savefig(save_dir + '/' + well_name + file_extension + '.png', bbox_inches="tight")
    
    # Description: 
    #    Plots the correlations with the physical plots as well as the important analytes over time for each well in the dataset.
    # Parameters:
    #    remove_outliers (bool): choose whether or to remove the outliers.
    #    z_threshold (float): z_score threshold to eliminate outliers
    #    interpolate (bool): choose whether or to interpolate the data
    #    frequency (string): {‘D’, ‘W’, ‘M’, ‘Y’} frequency to interpolate. See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html for valid frequency inputs. (e.g. ‘W’ = every week, ‘D ’= every day, ‘2W’ = every 2 weeks)
    #    save_dir (string): name of the directory you want to save the plot to
    def plot_all_corr_by_well(self, remove_outliers=True, z_threshold=4, interpolate=False, frequency='2W', save_dir='plot_correlation'):
        data = self.data
        wells = np.array(data.STATION_ID.values)
        wells = np.unique(wells)
        for well in wells:
            self.plot_corr_by_well(well_name=well, remove_outliers=remove_outliers, z_threshold=z_threshold, interpolate=interpolate, frequency=frequency, save_dir=save_dir)
        
    # Description: 
    #    Plots the correlations with the physical plots as well as the correlations of the important analytes for ALL the wells on a specified date.
    # Parameters:
    #    date (string): date to be analyzed
    #    min_samples (int): minimum number of samples the result should contain in order to execute.
    #    save_dir (string): name of the directory you want to save the plot to    
    def plot_corr_by_date(self, date, min_samples=48, save_dir='plot_corr_by_date'):
        data = self.data
        data = self.simplify_data(data=data)
        query = data[data.COLLECTION_DATE == date]
        a = list(np.unique(query.ANALYTE_NAME.values))
        b = ['TRITIUM','IODINE-129','SPECIFIC CONDUCTANCE', 'PH','URANIUM-238', 'DEPTH_TO_WATER']
        analytes = self.__custom_analyte_sort(list(set(a) and set(b)))
        query = query.loc[query.ANALYTE_NAME.isin(analytes)]

        if(query.shape[0] == 0):
            return 'ERROR: {} has no data for the 6 analytes.'.format(date)
        samples = query[['COLLECTION_DATE', 'STATION_ID', 'ANALYTE_NAME']].duplicated().value_counts()[0]
        if(samples < min_samples):
            return 'ERROR: {} does not have at least {} samples.'.format(date, min_samples)
        if(len(np.unique(query.ANALYTE_NAME.values)) < 6):
            return 'ERROR: {} has less than the 6 analytes we want to analyze.'.format(date)
        else:
            analytes = self.__custom_analyte_sort(np.unique(query.ANALYTE_NAME.values))
            piv = query.reset_index().pivot_table(index = 'STATION_ID', columns='ANALYTE_NAME', values='RESULT',aggfunc=np.mean)
            title = date + '_correlation'

            sns.set_style("white", {"axes.facecolor": "0.95"})
            g = sns.PairGrid(piv, aspect=1.2, diag_sharey=False, despine=False)
            g.fig.suptitle(title, fontweight='bold', y=1.08, fontsize=25)
            g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'red', 'lw': 3},
                                                            scatter_kws={'color': 'black', 's': 20})
            g.map_diag(sns.distplot, kde_kws={'color': 'black', 'lw': 3},
                                     hist_kws={'histtype': 'bar', 'lw': 2, 'edgecolor': 'k', 'facecolor':'grey'})
            g.map_upper(self.__plotUpperHalf)
            for ax in g.axes.flat: 
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            g.fig.subplots_adjust(wspace=0.3, hspace=0.3)
            ax = plt.gca()

            props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
            ax.text(1.3, 3, 'Date:  {}\n\nSamples used:     {}'.format(date, samples), 
                        transform=ax.transAxes, fontsize=20, fontweight='bold', verticalalignment='bottom', bbox=props)
            # Add titles to the diagonal axes/subplots
            for ax, col in zip(np.diag(g.axes), piv.columns):
                ax.set_title(col, y=0.82, fontsize=15)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            g.fig.savefig(save_dir + '/' + date + '.png', bbox_inches="tight")

    
    # Description: 
    #    Plots the correlations with the physical plots as well as the correlations of the important analytes for ALL the wells in specified year.
    # Parameters:
    #    year (int): year to be analyzed
    #    min_samples (int): minimum number of samples the result should contain in order to execute.
    #    save_dir (string): name of the directory you want to save the plot to
    def plot_corr_by_year(self, year, remove_outliers=True, z_threshold=4, min_samples=500, save_dir='plot_corr_by_year'):
        data = self.data
        query = data
        query = self.simplify_data(data=query)
        query.COLLECTION_DATE = pd.to_datetime(query.COLLECTION_DATE)
        query = query[query.COLLECTION_DATE.dt.year == year]
        a = list(np.unique(query.ANALYTE_NAME.values))
        b = ['TRITIUM','IODINE-129','SPECIFIC CONDUCTANCE', 'PH','URANIUM-238', 'DEPTH_TO_WATER']
        analytes = self.__custom_analyte_sort(list(set(a) and set(b)))
        query = query.loc[query.ANALYTE_NAME.isin(analytes)]
        if(query.shape[0] == 0):
            return 'ERROR: {} has no data for the 6 analytes.'.format(year)
        samples = query[['COLLECTION_DATE', 'STATION_ID', 'ANALYTE_NAME']].duplicated().value_counts()[0]
        if(samples < min_samples):
            return 'ERROR: {} does not have at least {} samples.'.format(date, min_samples)
        if(len(np.unique(query.ANALYTE_NAME.values)) < 6):
            return 'ERROR: {} has less than the 6 analytes we want to analyze.'.format(year)
        else:
            analytes = self.__custom_analyte_sort(np.unique(query.ANALYTE_NAME.values))
            piv = query.reset_index().pivot_table(index = 'STATION_ID', columns='ANALYTE_NAME', values='RESULT',aggfunc=np.mean)
            # Remove outliers
            if(remove_outliers):
                piv = self.remove_outliers(piv, z_threshold=z_threshold)
            samples = piv.shape[0] * piv.shape[1]

            title = str(year) + '_correlation'

            sns.set_style("white", {"axes.facecolor": "0.95"})
            g = sns.PairGrid(piv, aspect=1.2, diag_sharey=False, despine=False)
            g.fig.suptitle(title, fontweight='bold', y=1.08, fontsize=25)
            g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'red', 'lw': 3},
                                                            scatter_kws={'color': 'black', 's': 20})
            g.map_diag(sns.distplot, kde_kws={'color': 'black', 'lw': 3},
                                     hist_kws={'histtype': 'bar', 'lw': 2, 'edgecolor': 'k', 'facecolor':'grey'})
            g.map_upper(self.__plotUpperHalf)
            for ax in g.axes.flat: 
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            g.fig.subplots_adjust(wspace=0.3, hspace=0.3)
            ax = plt.gca()

            props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
            ax.text(1.3, 3, 'Date:  {}\n\nSamples used:     {}'.format(year, samples), 
                        transform=ax.transAxes, fontsize=20, fontweight='bold', verticalalignment='bottom', bbox=props)
            # Add titles to the diagonal axes/subplots
            for ax, col in zip(np.diag(g.axes), piv.columns):
                ax.set_title(col, y=0.82, fontsize=15)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            g.fig.savefig(save_dir + '/' + str(year) + '.png', bbox_inches="tight")
            
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
                            bbox=dict(boxstyle="round", alpha=0.1),ha='center',
                           arrowprops=dict(arrowstyle="->", color='blue'), fontsize=small_fontSize, fontweight='bold')
                props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
                ax.text(1.1, 0.5, 'Lower confidence:  {}\n            Prediction:  {}\nUpper confidence:  {}'.format(l_predict, predict, u_predict), 
                        transform=ax.transAxes, fontsize=small_fontSize, fontweight='bold', verticalalignment='bottom', bbox=props)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.savefig(save_dir + '/' + well_name + '-' + analyte_name +'.png', bbox_inches="tight")

            except:
                print('ERROR: Something went wrong')
                return None
    

    # Description: 
    #    Gernates a PCA biplot (PCA score plot + loading plot) of the data given a date in the dataset. Only uses the 6 important analytes. The data is also clustered into n_clusters.
    # Parameters:
    #    date (string): date to be analyzed
    #    n_clusters (int): number of clusters to split the data into.
    #    filter (bool): Flag to indicate well filtering.
    #    filter_well_by (list of strings): Letter of the well to be filtered (e.g. [‘A’] or [‘A’, ‘D’])
    #    return_clusters (bool): Flag to return the cluster data to be used for spatial plotting.
    #    min_samples (int): minimum number of samples the result should contain in order to execute.
    #    show_labels (bool): choose whether or not to show the name of the wells.
    #    save_dir (string): name of the directory you want to save the plot to
    def plot_PCA_by_date(self, date, n_clusters=4, filter=False, filter_well_by=['D'], return_clusters=False, min_samples=48, show_labels=True, save_dir='plot_PCA_by_date'):
        data = self.data
        data = self.simplify_data(data=data)
        query = data[data.COLLECTION_DATE == date]
        a = list(np.unique(query.ANALYTE_NAME.values))
        b = ['TRITIUM','IODINE-129','SPECIFIC CONDUCTANCE', 'PH','URANIUM-238', 'DEPTH_TO_WATER']
        analytes = self.__custom_analyte_sort(list(set(a) and set(b)))
        query = query.loc[query.ANALYTE_NAME.isin(analytes)]

        if(query.shape[0] == 0):
            return 'ERROR: {} has no data for the 6 analytes.'.format(date)
        samples = query[['COLLECTION_DATE', 'STATION_ID', 'ANALYTE_NAME']].duplicated().value_counts()[0]
        if(samples < min_samples):
            return 'ERROR: {} does not have at least {} samples.'.format(date, min_samples)
        if(len(np.unique(query.ANALYTE_NAME.values)) < 6):
            return 'ERROR: {} has less than the 6 analytes we want to analyze.'.format(date)
        else:
            analytes = self.__custom_analyte_sort(np.unique(query.ANALYTE_NAME.values))
            piv = query.reset_index().pivot_table(index = 'STATION_ID', columns='ANALYTE_NAME', values='RESULT',aggfunc=np.mean)
            
            # FILTERING CODE
            main_data = piv.dropna()
            if(filter):
                res_wells = self.filter_wells(filter_well_by)
                main_data = main_data.loc[main_data.index.isin(res_wells)]
            
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
                color_df = pd.DataFrame(merge(stations, color_wells), columns=['station_id', 'color'])
                if(self.get_GPS_Data==None):
                    print('You need to set the GPS data first using the set_GPS_Data function.')
                    return None
                else:
                    gps_color = pd.merge(self.get_GPS_Data(), color_df, on=['station_id'])
                    return gps_color
    
    
    # Description: 
    #    Gernates a PCA biplot (PCA score plot + loading plot) of the data given a year in the dataset. Only uses the 6 important analytes. The data is also clustered into n_clusters.
    # Parameters:
    #    year (int): date to be analyzed
    #    n_clusters (int): number of clusters to split the data into.
    #    filter (bool): Flag to indicate well filtering.
    #    filter_well_by (list of strings): Letter of the well to be filtered (e.g. [‘A’] or [‘A’, ‘D’])
    #    return_clusters (bool): Flag to return the cluster data to be used for spatial plotting.
    #    min_samples (int): minimum number of samples the result should contain in order to execute.
    #    show_labels (bool): choose whether or not to show the name of the wells.
    #    save_dir (string): name of the directory you want to save the plot to
    def plot_PCA_by_year(self, year, n_clusters=4, filter=False, filter_well_by=['D'], return_clusters=False, min_samples=48, show_labels=True, save_dir='plot_PCA_by_year'):
        data = self.data
        query = self.simplify_data(data=data)
        query.COLLECTION_DATE = pd.to_datetime(query.COLLECTION_DATE)
        query = query[query.COLLECTION_DATE.dt.year == year]
        a = list(np.unique(query.ANALYTE_NAME.values))
        b = ['TRITIUM','IODINE-129','SPECIFIC CONDUCTANCE', 'PH','URANIUM-238', 'DEPTH_TO_WATER']
        analytes = self.__custom_analyte_sort(list(set(a) and set(b)))
        query = query.loc[query.ANALYTE_NAME.isin(analytes)]

        if(query.shape[0] == 0):
            return 'ERROR: {} has no data for the 6 analytes.'.format(year)
        samples = query[['COLLECTION_DATE', 'STATION_ID', 'ANALYTE_NAME']].duplicated().value_counts()[0]
        if(samples < min_samples):
            return 'ERROR: {} does not have at least {} samples.'.format(year, min_samples)
        if(len(np.unique(query.ANALYTE_NAME.values)) < 6):
            return 'ERROR: {} has less than the 6 analytes we want to analyze.'.format(year)
        else:
            analytes = self.__custom_analyte_sort(np.unique(query.ANALYTE_NAME.values))
            piv = query.reset_index().pivot_table(index = 'STATION_ID', columns='ANALYTE_NAME', values='RESULT',aggfunc=np.mean)
            
            # FILTERING CODE
            main_data = piv.dropna()
            if(filter):
                res_wells = self.filter_wells(filter_well_by)
                main_data = main_data.loc[main_data.index.isin(res_wells)]
            
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
                color_df = pd.DataFrame(merge(stations, color_wells), columns=['station_id', 'color'])
                if(self.get_GPS_Data==None):
                    print('You need to set the GPS data first using the set_GPS_Data function.')
                    return None
                else:
                    gps_color = pd.merge(self.get_GPS_Data(), color_df, on=['station_id'])
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
    def plot_PCA_by_well(self, well_name, interpolate=False, frequency='2W', min_samples=48, show_labels=True, save_dir='plot_PCA_by_well'):
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
        if(len(np.unique(query.ANALYTE_NAME.values)) < 6):
            return 'ERROR: {} has less than the 6 analytes we want to analyze.'.format(date)
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

            loc = [row.loc['latitude'],row.loc['longitude']]
            station = HTML(value=row.loc['station_id'])

            marker = Marker(location=loc,
                            icon=icon,
                            draggable=False,
                       )

            m.add_layer(marker)

            popup = Popup(child=station,
                          max_height=1)

            marker.popup = popup

        return m
    
    # IN THE WORKS
    def transform_time_series(self, analytes=[], resample='2W', remove_outliers=False, z_threshold=4):
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
            if(remove_outliers):
                ana_data = self.remove_outliers(ana_data, z_threshold=z_threshold)
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
    
    def cluster_data(self, data, n_clusters=4, log_transform=False, filter=False, filter_well_by=['D'], return_clusters=False):
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
            color_df['station_id'] = color_df.index
            if(self.get_GPS_Data==None):
                print('You need to set the GPS data first using the set_GPS_Data function.')
                return None
            else:
                gps_color = pd.merge(self.get_GPS_Data(), color_df, on=['station_id'])
                return gps_color