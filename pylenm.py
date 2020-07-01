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
import matplotlib.dates as mdates
from matplotlib.dates import date2num, num2date
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing
pd.set_option('display.max_columns', None) # to view all columns
from scipy.optimize import curve_fit
from supersmoother import SuperSmoother
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")


# Description:
#    Removes all columns except 'COLLECTION_DATE', 'STATION_ID', 'ANALYTE_NAME', 'RESULT', and 'RESULT_UNITS'.
#    Saves the result as a csv file in the same directory as the original file.
# Parameters:
#    dataPath: path of the dataset on the computer.
def simplify_csv_Data(dataPath):
    if(os.path.isfile(dataPath)==False):
        print('ERROR: "', dataPath, '" does not exist.')
    else:
        data = pd.read_csv(dataPath)
        sel_cols = ['COLLECTION_DATE','STATION_ID','ANALYTE_NAME','RESULT','RESULT_UNITS']
        data = data[sel_cols]
        data.COLLECTION_DATE = pd.to_datetime(data.COLLECTION_DATE)
        data = data.sort_values(by="COLLECTION_DATE")
        dup = data[data.duplicated(['COLLECTION_DATE', 'STATION_ID','ANALYTE_NAME', 'RESULT'])]
        data = data.drop(dup.index)
        newPath = 'data/FASB_Data_thru_3Q2015.csv'
        newPath = newPath.strip('.csv') + '_simplified' + '.csv'
        data.to_csv(newPath, index=False)
        if(os.path.isfile(newPath)):
            print("File saved successfully!")
        else:
            print("ERROR: File not saved!")


def get_MCL(analyte_name):
    mcl_dictionary = {'TRITIUM': 1.3, 'URANIUM-238': 1.31,  'NITRATE-NITRITE AS NITROGEN': 1,  'TECHNETIUM-99': 2.95}
    return mcl_dictionary[analyte_name]


# In[4]:


# Description:
#    Returns the unit of the analyte you specify.
#    Example: 'DEPTH_TO_WATER' returns 'ft'
# Parameters:
#    analyte_name: name of the analyte to be processed
def get_unit(analyte_name):
    unit_dictionary = {"1,1'-BIPHENYL": 'ug/L', '1,1,1,2-TETRACHLOROETHANE': 'ug/L', '1,1,1-TRICHLOROETHANE': 'ug/L', '1,1,2,2-TETRACHLOROETHANE': 'ug/L', '1,1,2-TRICHLORO-1,2,2-TRIFLUOROETHANE': 'ug/L', '1,1,2-TRICHLOROETHANE': 'ug/L', '1,1-DICHLOROETHANE': 'ug/L', '1,1-DICHLOROETHYLENE': 'ug/L', '1,1-DICHLOROPROPENE': 'ug/L', '1,2,3,4,6,7,8-HPCDD': 'ng/L', '1,2,3,4,6,7,8-HPCDF': 'ng/L', '1,2,3,4,7,8-HXCDD': 'ng/L', '1,2,3,4,7,8-HXCDF': 'ng/L', '1,2,3,7,8-PCDF': 'ng/L', '1,2,3-TRICHLOROBENZENE': 'ug/L', '1,2,3-TRICHLOROPROPANE': 'ug/L', '1,2,4,5-TETRACHLOROBENZENE': 'ug/L', '1,2,4-TRICHLOROBENZENE': 'ug/L', '1,2-DIBROMO-3-CHLOROPROPANE': 'ug/L', '1,2-DIBROMOETHANE': 'ug/L', '1,2-DICHLOROBENZENE': 'ug/L', '1,2-DICHLOROETHANE (EDC)': 'ug/L', '1,2-DICHLOROETHYLENE': 'ug/L', '1,2-DICHLOROPROPANE': 'ug/L', '1,2-DIPHENYLHYDRAZINE': 'ug/L', '1,3,5-TRIMETHYLBENZENE': 'ug/L', '1,3,5-TRINITROBENZENE': 'ug/L', '1,3-DICHLOROBENZENE': 'ug/L', '1,3-DICHLOROPROPANE': 'ug/L', '1,3-DINITROBENZENE': 'ug/L', '1,4-DICHLOROBENZENE': 'ug/L', '1,4-DIOXANE': 'ug/L', '1,4-NAPHTHOQUINONE': 'ug/L', '1-NAPHTHYLAMINE': 'ug/L', '2,2-DICHLOROPROPANE': 'ug/L', '2,3,4,6-TETRACHLOROPHENOL': 'ug/L', '2,3,7,8-TCDD': 'ng/L', '2,3,7,8-TCDF': 'ng/L', '2,4,5-T': 'ug/L', '2,4,5-TP (SILVEX)': 'ug/L', '2,4,5-TRICHLOROPHENOL': 'ug/L', '2,4,6-TRICHLOROPHENOL': 'ug/L', '2,4-DICHLOROPHENOL': 'ug/L', '2,4-DICHLOROPHENOXYACETIC ACID (2,4-D)': 'ug/L', '2,4-DIMETHYLPHENOL': 'ug/L', '2,4-DINITROPHENOL': 'ug/L', '2,4-DINITROTOLUENE': 'ug/L', '2,6-DICHLOROPHENOL': 'ug/L', '2,6-DINITROTOLUENE': 'ug/L', '2-ACETYLAMINOFLUORENE': 'ug/L', '2-CHLOROETHYL VINYL ETHER': 'ug/L', '2-CHLORONAPHTHALENE': 'ug/L', '2-CHLOROPHENOL': 'ug/L', '2-HEXANONE': 'ug/L', '2-METHYLANILINE (O-TOLUIDINE)': 'ug/L', '2-METHYLNAPHTHALENE': 'ug/L', '2-NAPHTHYLAMINE': 'ug/L', '2-NITROANILINE': 'ug/L', '2-NITROPHENOL': 'ug/L', '2-NITROPROPANE': 'ug/L', '2-PICOLINE': 'ug/L', "3,3'-DIMETHYLBENZIDINE": 'ug/L', '3,3-DICHLOROBENZIDINE': 'ug/L', '3-METHYLCHOLANTHRENE': 'ug/L', '4-AMINOBIPHENYL': 'ug/L', '4-BROMOPHENYL PHENYL ETHER': 'ug/L', '4-CHLOROANILINE': 'ug/L', '4-CHLOROPHENYL PHENYL ETHER': 'ug/L', '4-CHLOROTOLUENE': 'ug/L', '4-NITROPHENOL': 'ug/L', '4-NITROQUINOLINE-1-OXIDE': 'ug/L', '5-NITRO-O-TOLUIDINE': 'ug/L', '7,12-DIMETHYLBENZ(A)ANTHRACENE': 'ug/L', 'A,A-DIMETHYLPHENETHYLAMINE': 'ug/L', 'ACENAPHTHENE': 'ug/L', 'ACENAPHTHYLENE': 'ug/L', 'ACETONE': 'ug/L', 'ACETONITRILE (METHYL CYANIDE)': 'ug/L', 'ACETOPHENONE': 'ug/L', 'ACROLEIN': 'ug/L', 'ACRYLONITRILE': 'ug/L', 'ACTINIUM-228': 'pCi/L', 'AIR TEMPERATURE': 'degC', 'ALDRIN': 'ug/L', 'ALLYL CHLORIDE': 'ug/L', 'ALPHA-BENZENE HEXACHLORIDE': 'ug/L', 'ALPHA-CHLORDANE': 'ug/L', 'ALUMINUM': 'ug/L', 'AMERICIUM-241': 'pCi/L', 'AMERICIUM-241/CURIUM-246': 'pCi/L', 'AMERICIUM-243': 'pCi/L', 'AMMONIA': 'mg/L', 'ANILINE': 'ug/L', 'ANTHRACENE': 'ug/L', 'ANTIMONY': 'ug/L', 'ANTIMONY-124': 'pCi/L', 'ANTIMONY-125': 'pCi/L', 'ARAMITE': 'ug/L', 'AROCLOR 1016': 'ug/L', 'AROCLOR 1221': 'ug/L', 'AROCLOR 1232': 'ug/L', 'AROCLOR 1242': 'ug/L', 'AROCLOR 1248': 'ug/L', 'AROCLOR 1254': 'ug/L', 'AROCLOR 1260': 'ug/L', 'ARSENIC': 'ug/L', 'ATRAZINE': 'ug/L', 'BARIUM': 'ug/L', 'BARIUM-133': 'pCi/L', 'BARIUM-140': 'pCi/L', 'BENZALDEHYDE': 'ug/L', 'BENZENE': 'ug/L', 'BENZIDINE': 'ug/L', 'BENZO(G,H,I)PERYLENE': 'ug/L', 'BENZOIC ACID': 'ug/L', 'BENZO[A]ANTHRACENE': 'ug/L', 'BENZO[A]PYRENE': 'ug/L', 'BENZO[B]FLUORANTHENE': 'ug/L', 'BENZO[K]FLUORANTHENE': 'ug/L', 'BENZYL ALCOHOL': 'ug/L', 'BERYLLIUM': 'ug/L', 'BERYLLIUM-7': 'pCi/L', 'BETA-BENZENE HEXACHLORIDE': 'ug/L', 'BIS(2-CHLORO-1-METHYLETHYL)ETHER': 'ug/L', 'BIS(2-CHLOROETHOXY)METHANE': 'ug/L', 'BIS(2-CHLOROETHYL)ETHER': 'ug/L', 'BIS(2-ETHYLHEXYL)PHTHALATE (DEHP)': 'ug/L', 'BIS(CHLOROMETHYL)ETHER': 'ug/L', 'BISMUTH-212': 'pCi/L', 'BISMUTH-214': 'pCi/L', 'BORON': 'ug/L', 'BROMIDE': 'mg/L', 'BROMOBENZENE': 'ug/L', 'BROMOCHLOROMETHANE': 'ug/L', 'BROMODICHLOROMETHANE': 'ug/L', 'BROMOFORM (TRIBROMOMETHANE)': 'ug/L', 'BROMOMETHANE (METHYL BROMIDE)': 'ug/L', 'BUTYL BENZYL PHTHALATE': 'ug/L', 'CADMIUM': 'ug/L', 'CALCIUM': 'ug/L', 'CAPROLACTAM': 'ug/L', 'CARBAZOLE': 'ug/L', 'CARBON 13-LABELED 2,3,7,8-TCDD': 'ng/L', 'CARBON 13-LABELED 2,3,7,8-TCDF': 'ng/L', 'CARBON DISULFIDE': 'ug/L', 'CARBON TETRACHLORIDE': 'ug/L', 'CARBON-14': 'pCi/L', 'CARBONATE': 'mg/L', 'CERIUM-141': 'pCi/L', 'CERIUM-144': 'pCi/L', 'CESIUM': 'ug/L', 'CESIUM-134': 'pCi/L', 'CESIUM-137': 'pCi/L', 'CHLORIDE': 'mg/L', 'CHLOROBENZENE': 'ug/L', 'CHLOROBENZILATE': 'ug/L', 'CHLOROETHANE (ETHYL CHLORIDE)': 'ug/L', 'CHLOROETHENE (VINYL CHLORIDE)': 'ug/L', 'CHLOROFORM': 'ug/L', 'CHLOROMETHANE (METHYL CHLORIDE)': 'ug/L', 'CHLOROPRENE': 'ug/L', 'CHROMIUM': 'ug/L', 'CHROMIUM-51': 'pCi/L', 'CHRYSENE': 'ug/L', 'CIS-1,2-DICHLOROETHYLENE': 'ug/L', 'CIS-1,3-DICHLOROPROPENE': 'ug/L', 'COBALT': 'ug/L', 'COBALT-57': 'pCi/L', 'COBALT-58': 'pCi/L', 'COBALT-60': 'pCi/L', 'COPPER': 'ug/L', 'CUMENE (ISOPROPYLBENZENE)': 'ug/L', 'CURIUM-242': 'pCi/L', 'CURIUM-243': 'pCi/L', 'CURIUM-243/244': 'pCi/L', 'CURIUM-244': 'pCi/L', 'CURIUM-245/246': 'pCi/L', 'CURIUM-246': 'pCi/L', 'CYANIDE': 'ug/L', 'CYCLOHEXANE': 'ug/L', 'CYCLOHEXANONE': 'ug/L', 'DDD': 'ug/L', 'DDE': 'ug/L', 'DDT': 'ug/L', 'DELTA-BENZENE HEXACHLORIDE': 'ug/L', 'DEPTH_TO_WATER': 'ft', 'DI-N-BUTYL PHTHALATE': 'ug/L', 'DIALLATE': 'ug/L', 'DIBENZOFURAN': 'ug/L', 'DIBENZ[AH]ANTHRACENE': 'ug/L', 'DIBROMOCHLOROMETHANE': 'ug/L', 'DIBROMOMETHANE (METHYLENE BROMIDE)': 'ug/L', 'DICHLORODIFLUOROMETHANE': 'ug/L', 'DICHLOROMETHANE (METHYLENE CHLORIDE)': 'ug/L', 'DIELDRIN': 'ug/L', 'DIETHYL PHTHALATE': 'ug/L', 'DIMETHOATE': 'ug/L', 'DIMETHYL PHTHALATE': 'ug/L', 'DINITRO-O-CRESOL': 'ug/L', 'DINOSEB': 'ug/L', 'DIPHENYLAMINE': 'ug/L', 'DISULFOTON': 'ug/L', 'ENDOSULFAN I': 'ug/L', 'ENDOSULFAN II': 'ug/L', 'ENDOSULFAN SULFATE': 'ug/L', 'ENDRIN': 'ug/L', 'ENDRIN ALDEHYDE': 'ug/L', 'ENDRIN KETONE': 'ug/L', 'ETHANE': 'ug/L', 'ETHYL ACETATE': 'ug/L', 'ETHYL METHACRYLATE': 'ug/L', 'ETHYL METHANESULFONATE': 'ug/L', 'ETHYLBENZENE': 'ug/L', 'ETHYLENE': 'ug/L', 'EUROPIUM-152': 'pCi/L', 'EUROPIUM-154': 'pCi/L', 'EUROPIUM-155': 'pCi/L', 'FAMPHUR': 'ug/L', 'FLOW RATE': 'gpm', 'FLUORANTHENE': 'ug/L', 'FLUORENE': 'ug/L', 'FLUORIDE': 'mg/L', 'GAMMA-CHLORDANE': 'ug/L', 'GROSS ALPHA': 'pCi/L', 'HARDNESS AS CACO3': 'ug/L', 'HEPTACHLOR': 'ug/L', 'HEPTACHLOR EPOXIDE': 'ug/L', 'HEPTACHLORODIBENZO-P-DIOXINS': 'ng/L', 'HEPTACHLORODIBENZO-P-FURANS': 'ng/L', 'HEPTACHLORODIBENZOFURAN': 'ng/L', 'HEXACHLOROBENZENE': 'ug/L', 'HEXACHLOROBUTADIENE': 'ug/L', 'HEXACHLOROCYCLOPENTADIENE': 'ug/L', 'HEXACHLORODIBENZO-P-DIOXINS': 'ng/L', 'HEXACHLORODIBENZO-P-FURANS': 'ng/L', 'HEXACHLOROETHANE': 'ug/L', 'HEXACHLOROPHENE': 'ug/L', 'HEXACHLOROPROPENE': 'ug/L', 'HEXACHLORORDIBENZOFURAN': 'ng/L', 'HEXANE': 'ug/L', 'INDENO[1,2,3-CD]PYRENE': 'ug/L', 'IODINE-129': 'pCi/L', 'IODINE-131': 'pCi/L', 'IODOMETHANE (METHYL IODIDE)': 'ug/L', 'IRON': 'ug/L', 'IRON-55': 'pCi/L', 'IRON-59': 'pCi/L', 'ISOBUTANOL': 'ug/L', 'ISODRIN': 'ug/L', 'ISOPHORONE': 'ug/L', 'ISOSAFROLE': 'ug/L', 'KEPONE': 'ug/L', 'LEAD': 'ug/L', 'LEAD-212': 'pCi/L', 'LEAD-214': 'pCi/L', 'LINDANE': 'ug/L', 'LITHIUM': 'ug/L', 'M,P-XYLENE': 'ug/L', 'M-CRESOL': 'ug/L', 'M-NITROANILINE': 'ug/L', 'M/P-CRESOL': 'ug/L', 'MAGNESIUM': 'ug/L', 'MANGANESE': 'ug/L', 'MANGANESE-54': 'pCi/L', 'MERCURY': 'ug/L', 'METHACRYLONITRILE': 'ug/L', 'METHANE': 'ug/L', 'METHAPYRILENE': 'ug/L', 'METHOXYCHLOR': 'ug/L', 'METHYL ACETATE': 'ug/L', 'METHYL ETHYL KETONE': 'ug/L', 'METHYL ISOBUTYL KETONE': 'ug/L', 'METHYL METHACRYLATE': 'ug/L', 'METHYL METHANESULFONATE': 'ug/L', 'METHYL PARATHION': 'ug/L', 'METHYL TERTIARY BUTYL ETHER (MTBE)': 'ug/L', 'METHYLCYCLOHEXANE': 'ug/L', 'MOLYBDENUM': 'ug/L', 'N-BUTYLBENZENE': 'ug/L', 'N-DIOCTYL PHTHALATE': 'ug/L', 'N-NITROSO-N-METHYLETHYLAMINE': 'ug/L', 'N-NITROSODI-N-BUTYLAMINE': 'ug/L', 'N-NITROSODIETHYLAMINE': 'ug/L', 'N-NITROSODIMETHYLAMINE': 'ug/L', 'N-NITROSODIPHENYLAMINE': 'ug/L', 'N-NITROSODIPHENYLAMINE+DIPHENYLAMINE': 'ug/L', 'N-NITROSODIPROPYLAMINE': 'ug/L', 'N-NITROSOMORPHOLINE': 'ug/L', 'N-NITROSOPIPERIDINE': 'ug/L', 'N-NITROSOPYRROLIDINE': 'ug/L', 'N-PROPYLBENZENE': 'ug/L', 'NAPHTHALENE': 'ug/L', 'NEPTUNIUM-237': 'pCi/L', 'NEPTUNIUM-239': 'pCi/L', 'NICKEL': 'ug/L', 'NICKEL-59': 'pCi/L', 'NICKEL-63': 'pCi/L', 'NIOBIUM-95': 'pCi/L', 'NITRATE': 'mg/L', 'NITRATE-NITRITE AS NITROGEN': 'mg/L', 'NITRITES': 'mg/L', 'NITROBENZENE': 'ug/L', 'NONVOLATILE BETA': 'pCi/L', 'O,O,O-TRIETHYL PHOSPHOROTHIOATE': 'ug/L', 'O-CRESOL (2-METHYLPHENOL)': 'ug/L', 'O-XYLENE': 'ug/L', 'OCTACHLORODIBENZO-P-DIOXIN': 'ng/L', 'OCTACHLORODIBENZO-P-FURAN': 'ng/L', 'ORTHOCHLOROTOLUENE': 'ug/L', 'ORTHOPHOSPHATE': 'mg/L', 'OXALATE': 'mg/L', 'OXIDATION/REDUCTION POTENTIAL': 'mV', 'OXYGEN': 'mg/L', 'P-CHLORO-M-CRESOL': 'ug/L', 'P-CRESOL': 'ug/L', 'P-DIMETHYLAMINOAZOBENZENE': 'ug/L', 'P-NITROANILINE': 'ug/L', 'P-PHENYLENEDIAMINE': 'ug/L', 'PARACYMEN': 'ug/L', 'PARATHION': 'ug/L', 'PENTACHLOROBENZENE': 'ug/L', 'PENTACHLORODIBENZO-P-DIOXINS': 'ng/L', 'PENTACHLORODIBENZO-P-FURANS': 'ng/L', 'PENTACHLORODIBENZOFURAN': 'ng/L', 'PENTACHLOROETHANE': 'ug/L', 'PENTACHLORONITROBENZENE': 'ug/L', 'PENTACHLOROPHENOL': 'ug/L', 'PH': 'pH', 'PHENACETIN': 'ug/L', 'PHENANTHRENE': 'ug/L', 'PHENOL': 'ug/L', 'PHENOLPHTHALEIN ALKALINITY (AS CACO3)': 'mg/L', 'PHENOLS': 'mg/L', 'PHORATE': 'ug/L', 'PLUTONIUM-238': 'pCi/L', 'PLUTONIUM-239': 'pCi/L', 'PLUTONIUM-239/240': 'pCi/L', 'PLUTONIUM-242': 'pCi/L', 'POTASSIUM': 'ug/L', 'POTASSIUM-40': 'pCi/L', 'PROMETHIUM-144': 'pCi/L', 'PROMETHIUM-146': 'pCi/L', 'PRONAMIDE': 'ug/L', 'PROPIONITRILE': 'ug/L', 'PYRENE': 'ug/L', 'PYRIDINE': 'ug/L', 'RADIUM, TOTAL ALPHA-EMITTING': 'pCi/L', 'RADIUM-226': 'pCi/L', 'RADIUM-228': 'pCi/L', 'RADON-222': 'pCi/L', 'RUTHENIUM-103': 'pCi/L', 'RUTHENIUM-106': 'pCi/L', 'SAFROLE': 'ug/L', 'SEC-BUTYLBENZENE': 'ug/L', 'SELENIUM': 'ug/L', 'SILICA': 'ug/L', 'SILICON': 'ug/L', 'SILVER': 'ug/L', 'SODIUM': 'ug/L', 'SODIUM-22': 'pCi/L', 'SPECIFIC CONDUCTANCE': 'uS/cm', 'STRONTIUM': 'ug/L', 'STRONTIUM-89': 'pCi/L', 'STRONTIUM-89/90': 'pCi/L', 'STRONTIUM-90': 'pCi/L', 'STYRENE': 'ug/L', 'SULFATE': 'mg/L', 'SULFIDE': 'mg/L', 'SULFOTEPP': 'ug/L', 'SULFUR': 'ug/L', 'TECHNETIUM-99': 'pCi/L', 'TEMPERATURE': 'degC', 'TERT-BUTYLBENZENE': 'ug/L', 'TETRACHLORODIBENZO-P-DIOXIN': 'ng/L', 'TETRACHLORODIBENZO-P-FURANS': 'ng/L', 'TETRACHLORODIBENZOFURAN': 'ng/L', 'TETRACHLOROETHYLENE (PCE)': 'ug/L', 'THALLIUM': 'ug/L', 'THALLIUM-208': 'pCi/L', 'THIONAZIN': 'ug/L', 'THORIUM': 'ug/L', 'THORIUM-228': 'pCi/L', 'THORIUM-230': 'pCi/L', 'THORIUM-232': 'pCi/L', 'THORIUM-234': 'pCi/L', 'TIN': 'ug/L', 'TIN-113': 'pCi/L', 'TITANIUM': 'ug/L', 'TOLUENE': 'ug/L', 'TOTAL ACTIVITY': 'pCi/mL', 'TOTAL ALKALINITY (AS CACO3)': 'mg/L', 'TOTAL CHLORDANE': 'ug/L', 'TOTAL DISSOLVED SOLIDS': 'mg/L', 'TOTAL ORGANIC CARBON': 'mg/L', 'TOTAL ORGANIC HALOGENS': 'mg/L', 'TOTAL PHOSPHATES (AS  P)': 'ug/L', 'TOTAL SUSPENDED SOLIDS': 'mg/L', 'TOXAPHENE': 'ug/L', 'TRANS-1,2-DICHLOROETHYLENE': 'ug/L', 'TRANS-1,3-DICHLOROPROPENE': 'ug/L', 'TRANS-1,4-DICHLORO-2-BUTENE': 'ug/L', 'TRIBUTYL PHOSPHATE': 'ug/L', 'TRICHLOROETHYLENE (TCE)': 'ug/L', 'TRICHLOROFLUOROMETHANE': 'ug/L', 'TRICHLOROTRIFLUOROETHANE': 'ug/L', 'TRITIUM': 'pCi/mL', 'TURBIDITY': 'NTU', 'URANIUM': 'ug/L', 'URANIUM-233/234': 'pCi/L', 'URANIUM-234': 'pCi/L', 'URANIUM-235': 'pCi/L', 'URANIUM-235/236': 'pCi/L', 'URANIUM-238': 'pCi/L', 'VANADIUM': 'ug/L', 'VINYL ACETATE': 'ug/L', 'VOLUME PURGED': 'gal', 'WATER TEMPERATURE': 'degC', 'XYLENES': 'ug/L', 'YTTRIUM-88': 'pCi/L', 'ZINC': 'ug/L', 'ZINC-65': 'pCi/L', 'ZIRCONIUM-95': 'pCi/L'}
    return unit_dictionary[analyte_name]



# Description:
#    Returns a csv file saved to save_dir with details pertaining to the specified analyte.
#    Details include the well names, the date ranges and the number of unique samples.
# Parameters:
#    data: data to be processed
#    analyte_name: name of the analyte to be processed
#    save_dir: name of the directory you want to save the csv file to
def get_analyte_details(data, analyte_name, save_dir='analyte_details'):
    data = data[data.ANALYTE_NAME == analyte_name].reset_index().drop('index', axis=1)
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
                     'Date Range (days)':endDate-startDate ,
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
#    Filters data by passing the data and specifying the well_name and analyte_name
# Parameters:
#    data: data to be processed
#    well_name: name of the well to be processed
#    analyte_name: name of the analyte to be processed
def query_data(data, well_name, analyte_name):
    query = data[data.STATION_ID == well_name]
    query = query[query.ANALYTE_NAME == analyte_name]
    if(query.shape[0] == 0):
        return 0
    else:
        return query



# Helper function for plot_correlation
# Sorts analytes in a specific order: 'TRITIUM', 'URANIUM-238','IODINE-129','SPECIFIC CONDUCTANCE', 'PH', 'DEPTH_TO_WATER'
def custom_analyte_sort(analytes):
    my_order = 'TURISPDABCEFGHJKLMNOQVWXYZ-_abcdefghijklmnopqrstuvwxyz135790 2468'
    return sorted(analytes, key=lambda word: [my_order.index(c) for c in word])


# Description: 
#    Plot concentrations over time of a specified well and analyte with a smoothed curve on interpolated data points.
# Parameters:
#    data: data to be processed
#    well_name: name of the well to be processed
#    analyte_name: name of the analyte to be processed
#    log_transform: boolean(True or False) choose whether or not the data should be transformed to log base 10 values
#    alpha: value between 0 and 10 for line smoothing
#    year_interval: plot by how many years to appear in the axis e.g.(1 = every year, 5 = every 5 years, ...)
#    plot_inline: boolean(True or False) choose whether or not to show plot inline
#    save_dir: name of the directory you want to save the plot to
def plot_data(data, well_name, analyte_name, log_transform=True, alpha=0,
              plot_inline=True, year_interval=2, save_dir='plot_data'):
    
    # Gets appropriate data (well_name and analyte_name)
    query = query_data(data, well_name, analyte_name)
    
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
#    data: data to be processed
#    log_transform: boolean(True or False) choose whether or not the data should be transformed to log base 10 values
#    alpha: value between 0 and 10 for line smoothing
#    year_interval: plot by how many years to appear in the axis e.g.(1 = every year, 5 = every 5 years, ...)
#    plot_inline: boolean(True or False) choose whether or not to show plot inline
#    save_dir: name of the directory you want to save the plot to
def plot_all_data(data, log_transform=True, alpha=0, year_interval=2, plot_inline=True, save_dir='plot_data'):
    analytes = ['TRITIUM','URANIUM-238','IODINE-129','SPECIFIC CONDUCTANCE', 'PH', 'DEPTH_TO_WATER']
    wells = np.array(data.STATION_ID.values)
    wells = np.unique(wells)
    success = 0
    errors = 0
    for well in wells:
        for analyte in analytes:
            plot = plot_data(data, well, analyte, 
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
#    data: data to be processed
#    well_name: name of the well to be processed
#    show_symmetry: boolean(True or False) choose whether or not the heatmap should show the same information twice over the diagonal
#    color: boolean(True or False) choose whether or not the plot should be in color or in greyscale
#    save_dir: name of the directory you want to save the plot to
def plot_correlation_heatmap(data, well_name, show_symmetry=True, color=True, save_dir='plot_correlation_heatmap'):
    query = data[data.STATION_ID == well_name]
    a = list(np.unique(query.ANALYTE_NAME.values))
    b = ['TRITIUM','IODINE-129','SPECIFIC CONDUCTANCE', 'PH','URANIUM-238', 'DEPTH_TO_WATER']
    analytes = custom_analyte_sort(list(set(a) and set(b)))
    query = query.loc[query.ANALYTE_NAME.isin(analytes)]
    analytes = custom_analyte_sort(np.unique(query.ANALYTE_NAME.values))
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
#    data: data to be processed
#    show_symmetry: boolean(True or False) choose whether or not the heatmap should show the same information twice over the diagonal
#    color: boolean(True or False) choose whether or not the plot should be in color or in greyscale
#    save_dir: name of the directory you want to save the plot to
def plot_all_correlation_heatmap(data, show_symmetry=True, color=True, save_dir='plot_correlation_heatmap'):
    wells = np.array(data.STATION_ID.values)
    wells = np.unique(wells)
    for well in wells:
        plot_correlation_heatmap(data, well_name=well,
                                 show_symmetry=show_symmetry,
                                 color=color,
                                 save_dir=save_dir)

# Description: 
#    Resamples the data based on the frequency specified and interpolates the values of the analytes.
# Parameters:
#    data: data to be processed
#    well_name: name of the well to be processed
#    analytes: list of analyte names to use
#    frequency: frequency to interpolate. See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html for valid frequency inputs. 
#        (e.g. ‘W’ = every week, ‘D ’= every day, ‘2W’ = every 2 weeks)
def interpolate_well_data(data, well_name, analytes, frequency='2W'):
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
#    data: data to be processed
#    well_name: name of the well to be processed
#    interpolate: boolean(True or False) choose whether or to interpolate the data
#    frequency: frequency to interpolate.
#    save_dir: name of the directory you want to save the plot to
def plot_corr_by_well(data, well_name, interpolate=False, frequency='2W', save_dir='plot_correlation'):
    def plotUpperHalf(*args, **kwargs):
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
    query = data[data.STATION_ID == well_name]
    a = list(np.unique(query.ANALYTE_NAME.values))
    b = ['TRITIUM','IODINE-129','SPECIFIC CONDUCTANCE', 'PH','URANIUM-238', 'DEPTH_TO_WATER']
    analytes = custom_analyte_sort(list(set(a) and set(b)))
    query = query.loc[query.ANALYTE_NAME.isin(analytes)]
    analytes = custom_analyte_sort(np.unique(query.ANALYTE_NAME.values))
    x = query[['COLLECTION_DATE', 'ANALYTE_NAME']]
    unique = ~x.duplicated()
    query = query[unique]
    piv = query.reset_index().pivot(index='COLLECTION_DATE',columns='ANALYTE_NAME', values='RESULT')
    piv = piv[analytes]
    piv.index = pd.to_datetime(piv.index)
    totalSamples = piv.shape[0]
    piv = piv.dropna()
    if(interpolate):
        piv = interpolate_well_data(data, well_name, analytes, frequency=frequency)
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

        sns.set_style("white", {"axes.facecolor": "0.95"})
        g = sns.PairGrid(piv, aspect=1.2, diag_sharey=False, despine=False)
        g.fig.suptitle(title, fontweight='bold', y=1.08, fontsize=25)
        g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'red', 'lw': 3},
                                                        scatter_kws={'color': 'black', 's': 20})
        g.map_diag(sns.distplot, kde_kws={'color': 'black', 'lw': 3},
                                 hist_kws={'histtype': 'bar', 'lw': 2, 'edgecolor': 'k', 'facecolor':'grey'})
        g.map_upper(plotUpperHalf)
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
#    data: data to be processed
#    interpolate: boolean(True or False) choose whether or to interpolate the data
#    frequency: frequency to interpolate.
#    save_dir: name of the directory you want to save the plot to
def plot_all_corr_by_well(data, interpolate=False, frequency='2W', save_dir='plot_correlation'):
    wells = np.array(data.STATION_ID.values)
    wells = np.unique(wells)
    for well in wells:
        plot_correlation(data, well_name=well, interpolate=interpolate, frequency=frequency, save_dir=save_dir)


# Description: 
#    Plots the correlations with the physical plots as well as the correlations of the important analytes for ALL the wells on a specified date.
# Parameters:
#    data: data to be processed
#    date: date to be analyzed
#    min_samples: minimum number of samples the result should contain in order to execute.
#    save_dir: name of the directory you want to save the plot to
def plot_corr_by_date(data, date, min_samples=48, save_dir='plot_corr_by_date'):
    def plotUpperHalf(*args, **kwargs):
        corr_r = args[0].corr(args[1], 'pearson')
        corr_text = f"{corr_r:2.2f}"
        ax = plt.gca()
        ax.set_axis_off()
        marker_size = abs(corr_r) * 10000
        ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
                   vmin=-1, vmax=1, transform=ax.transAxes)
        font_size = abs(corr_r) * 40 + 5
        ax.annotate(corr_text, [.5, .48,],  xycoords="axes fraction",
                    ha='center', va='center', fontsize=font_size, fontweight='bold')
    query = data[data.COLLECTION_DATE == date]
    a = list(np.unique(query.ANALYTE_NAME.values))
    b = ['TRITIUM','IODINE-129','SPECIFIC CONDUCTANCE', 'PH','URANIUM-238', 'DEPTH_TO_WATER']
    analytes = custom_analyte_sort(list(set(a) and set(b)))
    query = query.loc[query.ANALYTE_NAME.isin(analytes)]
    
    if(query.shape[0] == 0):
        return 'ERROR: {} has no data for the 6 analytes.'.format(date)
    samples = query[['COLLECTION_DATE', 'STATION_ID', 'ANALYTE_NAME']].duplicated().value_counts()[0]
    if(samples < min_samples):
        return 'ERROR: {} does not have at least {} samples.'.format(date, min_samples)
    if(len(np.unique(query.ANALYTE_NAME.values)) < 6):
        return 'ERROR: {} has less than the 6 analytes we want to analyze.'.format(date)
    else:
        analytes = custom_analyte_sort(np.unique(query.ANALYTE_NAME.values))
        piv = query.reset_index().pivot_table(index = 'STATION_ID', columns='ANALYTE_NAME', values='RESULT',aggfunc=np.mean)
        title = date + '_correlation'

        sns.set_style("white", {"axes.facecolor": "0.95"})
        g = sns.PairGrid(piv, aspect=1.2, diag_sharey=False, despine=False)
        g.fig.suptitle(title, fontweight='bold', y=1.08, fontsize=25)
        g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'red', 'lw': 3},
                                                        scatter_kws={'color': 'black', 's': 20})
        g.map_diag(sns.distplot, kde_kws={'color': 'black', 'lw': 3},
                                 hist_kws={'histtype': 'bar', 'lw': 2, 'edgecolor': 'k', 'facecolor':'grey'})
        g.map_upper(plotUpperHalf)
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
#    data: data to be processed
#    year: date to be analyzed
#    min_samples: minimum number of samples the result should contain in order to execute.
#    save_dir: name of the directory you want to save the plot to
def plot_corr_by_year(data, year, min_samples=500, save_dir='plot_corr_by_year'):
    def plotUpperHalf(*args, **kwargs):
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
    query = data
    query.COLLECTION_DATE = pd.to_datetime(query.COLLECTION_DATE)
    query = query[query.COLLECTION_DATE.dt.year == year]
    a = list(np.unique(query.ANALYTE_NAME.values))
    b = ['TRITIUM','IODINE-129','SPECIFIC CONDUCTANCE', 'PH','URANIUM-238', 'DEPTH_TO_WATER']
    analytes = custom_analyte_sort(list(set(a) and set(b)))
    query = query.loc[query.ANALYTE_NAME.isin(analytes)]
    if(query.shape[0] == 0):
        return 'ERROR: {} has no data for the 6 analytes.'.format(year)
    samples = query[['COLLECTION_DATE', 'STATION_ID', 'ANALYTE_NAME']].duplicated().value_counts()[0]
    if(samples < min_samples):
        return 'ERROR: {} does not have at least {} samples.'.format(date, min_samples)
    if(len(np.unique(query.ANALYTE_NAME.values)) < 6):
        return 'ERROR: {} has less than the 6 analytes we want to analyze.'.format(year)
    else:
        analytes = custom_analyte_sort(np.unique(query.ANALYTE_NAME.values))
        piv = query.reset_index().pivot_table(index = 'STATION_ID', columns='ANALYTE_NAME', values='RESULT',aggfunc=np.mean)
        title = str(year) + '_correlation'

        sns.set_style("white", {"axes.facecolor": "0.95"})
        g = sns.PairGrid(piv, aspect=1.2, diag_sharey=False, despine=False)
        g.fig.suptitle(title, fontweight='bold', y=1.08, fontsize=25)
        g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'red', 'lw': 3},
                                                        scatter_kws={'color': 'black', 's': 20})
        g.map_diag(sns.distplot, kde_kws={'color': 'black', 'lw': 3},
                                 hist_kws={'histtype': 'bar', 'lw': 2, 'edgecolor': 'k', 'facecolor':'grey'})
        g.map_upper(plotUpperHalf)
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
#    data: data to be processed
#    well_name: name of the well to be processed
#    analyte_name: name of the analyte to be processed
#    year_interval: plot by how many years to appear in the axis e.g.(1 = every year, 5 = every 5 years, ...)
#    save_dir: name of the directory you want to save the plot to
def plot_MCL(data, well_name, analyte_name, year_interval=5, save_dir='plot_MCL'):
    # finds the intersection point of 2 lines given the slopes and y-intercepts
    def line_intersect(m1, b1, m2, b2):
        if m1 == m2:
            print ('The lines are parallel')
            return None
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
        return x,y
    
    # Gets appropriate data (well_name and analyte_name)
    query = query_data(data, well_name, analyte_name)
    
    if(type(query)==int and query == 0):
        return 'No results found for {} and {}'.format(well_name, analyte_name)
    else:   

        test = query.groupby(['COLLECTION_DATE']).mean()
        test.index = pd.to_datetime(test.index)
    
        x = date2num(test.index)
        y = np.log10(test.RESULT)
        ylabel = 'log-Concentration (' + get_unit(analyte_name) + ')'
        y = y.rename(ylabel)
        
        p, cov = np.polyfit(x, y, 1, cov=True)  # parameters and covariance from of the fit of 1-D polynom.
        
        m_unc = np.sqrt(cov[0][0])
        b_unc = np.sqrt(cov[1][1])
        
        f = np.poly1d(p)

        try:
            MCL = get_MCL(analyte_name)
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
            ax.set_ylabel('log-Concentration (' + get_unit(analyte_name) + ')')

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