from ._imports import *

class init:
    """Object that initializes Pylenm.
    """
    
    def __init__(self, data: pd.DataFrame):
        """some details here

        Args:
            data (pd.DataFrame): Data to be imported.
        """
        self.setData(data)
        self.__jointData = [None, 0]

    # SETTING DATA
    def setData(self, data: pd.DataFrame, verbose: bool = True) -> None:
        """Saves the dataset into pylenm

        Args:
            data (pd.DataFrame): Dataset to be imported.
            verbose (bool, optional): Prints success message. Defaults to True.

        Returns:
            None: ggg
        """       
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

    def setConstructionData(self, construction_data: pd.DataFrame, verbose=True):
        """Imports the addtitional well information as a separate DataFrame.

        Args:
            construction_data (pd.DataFrame): Data with additonal details.
            verbose (bool, optional): Prints success message. Defaults to True.

        Returns:
            None: hhh
        """
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