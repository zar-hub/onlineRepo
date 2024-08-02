import numpy as np
import pandas as pd
import src.utils as utils


class FilterPipeline:
    '''
        This is a disposable class, should be created and destroyed 
        after use. It acts only as a wrapper to other functions and
        simplifies the code readibility.
        
        Raw data goes in and Filtered data comes out.
        Each step has some parameters that can be changed, but the 
        data can not be passed in a pipe, only at the beginning.
    '''
    def __init__(self, payload: pd.DataFrame) -> None:
        '''
            payload: contains the data, it's a list of 
            the measurements made. Each row of the dataframe
            has a unique id and contains all the informations
            about a particular measurement.
        '''
        self.payload = payload
        pass
    
    def denoise(self):
        self.payload = self.payload.apply(utils.denoise, axis=1)
        return self
    
    def corrGain(self, coeff : np.ndarray ):
        '''
        Scales every relevant field by the polinomial specyfied by the 
        coeff
        '''
        self.payload = utils.correctGain(self.payload, coeff)
        return self