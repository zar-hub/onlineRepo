import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# a container for keys
class k:
    freq = 'frequency_GHz'
    pek = 'peak_mV'
    sig_pek = 'sigma_peak_mV'
    nse = 'noise_mV'
    sig_nse = 'sigma_noise_mV'

def genLabel(df): 
        '''
            Generates the label for calibration_effects
        '''
        return ' '.join([str(df['sample'][0]),  str(df['antenna'][0])] )


def plotThis(subset : pd.DataFrame, ax):
        label_root = genLabel(subset)
        for mylabel in ['peak_mV', 'noise_mV']:
                
                # handle non existing cols and nan values
                if mylabel not in subset.columns:
                        continue
                if subset[mylabel].isnull().values.any():
                        continue

                # errors? errorbar, standard plot otherwise
                if subset['sigma_' + mylabel].isnull().values.any():
                        ax.plot(subset['frequency_GHz'], subset[mylabel], label = ' '.join([label_root , mylabel]))
                else:
                        ax.errorbar(subset['frequency_GHz'], subset[mylabel], 
                                yerr = subset['sigma_' + mylabel],
                                label = ' '.join([label_root , mylabel]))
                                                            

def plotByID(subset : pd.DataFrame, title : str):
        '''
                Draws the entire dataframe grouping measurements by id.
                The result is a graph with one line per measurement run.
        '''
        fig, ax = plt.subplots()
        
        subset.groupby(subset['id']).apply(plotThis, ax, include_groups=False)

        # set y scale to logarithmic
        # ax.set_yscale('log')
        
        fig.suptitle(title)
        ax.grid(True)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Power (mV)')
        ax.legend()

if __name__ == 'main':
    pass