import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def genLabel(df): 
        '''
            Generates the label for calibration_effects
        '''
        return ' '.join([df['sample'][0],  df['antenna'][0]])


def plot_this(subset : pd.DataFrame, ax):
                label_root = genLabel(subset)
                if subset['sigma_peak_mV'].isnull().values.any():
                        ax.plot(subset['frequency_GHz'], subset['peak_mV'], label = ' '.join([label_root , 'signal']))
                        ax.plot(subset['frequency_GHz'], subset['noise_mV'], label = ' '.join([label_root , 'noise']))
                else:
                        ax.errorbar(subset['frequency_GHz'], subset['peak_mV'], 
                                    yerr = subset['sigma_peak_mV'],
                                    label = ' '.join([label_root , 'signal']))
                        ax.errorbar(subset['frequency_GHz'], subset['noise_mV'],
                                    yerr = subset['sigma_noise_mV'],
                                    label = ' '.join([label_root , 'noise']))

if __name__ == 'main':
    pass