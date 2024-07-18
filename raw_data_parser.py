'''
    Gist: converts raw data into a dataset ready for data analysis.
    The main inconvinience is that measurements in different days have 
    different fields. 
    To overcome this problem each run has an unique id and some fields 
    common to every measure, some of wich are left empty.

    Fields:
        NAME            TYPE            DESCRIPTION
        - id            [date, int]     unique identifier for the run     
        - frequency     float           (Ghz)
        - peak          float           (mV)
        - sigma_peak    float           (mV)
        - noise         float           (mV)
        - sigma_noise   float           (mv)
        - sample        string          the sampling obj, can be one of
                                        PCR:        photonic crystal
                                        SIL:        silica
                                        SILHALF:    only half of the box is 
                                                    full of silica   
                                        SILFULL:    only half of the box is 
                                                    full of silica   
                                        AIR
                                        CAL: calibration, the reciever is
                                        directly connected with the antenna
        - antenna       string          can be HF, LF, CAB
                                        both lowFreq and highFreq are homemade

    Notes:
        - span is omitted for simplicity
        - LNA is omitted because almost all the data uses LNA
'''

import numpy as np
import pandas as pd
from os import listdir, getcwd
from os.path import isfile, join

sample_types = ['CAL', 'AIR', 'PCR', 'SIL', 'SILHALF', 'SILFULL', 'BOX']
antenna_types = ['HF', 'LF', 'CAB']

def getType( filename : str , from_types : list):
    for this_type in from_types:
        if filename.find(this_type) != -1:
            return this_type
    raise OSError(f'{filename} does not have a match... aborting')

prj_dir = getcwd()
rel_path = 'onlineRepo/rawdata'

# 30-05-24
date = '30-05-24'
full_path = join(prj_dir, rel_path, date)

files = [f for f in listdir(full_path) if isfile(join(full_path, f))]
print(f'files for {date}: {files}')

for i, f in enumerate(files):
    df = pd.read_csv( join(rel_path, date, f) )
    id = date + '_R' + str(i)
    sample = getType(f, sample_types)
    antenna = getType(f, antenna_types)
    
    # compile fields
    df['id'] = id
    df['sample'] = sample
    df['sigma_peak_mV'] = None
    df['sigma_noise_mV'] = None
    df['antenna'] = antenna
    
    # reorder columns and change names
    df.rename(columns={
        'Frequency (GHz)' : 'frequency_GHz', 
        'Peak (mV)' : 'peak_mV',
        'Noise (mV)' : 'noise_mV' 
    }, inplace=True)
    columns = ['id', 'frequency_GHz', 'peak_mV', 'sigma_peak_mV', 'noise_mV', 'sigma_noise_mV', 'sample', 'antenna']
    df = df[columns]

    name = id + '.txt'
    df.to_csv(  join(prj_dir, 'onlineRepo', 'labeldata', name),
                encoding='utf8', 
                index=False)

# 20-06-24
date = '20-06-24'
full_path = join(prj_dir, rel_path, date)

files = [f for f in listdir(full_path) if isfile(join(full_path, f))]
print(f'files for {date}: {files}')

for i, f in enumerate(files):
    df = pd.read_csv( join(rel_path, date, f) )
    id = date + '_R' + str(i)
    sample = getType(f, sample_types)
    antenna = getType(f, antenna_types)
    
    # compile fields
    df['id'] = id
    df['sample'] = sample
    df['antenna'] = antenna
    
    # reorder columns and change names
    df.rename(columns={
        'Frequency (GHz)' : 'frequency_GHz', 
        'Peak (mV)' : 'peak_mV',
        'Noise (mV)' : 'noise_mV',
        'SigNoise (mV)' : 'sigma_noise_mV',
        'SigPeak (mV)' : 'sigma_peak_mV',
    }, inplace=True)

    columns = ['id', 'frequency_GHz', 'peak_mV', 'sigma_peak_mV', 'noise_mV', 'sigma_noise_mV', 'sample', 'antenna']
    df = df[columns]

    name = id + '.txt'
    df.to_csv(  join(prj_dir, 'onlineRepo', 'labeldata', name),
                encoding='utf8', 
                index=False)

