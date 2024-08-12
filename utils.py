import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.optimize
import scipy.signal 

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
    if('id'  in df.columns):
        return ' '.join([str(df['id'].iloc[0]), str(df['sample'].iloc[0]),  str(df['antenna'].iloc[0])] )
    if('process'  in df.columns):
        return ' '.join([str(df['sample'].iloc[0]), str(df['process'].iloc[0]), str(df['antenna'].iloc[0])] )
    return ' '.join([str(df['sample'].iloc[0]),  str(df['antenna'].iloc[0])] )
         
def getValidKwargs(mykwargs, mylist):
    return {k: v for k, v in mykwargs.items() if k in mylist}

def formatPlot(axs : list):
    
    try:
        for ax in axs:
            ax.set(xlabel='Frequency (GHz)',
                    ylabel = 'Tension (mV)')
            ax.legend(fontsize = 'small')
    except:
        axs.set(xlabel='Frequency (GHz)',
                ylabel = 'Tension (mV)')
        axs.legend(fontsize = 'small')


def plotThis(subset : pd.DataFrame, ax, **kwargs):
    ''' 
    Plots relevant columns of the dataset on the given ax.
    Also passes to the ax eventual kwargs specified.
    '''
    label_root = genLabel(subset)
    
    # unpack all the kwargs in thei respective 
    # dicts, you cannot use axArgs in plt.plot()
    pltKeys = ['alpha']
    axKeys = ['xlim', 'ylim', 'xlabel', 'ylabel', 'title']
    pltArgs = getValidKwargs(kwargs, pltKeys)
    axArgs = getValidKwargs(kwargs, axKeys)
    
    for mylabel in ['peak_mV', 'noise_mV']:
            # handle non existing cols and nan values
            if mylabel not in subset.columns:
                    continue
            if subset[mylabel].isnull().values.any():
                    continue
            
            c = None
            if mylabel == 'noise_mV':
                # Get the color of the last line
                c = ax.get_lines()[-1].get_color()

           
            # errors? errorbar, standard plot otherwise      
            if 'sigma_' + mylabel in subset.columns: 
                if not subset['sigma_' + mylabel].isnull().values.any():
                    ax.errorbar(subset['frequency_GHz'], subset[mylabel], 
                                yerr = subset['sigma_' + mylabel],
                                label = ' '.join([label_root , mylabel]),
                                color = c,
                                **pltArgs)
                    # addidtional kwargs passed?
                    # filter kwargs to pass only valid ones to ax.set
                    ax.set(**axArgs)
                    continue
                        
            ax.plot(subset['frequency_GHz'], 
                    subset[mylabel],
                    label = ' '.join([label_root , mylabel]),
                    color = c,
                    **pltArgs)
                   
            
            # addidtional kwargs passed?
            # filter kwargs to pass only valid ones to ax.set
            ax.set(**axArgs)
                                                            

def plotByID(subset : pd.DataFrame, title : str  = 'GraphTitle'):
        '''
                Draws the entire dataframe grouping measurements by id.
                The result is a graph with one line per measurement run.
        '''
        
        fig, ax = plt.subplots()
        subset.groupby(subset['id']).apply(plotThis, ax)

        fig.suptitle(title)
        ax.grid(True)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Tension (mV)')
        ax.legend()

def correctGainPipe(subset : pd.DataFrame, coeff : np.ndarray):
    '''
        Scales every relevant field by the polinomial specyfied by the 
        coeff
    '''
    outset = subset.copy()
    for f in ['peak_mV', 'sigma_peak_mV', 'noise_mV', 'sigma_noise_mV']:
        if f not in outset.columns:
            continue
        outset.loc[:, f] = subset.loc[:, f] / np.polyval(coeff, subset.loc[:, 'frequency_GHz'])
    return outset

def denoise(x : pd.Series):
    '''
        Removes the noise from the peak for each row of the dataset.
        DROPS NOISE AND SIGNOISE COLUMNS
    '''
    x['peak_mV'] = x['peak_mV'] - x['noise_mV']
    x['sigma_peak_mV'] = np.sqrt(np.sum(np.square(x[['sigma_peak_mV', 'sigma_noise_mV']])))
    x = x.drop(['noise_mV','sigma_noise_mV'])
    return x

def denoisePipe(df : pd.DataFrame):
    return df.apply(denoise, axis=1)

def getResampledFreq(df : pd.DataFrame, by = 'id'):
    '''
    Gets the array of frequencies that all the groups have. 
    This is useful when trying to fit a model and one wants 
    to reduce all the grouping to a single group with this 
    specific frequency array common to all groups.
    '''
    grouped = df.groupby(by)
    first_key = list(grouped.groups.keys())[0]  # Get the first key
    return grouped.get_group(first_key)[k.freq]

def resampleGroup(group : {pd.DataFrame, dict}, newlen : int,  columns : list = []):
    '''
    Resamples dataframes or dicts with the desired shape. 
    Columns is only necessary when using a dict.
    Use >
        passing a df you need to provide the columns that will be reshaped. Everything 
        else is DROPPED to fit the new shape.
    
        passing a dict the dict is expected to contain an array for each key in it.
        The values in the dict (expexted to be arrays) are analogous to the colums 
        of the df.
    
    '''
    if isinstance(group, pd.DataFrame):
        if len(group.index) == newlen:
            return group
        # create a new dataframe with less colums,
        # this copyes the head of the old dataframe
        # and then updates the new values.
        newgroup = group.head(newlen)
        for c in columns:
            if c not in group.columns:
                continue
            # resample every relevant column
            newgroup[c] = scipy.signal.resample(group[c], newlen)
        
        # also resample the frequencies
        newgroup[k.freq] = np.linspace( group[k.freq].min(),
                                        group[k.freq].max(),
                                        newlen)
        
        if k.sig_pek in newgroup:
            newgroup = newgroup.drop(k.sig_pek, axis=1)
        
        return newgroup
    
    if isinstance(group, dict):
        # this dict is expected to contain the arrays 
        # of the data, so group.values is a list of 
        # arrays
        for key, arr in group.items(): 
            if len(arr) == newlen:
                continue
            group[key] = scipy.signal.resample(arr, newlen)
        return group
            

def resamplePipe(df : pd.DataFrame, by='id'):
    '''
        divides the dataframe into groups based on the id 
        and then resamples everything to the smallest id group.
        This is useful to merge all the signals.
    '''
    # get the groups divided by id 
    grouped = df.groupby(by)

    # get the minimun of the lenghts of the groups
    minlen = np.min([len(g.index) for name, g in grouped])
    
    # resample the appropriate columns
    # to the minimun lenght
    columns = [k.pek, k.nse]
        
    # this hack is necessary to assure the format of the data is correct,
    # the apply function f*** up the id column... now it was a multiindex
    # dropping the first level solves the problem.
    resampledGrouped = grouped.apply(resampleGroup, minlen, columns)
    resampledGrouped = resampledGrouped.droplevel(0)
    
    return  resampledGrouped
def FFTFilter(fft_signal: np.ndarray, percAmp: float, highPass: int):
    ''' 
        Removes every frequency below percAmp * MaxAmplitude
        and the frequencies that are highPass away from the mean 
        point of the fft
    '''
    # remove a part of the signal
    fft_signal = [ i if abs(i) > np.abs(fft_signal).max() * percAmp  else 0 for i in fft_signal]
    
    # high pass filter
    mid = int(len(fft_signal) / 2)
    highPass = int(highPass) # assure is int
    for i in range(mid - highPass , mid + highPass): 
        fft_signal[i] = 0
    return fft_signal

def savgovFilterPipe(data: pd.DataFrame, window_lenght: int, polyorder: int):
    # create a data structure to hold the processed signals
    processed_signals = {}
    
    for name, group in data.groupby('id'):
        processed_signals[name] = scipy.signal.savgol_filter(group[k.pek], window_lenght, polyorder)
        
    def updateDF(df, values):
        id = df['id'][0]
        df[k.pek] = values[id]
        return df
    
    data = data.groupby('id').apply(updateDF, processed_signals)     
    # drop error column and additional index left over from groupby
    if k.sig_pek in data.columns:
        return data.drop(k.sig_pek, axis=1).droplevel(0)
    return data.droplevel(0)

def FFTFilterPipe(data: pd.DataFrame, percAmp: float, highPass: int, plotFFT = False):
    '''
    This pipeline does the following things:
        -   for each group of data filters the fft  high freq 
            and all the frequencies below a certain amplitude.
        -   plots all the important data to debug the processing
            and a final graph to see the changes
    '''
    
    # create a data structure to hold the processed signals
    processed_signals = {}
    
    # for each group do the relative denoising and plotting
    for name, group in data.groupby('id'):
        
        # try to remove large oscillations of the data
        pc = np.polyfit(group[k.freq], group[k.pek], 3)
        
        # compute the fft on the difference between the polyfit and the data
        signal = group[k.pek] - np.polyval(pc, group[k.freq])
        fft_signal = np.fft.fft(signal)
        
        # filter the fft
        filtered_fft_signal = FFTFilter(fft_signal, percAmp, highPass)
        
        # go back to the direct space of the signal
        ifft = np.fft.ifft(filtered_fft_signal)
        
        # save it
        processed_signals[name] = [group[k.freq].values, np.polyval(pc, group[k.freq]) + np.real(ifft)]
        
        # plot the data
        if plotFFT:
            fig, axs = plt.subplots(2,2, figsize=(15,10))
            group.plot(x=k.freq, y=k.pek, yerr=k.sig_pek, label=name, alpha=.5, ax=axs[0,0])
            # plot the polyfit
            axs[0,0].plot(group[k.freq], np.polyval(pc, group[k.freq]), label='Approx', c='blue')
            # plot the difference
            axs[0,0].plot(group[k.freq], group[k.pek] - np.polyval(pc, group[k.freq]), 
                    label='Shifts', c='red')
            axs[0,1].plot(np.real(fft_signal), label='fft transform')
            axs[0,1].plot(np.real(filtered_fft_signal), label='filtered fft transform')

            # plot the main data and the denoised version
            axs[1,0].plot(group[k.freq], signal,        label='signal')
            axs[1,0].plot(group[k.freq], np.real(ifft), label='denoised signal')
            axs[1,1].plot(group[k.freq], group[k.pek],  label='signal')
            axs[1,1].plot(group[k.freq], np.polyval(pc, group[k.freq]) +  np.real(ifft), label='denoised signal')

            axs[0,0].set(title='Raw signal, removed instrument noise',
                        xlabel='Frequency GHz', ylabel='Voltage (mV)')
            axs[0,1].set(title='FFT of the signal',
                        xlabel='oneover frequency', ylabel='Partial amplitude')
            axs[1,0].set(title='Filtered shifts',
                        xlabel='Frequency GHz', ylabel='Voltage (mV)')
            axs[1,1].set(title='Filtered signal (shifts + polyfit)',
                        xlabel='Frequency GHz', ylabel='Voltage (mV)')
    if plotFFT:
        # make legends and title
        for ax in axs.flatten():
            ax.legend()
        fig.suptitle(f'Filtering {name}', fontsize = 16)

        axArgs = {
            'ylim' : [0,5],
            'xlabel' : 'Frequency (GHz)',
            'ylabel' : 'Tension (mV)'
        }
        
        # make a comparison between before and after
        fig, axs = plt.subplots(1,2, figsize = (20,5))
        fig.suptitle('Comparison between raw and filtered signals.')
        axs = axs.flatten()
        
        for key, pair in processed_signals.items():
            axs[1].plot(pair[0], pair[1], label=key)
        axs[1].set(**axArgs)
        
        data.groupby('id').apply(plotThis, axs[0], **axArgs)
        
        for ax in axs:
            ax.legend()
            
    # update the data in the dataframe
    def updateDF(df, values):
        
        id = df['id'][0]
        df[k.pek] = values[id][1]
        return df
    
    data = data.groupby('id').apply(updateDF, processed_signals)     
    # drop error column and additional index left over from groupby
    return data.drop(k.sig_pek, axis=1).droplevel(0)

# MODELS: each model should return a matrix with two rows, the first are the values and 
# the second are thei uncertainties.

def weightmean(df : pd.DataFrame, weights : {list, np.ndarray}):
        ''' The error is calculated with the inferred std
        and then by using variance propagation law. 
        This function fits only the Peak column of the dataset'''
     
        # unpack data into an array
        
        groups = df.groupby('id').apply(lambda x : x[k.pek].values)
        
        x = groups.to_numpy()
        y = (x * weights).sum(axis=0)

        # estimate the error from the samples
        xerr = np.std(x, axis=0, ddof=1)
        yerr = np.sqrt(np.square(weights*np.column_stack([xerr] * len(weights))).sum(axis=1))
 
        return y, yerr
    
def negcorr(weights, dataset):
    # here the dataset is fixed while the 
    # weights can change
    model = weightmean(dataset, weights)
    return -np.square(corrModelToData(dataset, model)).sum()

def corrModelToData(dataset : pd.DataFrame, model : {tuple, list, pd.DataFrame}):
    '''
    returns the last row of the correlation matrix between the model 
    and the data, each element is the correlation between the model
    and a specific run of the data.
    '''
    # get only the pek values in a np format
    data = np.stack(dataset.groupby('id').apply(lambda x : (x[k.pek].values)))
    if isinstance(model, (list, tuple)):
        data = np.append(data, [model[0]], axis=0)
    if isinstance(model, pd.DataFrame):
        data = np.append(data, [model[k.pek]], axis=0)
    return np.corrcoef(data)[-1, :]

def processPipe(dataset : pd.DataFrame, by='antenna', proc_name ='raw'):
    '''
    Returns a dataset when the model WEIGHTMEAN is appyed to the data,
    some standard output is performed.
    '''
    processed_data = pd.DataFrame()
    correlations = {}
    for group_name, group in dataset.groupby(by):
        # make the data homogeneous
        group = resamplePipe(group)

        # how many id do we get?
        samples = len(group['id'].unique())
        print(f'fitting group {group_name} with {samples} samples')
        
        # create an array of weights that sum to one
        # and minimize the model
        weights = np.array([1] * samples) / samples
        res = scipy.optimize.minimize(negcorr, weights, group)
        np.set_printoptions(precision=3)
        print('minimization status: ', res['message'])
        print('nomal minimization weights: ', res['x'])
        print('correlation with the data: ', corrModelToData(group, weightmean(group, res['x'])))
        
        # save the results
        newgroup = pd.DataFrame({
            'sample' : group['sample'].iloc[0],
            'process' : proc_name,
            'antenna' : group['antenna'].iloc[0],
            k.freq : getResampledFreq(group),
            k.pek   : weightmean(group, res['x'])[0],
            k.sig_pek : weightmean(group, res['x'])[1]
            })
        
        correlations[group_name] = corrModelToData(group, weightmean(group, res['x']))[:-1]
        processed_data = pd.concat([processed_data, newgroup])
    return processed_data

if __name__ == 'main':
    pass