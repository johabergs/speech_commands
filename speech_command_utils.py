import os
import numpy as np
from scipy.io import wavfile
import librosa
import math
import keras.backend as K


def pad_audio(samples,L):
    """
    
    """
    if len(samples) >= L:
        return samples
    else:
        diff = L - len(samples)
        samp = np.zeros((L,))
        ind1 = math.floor(diff/2)        
        ind2 = ind1 + len(samples)
        samp[ind1:ind2] = samples
        return samp
        



def log_uniform_sample(sample_range):
    """
    Sample from log-uniform distribution.

    Arguments:
        sample_range: list with two positive elements. 
        sample_range[0] is the minimum of the support and sample_range[1] is the maximum

    Returns:
        Log-uniformly distributed sample between sample_range[0] and sample_range[1]
    """
    log_min = np.log10(sample_range[0])
    log_max = np.log10(sample_range[1])
    u = np.random.rand()*(log_max-log_min) + log_min
    return np.power(10.0,u)
       

def list_and_label_using_folders(path, ext='wav'):
    
    file_names, labels = [],[]
    for (dirpath, dirnames, names) in os.walk(path):
        for name in names:
            if name.lower().endswith('.' + ext):    
                file_names.append(os.path.join(dirpath, name))
                labels.append(os.path.basename(dirpath))
            
    return file_names, labels


def filter_speech_files(known_words, file_names, file_labels, unknown_fraction):

    filtered_files, filtered_labels = [], []
    
    for i, label in enumerate(file_labels):       
        if label in known_words:
            filtered_files.append(file_names[i])
            filtered_labels.append(label)
        elif label == '_background_noise_':
            pass
        elif np.random.rand()<unknown_fraction:
            filtered_files.append(file_names[i])
            filtered_labels.append('unknown')
    
    return filtered_files, filtered_labels


    

# VOL_RANGE NOT IMPLEMENTED
def create_background_samples(bkg_files,num_examples,
                              sample_time=1,hop_time=0.01,win_time=0.025,n_mels=40,vol_range=[1e-4,1],fs=16000, prob_clip=0.5):
    ''' 
    Generates examples of background noise, using a similar number of examples from each background file
    Inputs:
     - bkg_files: paths to files with background noise  
     - num_examples: number of examples to extract
     - sample_time: the total sample time in seconds
     - hop_time: hop time for mel-spectrogram
     - win_time: window time for mel-spectrogram
     - n_mels: number of mel-frequencies
     - vol_range: volume range to randomly rescale the audio files with using a log-uniform distribution 
     - fs: Sampling rate of wavs
     
    Returns:
    - bkg_samples: extracted background waveforms - int16 ndarray with shape (num_examples,int(fs*sample_time)) 
    - bkg_spect: mel-spectrogram of background waveforms - float32 ndarray 

    '''
    bkg_spect = None
    
    
    num_samples = fs*sample_time

    
    num_bgk_files = len(bkg_files)
    num_examples_per_file, _ = np.histogram(range(num_examples), bins=num_bgk_files)

    # Preallocate arrays of the correct shape
    # Input should have shape (num_samples, num_channels, height, width)
    spectrogram_shape = (num_examples, n_mels, int(sample_time/hop_time)+1, 1)
    bkg_spect = np.zeros(spectrogram_shape, dtype=np.float32)
    

    # Loop though all the files and take an equal number of examples from each file
    example_idx = 0
    for k, fname in enumerate(bkg_files):
        
        # Read the whole background file. The background files can have different sampling rates
        _, full_wave = wavfile.read(fname)
        
        # Extract num_examples_per_file[k] clips of length sample_time, starting at a random position
        for i in range(num_examples_per_file[k]):
            

            ind = np.random.randint(full_wave.size - num_samples)
            x = full_wave[ind: ind + num_samples]
            sample = x*log_uniform_sample(vol_range)

            if np.random.rand() < prob_clip:
                ind1 = math.floor(np.random.rand()*num_samples)
                ind2 = math.floor(np.random.rand()*num_samples)
                inds = np.sort([ind1,ind2])
                sample[inds[0]:inds[1]] = x[inds[0]:inds[1]]*log_uniform_sample(vol_range)

                

            # There returned waveform and the returned spectrogram are the same, except that they are rescaled by 
            # different random volume factors (within the same range). 
            # This should be good - or should we use the same rescaling factor
            
            bkg_spect[example_idx,:,:,0] = librosa.feature.melspectrogram(sample, 
                                                           sr=fs, 
                                                           n_mels=n_mels,
                                                           n_fft=int(fs*win_time),
                                                           hop_length=int(fs*hop_time))
                                                           
            example_idx += 1
            if(example_idx % 1000 == 0):
                print('Processed ' + str(example_idx) + ' background files out of ' + str(bkg_spect.shape[0]) )
            
    return bkg_spect


#from speech_command_functions.py import test_data_generator
from IPython.core.debugger import Tracer




def compute_speech_spectrograms(file_names,hop_time=0.01,win_time=0.025,n_mels=40):
    '''
    TODO: use background files to mix in
    
    '''
    sample_time = 1
    fs = 16000

    spectrogram_shape = (len(file_names), n_mels, int(sample_time/hop_time)+1, 1)
    speech_spect = np.zeros(spectrogram_shape, dtype=np.float32)
    
    for k, fname in enumerate(file_names):
        
        # Spectrogram calculation starts here
        _, wave = wavfile.read(fname)
        wave = pad_audio(wave,fs*sample_time) # How make this more general?
        spect  = librosa.feature.melspectrogram(wave, 
                                                sr=fs, 
                                                n_mels=n_mels,
                                                n_fft=int(fs*win_time),
                                                hop_length=int(fs*hop_time))              
        speech_spect[k,:,:,0] = spect
        # Spectrogram calculation ends here
        
        if(k % 1000 == 0):
            print('Processed ' + str(k) + ' speech files out of ' + str(len(file_names)) )
        
    return speech_spect
    
    
def test_data_generator(test_data_path,batch_size=1024,ext='wav',
                        sample_time=1,hop_time=0.01,win_time=0.025,n_mels=40):
    fpaths = []
    for (dirpath, dirnames, names) in os.walk(test_data_path):
        for name in names:
             if name.lower().endswith('.' + ext):    
                fpaths.append(os.path.join(dirpath, name))

    spectrogram_shape = (batch_size,n_mels,int(sample_time/hop_time) +1,1)
    
    i = 0
    for path in fpaths:
        if i == 0:
            fnames = [] 
            spectrograms = np.zeros(spectrogram_shape, dtype=np.float32)
        i += 1
        
        # Spectrogram calculation starts here
        fs, wave = wavfile.read(path)
        wave = pad_audio(wave, fs*sample_time)      
        spect  = librosa.feature.melspectrogram(wave, 
                                                sr=fs, 
                                                n_mels=n_mels,
                                                n_fft=int(fs*win_time),
                                                hop_length=int(fs*hop_time))
        
        #spect = np.log10(spect + spec_thr)
        spectrograms[i-1,:,:,0] = spect
        # Spectrogram calculation ends here

        fnames.append(path.split('\\')[-1])

        if i == batch_size:
            i = 0
            yield spectrograms, fnames
            
    if i>0:
        spectrograms = spectrograms[:i,:,:,:]
        yield spectrograms, fnames
        
   
def label_speech_test_data(model,test_data_path,spec_thr,label_index,sub_file='submission',test_batch_size=1024,output_acts=None):

    files = []
    probs = []
    activations = []
    test_generator = test_data_generator(test_data_path,batch_size=test_batch_size)


    # For calculating layer activations
    if output_acts is not None: 
        outputs    = [model.layers[output_acts].output]           # layer outputs
        comp_graph = [K.function([model.input]+ [K.learning_phase()], [output]) for output in outputs]  # evaluation functions


    print('Predicting on test files...')
    print('Predicted: ', end='')
    for spects,fnames in test_generator:
        # Load spectrograms for batch
        spects = np.log10(spects + spec_thr)

        # Calculate predicted probabilities
        pred_probs = model.predict(spects, batch_size=test_batch_size)
        probs.append(pred_probs)

        # Caluclate activations of chosen layer
        # Can be done more efficiently - calculations are now independent!
        # BUT model.predict gives different answer than activations of final layers
        if output_acts is not None: 
            acts = [op([spects, 1.]) for op in comp_graph][0][0]
            activations.append(acts)

        files.extend(fnames)    

        print(str(len(files)), end=', ')


    probs = np.concatenate(probs,axis=0)
    if output_acts is not None: 
        activations = np.concatenate(activations,axis=0)
    pred_idx = np.argmax(probs, axis=1)
    labels = [label_index[p] for p in pred_idx]

    print('\nWriting submission results to file...')
    with open(sub_file+'.csv','w') as sfile:
        sfile.write('fname,label\n')
        for i,lab in enumerate(labels):
            sfile.write(files[i] + ',' + lab + '\n')      
    print('Done. Written ' + str(i) + ' results')    

    return files, labels, probs, activations




def step_decay(epoch,initial_lrate=0.001,drop=0.1,epochs_drop=[10]):
    num_drops = sum(e <= epoch for e in epochs_drop)
    lrate = initial_lrate * (drop**num_drops)
    print('Epoch: ' + str(epoch) + ', Learning rate: ' + str(lrate) + ', Num drops: ' + str(num_drops))
    return lrate

