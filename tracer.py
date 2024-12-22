from scipy.signal import stft
import numpy as np
import aifc
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from skimage.util import random_noise
from skimage import feature
import statistics
from scipy.ndimage import label
from dtw import *
from itertools import product
import math

SAMPLE_RATE = 192000
FRAME_TIME_LENGTH = 2 #seconds
WINDOW_SIZE = 2048
WINDOW_TYPE = "hann"
# Each freq bin represents approx 92 Hz
LOWEST_FREQ = 5000 # 5000
HIGHEST_FREQ= 20000 # 20000
WIN = signal.get_window(window=WINDOW_TYPE, Nx=WINDOW_SIZE) 
POWER_THRESHOLD = 10

#Power Threshold Percentile, Inversing, Filtering, Supression Percentile, Connected Component Number, Comparison Range)
parameter_options = [
    (75, 90),
    (True, False),
    (True, False),
    (75, 90),
    (750, 1500),
    (2, 3)
]

def generate_parameter_combinations():
    return list(product(*parameter_options))

def get_spectrogram(filename):
    num_samples = SAMPLE_RATE * FRAME_TIME_LENGTH
    sample_rate, samples = wavfile.read(filename)
    samples = samples.T[0] 
    frequencies, times, spectrogram = signal.spectrogram(
        samples, sample_rate, window=WIN)
    spectrogram = 10 * np.log10(spectrogram) 
    return frequencies, times, spectrogram

frequencies, times, spectrogram = get_spectrogram('ex.wav')
LOWEST_FREQ_BIN = int(LOWEST_FREQ / ((SAMPLE_RATE / 2) / len(spectrogram)))
HIGHEST_FREQ_BIN = int(HIGHEST_FREQ / ((SAMPLE_RATE / 2) / len(spectrogram)))

#Spectrogram is dimensioned as freq, time
def trace(spectrogram, power_threshold, comparison_range):
    trace = [0]
    curr_trace = 0
    power_threshold = np.percentile(spectrogram.flatten(), power_threshold)
    for i in range(1, len(spectrogram[0]) - 1):
        best_shift = -curr_trace
        best_diff = None
        for freq_shift in range(-comparison_range, comparison_range):
            diff = None
            for j in range(0, len(spectrogram)):                    
                if j + freq_shift in range(0, len(spectrogram) - 1) and spectrogram[j][i-1] > power_threshold and spectrogram[j + freq_shift][i] > power_threshold:
                    if diff is None:
                        diff = 0
                    diff += abs(spectrogram[j][i-1] - spectrogram[j + freq_shift][i])
            if diff is None:
                continue
            if not best_diff or diff < best_diff:
                best_diff = diff
                best_shift = freq_shift
        
        curr_trace = curr_trace + best_shift
        trace.append(curr_trace)
    return trace

def get_filter():
    filter = np.ones((40, 2))
    filter[0:10] = -1
    filter[30:40] = -1
    return filter 

def filter_spectrogram(spectrogram, percentile, connected_component_size):
    filter = get_filter()
    filtered = signal.convolve(spectrogram, filter, 'same')[LOWEST_FREQ_BIN:HIGHEST_FREQ_BIN]
    supression_percentile = np.percentile(filtered.flatten(), percentile)
    suppressed = filtered
    suppressed[suppressed < supression_percentile] = 0
    suppressed[suppressed > supression_percentile] = 1
    structure = np.ones((3, 3), dtype=np.int32)  
    labeled, ncomponents = label(suppressed, structure)
    for i in range(1, ncomponents + 1):
        if len(labeled[labeled == i]) < connected_component_size:
            labeled[labeled == i] = 0
    smooth = np.ones((2,2)) / (2 * 2)
    smoothed = signal.convolve(labeled, smooth)
    return smoothed

def zscore_normalize(trace):
    mean = np.mean(trace)
    stdev = np.std(trace)
    if(stdev == 0):
        return trace - mean
    return (trace - mean) / stdev

def output_traces(input_file, output_dirname):
    frequencies, times, orig_spectrogram = get_spectrogram("./audio/" + input_file)

    for parameters in generate_parameter_combinations():
        print(parameters)
        spectrogram = orig_spectrogram
        power_threshold, is_inversing, is_filtering, supression_percentile, connected_component_size, comparison_range = parameters
        if is_inversing:
            spectrogram = spectrogram * -1
        if is_filtering:
            spectrogram = filter_spectrogram(spectrogram, supression_percentile, connected_component_size)
        else:
            spectrogram = spectrogram[LOWEST_FREQ_BIN:HIGHEST_FREQ_BIN]
        traced = trace(spectrogram, power_threshold, comparison_range)
        traced = zscore_normalize(traced)
        if np.count_nonzero(traced) == 0:
            continue

        config_name = f"/trace_{power_threshold}_{is_inversing}_{is_filtering}_{supression_percentile}_{connected_component_size}_{comparison_range}"
        np.savetxt(output_dirname + config_name + ".csv", traced, delimiter=',')
        plt.plot(traced)
        plt.gca().invert_yaxis()
        plt.title(config_name)
        plt.xlabel("Time bins")
        plt.ylabel("Freq bins")
        plt.savefig(output_dirname + config_name + ".png")
        plt.close()
        
                
#output_traces("06211501 172402 2145 SW FM mimic.wav", "traces/mimics/2145")
frequencies, times, orig_spectrogram = get_spectrogram("./audio/" + "06211501 172402 1627 SC mimic HF FM_.wav")
orig_spectrogram = orig_spectrogram[LOWEST_FREQ_BIN:HIGHEST_FREQ_BIN]
traced = trace(orig_spectrogram, 93, 7) # 3 doesn't work.
power_threshold = np.percentile(orig_spectrogram.flatten(), 95)
orig_spectrogram[orig_spectrogram < -43] = 0
#print(power_threshold)
#plt.imshow(orig_spectrogram)
plt.plot(traced)
plt.gca().invert_yaxis()
plt.show()