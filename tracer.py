from scipy.signal import stft
import numpy as np
import aifc
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from skimage.util import random_noise
from skimage import feature
import statistics


SAMPLE_RATE = 192000
FRAME_TIME_LENGTH = 2 #seconds
WINDOW_SIZE = 2048
WINDOW_TYPE = "hann"

LOWEST_FREQ = 5000 # 5000
HIGHEST_FREQ= 20000 # 20000
# Each freq bin represents approx 92 Hz

num_samples = SAMPLE_RATE * FRAME_TIME_LENGTH

sample_rate, samples = wavfile.read('ex.wav') # load the data
samples = samples.T[0] # Get first track
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, window=signal.get_window(window=WINDOW_TYPE, Nx=WINDOW_SIZE) )

spectrogram = 10 * np.log10(spectrogram) #(Freq, time)

spectrogram = -spectrogram

trace = [0]
curr_trace = 0
comp_range = 10

print(len(spectrogram))

LOWEST_FREQ_BIN = int(LOWEST_FREQ / ((SAMPLE_RATE / 2) / len(spectrogram)))
HIGHEST_FREQ_BIN = int(HIGHEST_FREQ / ((SAMPLE_RATE / 2) / len(spectrogram)))

print(np.amax(spectrogram[LOWEST_FREQ_BIN:HIGHEST_FREQ_BIN]))
print(np.amin(spectrogram[LOWEST_FREQ_BIN:HIGHEST_FREQ_BIN]))

POWER_THRESHOLD = 10

for i in range(1, len(times)):
    best_shift = -curr_trace
    best_diff = None
    for freq_shift in range(-comp_range, comp_range):
        diff = None
        for j in range(LOWEST_FREQ_BIN, HIGHEST_FREQ_BIN):
            if(spectrogram[j][i-1] > POWER_THRESHOLD and spectrogram[j + freq_shift][i] > POWER_THRESHOLD):
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
        

spectrogram[spectrogram <= POWER_THRESHOLD] = 0


f1 = plt.figure(1)
#plt.pcolormesh(times, frequencies[LOWEST_FREQ_BIN:HIGHEST_FREQ_BIN], spectrogram[LOWEST_FREQ_BIN:HIGHEST_FREQ_BIN], shading='auto')    
plt.plot(trace) #Shape: (Freq bins, time bins)
print(np.shape(spectrogram))
plt.title("Masked Results(comp 10 bins)")
plt.ylabel('Freq ')
plt.xlabel('Time')

plt.show()