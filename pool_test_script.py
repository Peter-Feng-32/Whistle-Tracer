from scipy.signal import stft, medfilt
import numpy as np
import os
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage import convolve
from dtw import accelerated_dtw
from itertools import product
from sklearn.metrics import confusion_matrix, classification_report
import librosa
import matplotlib.pyplot as plt
from skimage import feature
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing, binary_fill_holes, binary_opening
from collections import defaultdict
from statistics import median, mean
from scipy.ndimage import label as scipy_label, find_objects
from scipy.spatial.distance import euclidean
from skimage.filters import try_all_threshold, threshold_yen, hessian, meijering
from skimage.restoration import denoise_bilateral
from skimage.morphology import skeletonize
import copy
import scipy as sp
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
from scipy.signal import wiener
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

import cv2


SAMPLE_RATE = 192000
FRAME_TIME_LENGTH = 2  # seconds
WINDOW_SIZE = 4096
WINDOW_TYPE = "hann"
# Each freq bin represents approx 92 Hz
LOWEST_FREQ = 5000  # 5000
HIGHEST_FREQ = 20000  # 20000
WIN = signal.get_window(window=WINDOW_TYPE, Nx=WINDOW_SIZE)

# Updated mapping of class names to file labels
class_mapping = {
    'DEN': 'denise',
    'ROP': 'rope',
    'SCA': 'scarf',
    'GRA': 'grass',
    'SAR': 'sargassum',
    'NULL': 'noise'
}

class_parameters = {
    'DEN': {'power_threshold': 75, 'comparison_range': 2, 'distance_threshold': 100, 'window_size': 5, 'sigma': 3, 'low_threshold': 0.4, 'high_threshold': 0.7},
    'ROP': {'power_threshold': 80, 'comparison_range': 3, 'distance_threshold': 120, 'window_size': 5, 'sigma': 3, 'low_threshold': 0.4, 'high_threshold': 0.7},
    'SCA': {'power_threshold': 70, 'comparison_range': 2, 'distance_threshold': 110, 'window_size': 5, 'sigma': 3, 'low_threshold': 0.4, 'high_threshold': 0.7},
    'GRA': {'power_threshold': 85, 'comparison_range': 4, 'distance_threshold': 130, 'window_size': 5, 'sigma': 3, 'low_threshold': 0.4, 'high_threshold': 0.7},
    'SAR': {'power_threshold': 75, 'comparison_range': 3, 'distance_threshold': 115, 'window_size': 5, 'sigma': 3, 'low_threshold': 0.4, 'high_threshold': 0.7},
    'NULL': {'power_threshold': 75, 'comparison_range': 3, 'distance_threshold': 115, 'window_size': 5, 'sigma': 3, 'low_threshold': 0.4, 'high_threshold': 0.7},

}

def normalize_spectrogram(spectrogram):
    desired_mean=0,
    desired_stdev=1
    curr_stdev = np.std(spectrogram, axis=0)
    curr_mean = np.mean(spectrogram, axis=0)
    curr_stdev[curr_stdev == 0] = 1e-8

    # Normalize per column
    normalized = (spectrogram - curr_mean[np.newaxis, :]) / curr_stdev[np.newaxis, :]
    normalized = normalized * desired_stdev + desired_mean
    return normalized


def get_spectrogram(filename):
    try:
        sample_rate, samples = wavfile.read(filename)
        if samples.ndim > 1:  # Handle stereo files by selecting one channel
            samples = samples.T[0]
        if len(samples) < WINDOW_SIZE:  # Skip files that are too short
            print(f"File {filename} is too short. Skipping.")
            return None, None, None
        frequencies, times, spectrogram = signal.spectrogram(
            samples, sample_rate, window=WIN, nperseg=4096, noverlap=4096-960)
        spectrogram = 10 * np.log10(spectrogram + 1e-10)  # Avoid log(0)

        #spectrogram = (spectrogram - np.mean(spectrogram, axis=0)) / np.std(spectrogram, axis=0)
        return frequencies, times, spectrogram
        
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return None, None, None
    
def filter_large_components(array, size_threshold=10):
    # Label the connected components
    labeled_array, num_features = scipy_label(array)
    
    # Initialize an array to store the filtered result
    filtered_array = np.zeros_like(array)
    
    # Iterate through each connected component
    for i in range(1, num_features + 1):
        # Get the mask for the current component
        component_mask = (labeled_array == i)
        # Check the size of the current component
        if component_mask.sum() > size_threshold:
            # Keep the component if it exceeds the size threshold
            filtered_array[component_mask] = array[component_mask]
    
    return filtered_array
    
def keep_top_connected_components(array, top_n=5):
    # Label the connected components
    labeled_array, num_features = scipy_label(array)
    
    # Measure the size of each connected component
    component_sizes = [(i, (labeled_array == i).sum()) for i in range(1, num_features + 1)]
    
    # Sort components by size in descending order
    top_components = sorted(component_sizes, key=lambda x: x[1], reverse=True)[:top_n]
    top_labels = {component[0] for component in top_components}
    
    # Create a new array with only the top components
    filtered_array = np.where(np.isin(labeled_array, list(top_labels)), array, 0)
    
    return filtered_array
    
def clean_template(spectrogram):
    rows, cols = spectrogram.shape
    spectrogram = spectrogram[250:750, ]
    spectrogram = normalize_spectrogram(spectrogram)
    
    #print(rows, cols) #(1025, 111)
    spectrogram = spectrogram[50:200, ]
    fact_ = 1.5
    mval = np.mean(spectrogram)
    sval = np.std(spectrogram)
    #=spectrogram[spectrogram > mval + fact_*sval] = mval + fact_*sval
    power_threshold = np.percentile(spectrogram.flatten(), 95.5)
    spectrogram[spectrogram < power_threshold] = 0
    
    spectrogram = filter_large_components(spectrogram, 5)
    #spectrogram[spectrogram > 1] = 5

    #spectrogram = normalize_spectrogram(spectrogram)
    #display_spectrogram(spectrogram)
    return spectrogram

def create_gaussian_pyramid(spectrogram):
    rows, cols = spectrogram.shape
    
    layer_1 = cv2.pyrDown(spectrogram, dstsize=(cols // 2, rows // 2))
    layer_2 = cv2.pyrDown(layer_1, dstsize=(cols // 4, rows // 4))
    pyramid = (spectrogram, layer_1, layer_2)
    return pyramid

    
levels = defaultdict(lambda: 0)
classified_scores = []
    
def classify(spectrogram, templates, filename, diff_threshold = 0.2):
    best_label = "NULL"
    
    num_layers = 3
    spec_pyramid = create_gaussian_pyramid(spectrogram)
    max_loc = None
    max_val = None
    
    for i in range(num_layers - 1, -1, -1):
        maxscores = defaultdict(lambda: (0, (-1, -1)))
        for cls, pyramids in templates.items():
            
            # Apply template matching using normalized cross-correlation
            for pyramid in pyramids:                
                #print(spec_pyramid[i].shape, pyramid[i].shape)
                result = cv2.matchTemplate(np.float32(spec_pyramid[i]), np.float32(pyramid[i]), cv2.TM_CCOEFF_NORMED)
                # Find the best match location and score
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                #print(cls, max_val, max_loc)

                maxscores[cls] = max(maxscores[cls], (max_val, max_loc, i))
                if(i == 0):
                    display_spectrogram(pyramid[i])
                    display_spectrogram(spec_pyramid[i])
                    display_spectrogram(result)
        scores = list(maxscores.items())
        scores = [(item[1], item[0]) for item in scores]
        scores.sort()
        if scores[-1][0][0] - scores[-2][0][0] >= diff_threshold:
            print(scores)
            levels[i] += 1
            classified_scores.append((scores[-1][0], filename))
            return scores[-1][1]
    levels[55] += 1
    print("level 0", scores)
    classified_scores.append((scores[-1][0], filename))
    return "NULL"
            
def create_class_templates(classes, training_dir, num_files_per_class=1, test_params=None, display=False):
    templates = {cls: [] for cls in classes}

    for cls in classes:
        label = class_mapping[cls]
        params = class_parameters[cls]

        class_dir = os.path.join(training_dir, label)
        if not os.path.exists(class_dir):
            print(f"Class directory {class_dir} does not exist. Skipping.")
            continue

        class_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]

        for i, filename in enumerate(class_files[:num_files_per_class]):
            filepath = os.path.join(class_dir, filename)
            _, _, spectrogram = get_spectrogram(filepath)
            if spectrogram is None:
                print(f"Skipping file {filename} due to invalid spectrogram.")
                continue
            #trace_result, modified_spectrogram = trace(spectrogram, power_threshold, comparison_range, pcen_alpha, pcen_delta, pcen_r)
            spectrogram = clean_template(spectrogram)
            gaussian_pyramid = create_gaussian_pyramid(spectrogram=spectrogram)
            templates[cls].append(gaussian_pyramid)

    return templates

def output_templates(templates):
    #print(templates)
    for cls, template_list in templates.items():
        for index, template_pyramid in enumerate(template_list):
            for img_index, image in enumerate(template_pyramid):
                np.savetxt(f"saved_templates/{class_mapping[cls]}/{index}/{img_index}.txt", np.transpose(image))
                print(cls, image.shape)
                
def output_spectrogram(name, label, spectrogram):
    np.savetxt(f"spectrograms/{label}/{name}.txt", spectrogram)
    

def display_spectrogram_with_array(original_spectrogram: np.ndarray, modified_spectrogram: np.ndarray, array: np.ndarray, 
                                   spectrogram_title: str = "Spectrogram", 
                                   array_title: str = "1D Array"):
    """
    Displays a 2D numpy array (spectrogram) side by side with a 1D numpy array.

    Args:
        spectrogram (np.ndarray): 2D array representing the spectrogram.
        array (np.ndarray): 1D array to display alongside the spectrogram.
        spectrogram_title (str): Title for the spectrogram plot.
        array_title (str): Title for the 1D array plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 3, 3]})
    
    # Plot the spectrogram
    ax1 = axes[0]
    im = ax1.imshow(original_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_title(spectrogram_title)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency')
    plt.colorbar(im, ax=ax1, orientation='vertical', label='Intensity')
    
    # Plot the spectrogram
    ax2 = axes[1]
    im = ax2.imshow(modified_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    ax2.set_title(spectrogram_title)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Frequency')
    plt.colorbar(im, ax=ax2, orientation='vertical', label='Intensity')
    
    # Plot the 1D array horizontally
    ax3 = axes[2]
    ax3.plot(array, label=array_title, color='orange')
    ax3.set_title(array_title)
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Value')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

def display_spectrogram(data):
    """Displays the spectrogram as an image."""
    plt.figure(figsize=(10, 5))
    plt.imshow(data, aspect="auto", origin="lower", cmap="inferno")  # Adjust colormap if needed
    plt.colorbar(label="Intensity")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title("Spectrogram")
    plt.show()

def load_spectrogram(filename, delimiter=" "):
    """Reads a spectrogram from a text file, removes trailing commas, and converts it into a NumPy array."""
    with open(filename, "r") as file:
        cleaned_lines = [line.replace(",", "").strip() for line in file]  # Remove commas
    data = np.loadtxt(cleaned_lines, delimiter=delimiter)
    return data

if __name__ == "__main__":
    classes = ['DEN', 'ROP', 'SCA', 'GRA', 'SAR', "NULL"]
    training_dir = "./2024DolphinDataset/training"
    testing_dir = "./2024DolphinDataset/testing"
    templates_dir = "./2024DolphinDataset/templates"
    
    filepath = "./FailingWhistle96khz.wav"
    templates = create_class_templates(classes, templates_dir, num_files_per_class=1, display=False)

    _, _, spectrogram = get_spectrogram(filepath)
    spectrogram = spectrogram[250:750]
    spectrogram = normalize_spectrogram(spectrogram)
    display_spectrogram(spectrogram)
    predicted_label = classify(spectrogram, templates, diff_threshold=0.1, filename=filepath)
    print (predicted_label)


