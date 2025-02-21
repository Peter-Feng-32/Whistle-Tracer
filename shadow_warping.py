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
WINDOW_SIZE = 2048
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
}


def get_spectrogram(filename):
    try:
        sample_rate, samples = wavfile.read(filename)
        if samples.ndim > 1:  # Handle stereo files by selecting one channel
            samples = samples.T[0]
        if len(samples) < WINDOW_SIZE:  # Skip files that are too short
            print(f"File {filename} is too short. Skipping.")
            return None, None, None
        frequencies, times, spectrogram = signal.spectrogram(
            samples, sample_rate, window=WIN)
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
    spectrogram = spectrogram[150:250, 0:110]
    fact_ = 1.5
    mval = np.mean(spectrogram)
    sval = np.std(spectrogram)
    #=spectrogram[spectrogram > mval + fact_*sval] = mval + fact_*sval
    power_threshold = np.percentile(spectrogram.flatten(), 95.5)
    spectrogram[spectrogram < power_threshold] = 0
    
    spectrogram = filter_large_components(spectrogram, 5)
    return spectrogram

def create_shadow_warp(spectrogram, threshold = 0.9):
    rows, cols = spectrogram.shape
    
    x_warp = (spectrogram > threshold).any(axis = 0).astype(int)
    y_warp = (spectrogram > threshold).any(axis = 1).astype(int)
    warp = (x_warp, y_warp)
    return warp

    
levels = defaultdict(lambda: 0)
    
def classify(spectrogram, templates, diff_threshold = 0.2):
    best_label = "NULL"
    spectrogram = clean_template(spectrogram)
    shadow_warp = create_shadow_warp(spectrogram)

    maxscores = defaultdict(lambda: 0)
    for cls, template_warps in templates.items():
        for template_warp in template_warps:
            spec_warp_x, spec_warp_y = shadow_warp
            template_warp_x, template_warp_y = template_warp            
            distance = np.sum(np.abs(spec_warp_x - template_warp_x)) + np.sum(np.abs(spec_warp_y - template_warp_y))
            # Reshape to a 2D array (single row)
            maxscores[cls] = max(maxscores[cls], distance)
    scores = list(maxscores.items())
    scores = [(item[1], item[0]) for item in scores]
    scores.sort()
    print(scores)
    return scores[0][1]
            
def create_class_templates(classes, training_dir, num_files_per_class=5, test_params=None, display=False):
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
            shadow_warp = create_shadow_warp(spectrogram=spectrogram)
            templates[cls].append(shadow_warp)

    return templates

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

if __name__ == "__main__":
    classes = ['DEN', 'ROP', 'SCA', 'GRA', 'SAR']
    training_dir = "./2024DolphinDataset/training"
    testing_dir = "./2024DolphinDataset/testing"
    templates_dir = "./2024DolphinDataset/templates"
    
    print(os.path.exists("."))
    

    print("Loading templates\n")
    templates = create_class_templates(classes, templates_dir, num_files_per_class=5, display=False)
    

    print("Running classification experiment\n")
    y_true = []
    y_pred = []
    
    for cls in classes:
        label = class_mapping[cls]
        class_dir = os.path.join(testing_dir, label)
        if not os.path.exists(class_dir):
            print(f"Class directory {class_dir} does not exist. Skipping.")
            continue
        class_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        for i, filename in enumerate(class_files):
            filepath = os.path.join(class_dir, filename)
            _, _, spectrogram = get_spectrogram(filepath)
            if spectrogram is None:
                print(f"Skipping file {filename} due to invalid spectrogram.")
                continue
            
            filename_no_ext = filename[:-4]
            predicted_label = classify(spectrogram, templates)
            y_true.append(cls)
            y_pred.append(predicted_label)
            best_distance = 0
            
            if cls != predicted_label:
                print(i, filename, f"predicted: {predicted_label}, actual: {cls}, distance: {best_distance}")
                #display_spectrogram_with_array(original_spectrogram, modified_spectrogram, trace_result, cls, predicted_label)
                
    print(y_pred)
    print(levels)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=classes))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=classes))



