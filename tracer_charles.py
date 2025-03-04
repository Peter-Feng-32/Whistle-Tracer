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

denoised_image = wiener(image)
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

from scipy.ndimage import uniform_filter

def guided_filter(I, p, r, eps):
    """
    Fast Guided Filter Implementation.

    Parameters:
    - I: Guidance image (grayscale, float32)
    - p: Input image to be filtered
    - r: Radius of local window
    - eps: Regularization parameter

    Returns:
    - Filtered image
    """
    mean_I = uniform_filter(I, size=2*r+1, mode='reflect')
    mean_p = uniform_filter(p, size=2*r+1, mode='reflect')
    mean_Ip = uniform_filter(I * p, size=2*r+1, mode='reflect')
    
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = uniform_filter(I * I, size=2*r+1, mode='reflect') - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = uniform_filter(a, size=2*r+1, mode='reflect')
    mean_b = uniform_filter(b, size=2*r+1, mode='reflect')
    
    return mean_a * I + mean_b


def generate_parameter_combinations():
    power_thresholds = [80, 90, 95]
    comparison_ranges = [2]
    distance_thresholds = [100]
    window_sizes = [21]
    
    
    power_cliff_percentiles=[]
    #pcen_alpha = [0.8, 0.85, 0.9, 0.95, 0.98, 1]
    #pcen_delta = [2, 4, 6, 8, 10]
    #pcen_r = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    sigma = [1]
    low_threshold = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    high_threshold = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    #return list(product(power_thresholds, comparison_ranges, distance_thresholds, pcen_alpha, pcen_delta, pcen_r))
    return list(product(power_thresholds, comparison_ranges, distance_thresholds, window_sizes, sigma, low_threshold, high_threshold))

    #return list(product(power_thresholds, comparison_ranges, distance_thresholds, window_sizes,))

#Spectrogram is dimensioned as freq, time
def trace_orig(spectrogram, comparison_range):
    trace = [0]
    curr_trace = 0
    for i in range(1, len(spectrogram[0]) - 1):
        best_shift = -curr_trace
        best_num_matches = 0
        for freq_shift in range(-comparison_range, comparison_range):
            num_matches = 0
            for j in range(0, len(spectrogram)):                    
                if j + freq_shift in range(0, len(spectrogram) - 1) and spectrogram[j][i-1] > 0 and spectrogram[j + freq_shift][i] > 0:
                    num_matches += 1
            if not num_matches:
                continue
            if num_matches > best_num_matches:
                best_num_matches = num_matches
                best_shift = freq_shift

        print(i, best_shift)
                
        
        curr_trace = curr_trace + best_shift
        trace.append(curr_trace)
    return trace

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

def convolve_skip_zeros(arr, kernel_size=3):
    kernel = np.ones(kernel_size)  # Uniform kernel
    smoothed = np.copy(arr)  # Copy to preserve zero positions

    # Count the number of nonzero elements in the window
    nonzero_counts = np.convolve((arr != 0).astype(float), kernel, mode='same')

    # Perform convolution, treating zero values as missing
    convolved_values = np.convolve(arr, kernel, mode='same')

    # Normalize by nonzero counts (avoid division by zero)
    mask = nonzero_counts > 0
    smoothed[mask] = convolved_values[mask] / nonzero_counts[mask]

    return smoothed



def get_entropy(window):
    # Normalize the matrix to obtain probabilities
    prob_mat = window / np.sum(window)

    # Replace 0 by small value to avoid log 0
    prob_mat = np.where(prob_mat > 0, prob_mat, 1e-10)

    # Calculate entropy
    entropy = -np.sum(prob_mat * np.log2(prob_mat))

    # Normalize by the size of the matrix
    entropy /= np.log2(np.size(window))

    return entropy



def denoise(original_matrix, output="", window_height=80, window_width=10, entropy_threshold=0.95,
            magnitude_threshold=0.4):
    matrix = original_matrix.copy()
    for line in range(0, matrix.shape[0], window_height):
        for column in range(0, matrix.shape[1], window_width):
            # window retrieval
            window = original_matrix[line:line + window_height, column:column + window_width]
            # Normalization
            if (np.max(window) == np.min(window)):
                continue
            #window = (window - np.min(window)) / (np.max(window) - np.min(window))
            # Entropy calculation
            entropy = get_entropy(window)
            if entropy > entropy_threshold:
                # Remove high entropy windows
                window = 0
            else:
                # Apply Magnitude filtering to windows that are kept
                window = np.where(window < magnitude_threshold, 0, window)
            if output == 'entropy':
                # show entropy value instead
                window = entropy
            # Replace old window values with new ones
            matrix[line:line + window_height, column:column + window_width] = window
    if output == 'original':
        # Get original non normalized values
        matrix = np.where(matrix != 0, original_matrix, 0)
    if output == 'mask':
        # Generate binary mask
        matrix = np.where(matrix != 0, 1, 0)
    return matrix

def mask_missing_values(series):
    # Create a mask where 1 represents valid values and 0 represents missing values
    mask = np.isnan(series)
    return mask

def trace(spectrogram, power_threshold, comparison_range, pcen_alpha=0.9, pcen_delta=2, pcen_r=0.5, kernel_size=3, window_size = 3, sigma = 3, high_threshold=0.5, low_threshold=0.2, filter = False, normalize_to_one=False):
    trace = [0]
    
    spectrogram = spectrogram[150:300, :]
    m, n = spectrogram.shape
    spectrogram = wiener(spectrogram)

    # Extreme values are capped to mean ± 1.5 std
    fact_ = 1.5
    mval = np.mean(spectrogram)
    sval = np.std(spectrogram)
    #=spectrogram[spectrogram > mval + fact_*sval] = mval + fact_*sval
    #spectrogram[spectrogram < mval - fact_*sval] = mval - fact_*sval
    inner=3
    outer=128
    wInner = np.ones(inner)
    wOuter = np.ones(outer)

    
    #R = spectrogram.copy()
    #for i in range(n):
        #spectrogram[:,i] = spectrogram[:,i] - (np.convolve(R[:,i],wOuter,'same') - np.convolve(R[:,i],wInner,'same'))/(outer - inner)


    modified_spectrogram = copy.deepcopy(spectrogram)
    yen_threshold = threshold_yen(modified_spectrogram)
    #print(yen_threshold)

    spec_min = np.min(modified_spectrogram)
    modified_spectrogram[modified_spectrogram < yen_threshold] = spec_min

    modified_spectrogram = (modified_spectrogram - np.min(modified_spectrogram)) / (np.max(modified_spectrogram) - np.min(modified_spectrogram)) # Normalization   

    #spectrogram = feature.canny(spectrogram, sigma=sigma, high_threshold=high_threshold, low_threshold=low_threshold)
        
    #Try closing - dilation then erosion
    structure = np.ones((3, 3))

    # Compute local density using convolution    
    #modified_spectrogram = denoise(modified_spectrogram)
    modified_spectrogram = feature.canny(modified_spectrogram, sigma=sigma, high_threshold=high_threshold, low_threshold=low_threshold)
    # Invert the binary image (skeletonize works better on white foreground)
    modified_spectrogram = binary_closing(input=modified_spectrogram, structure=structure)


    if filter:
        modified_spectrogram = filter_large_components(modified_spectrogram, 5)
                    
    # One-liner to compute the average of y-positions (row indices) where values > 0 for each column
    zeroes = np.where((np.array(modified_spectrogram) != 0).any(axis=0)== 0, 1, 0)
    #print(zeroes)
    trace = np.array([np.mean(np.where(modified_spectrogram[:, col] > 0)[0]) if np.any(modified_spectrogram[:, col] > 0) else 0 for col in range(modified_spectrogram.shape[1])])
    median_nonzero = np.median(trace[trace != 0])
    trace[trace==0] = median_nonzero
    #if filter: 
    #trace[trace[:] != 0]

    #trace = medfilt(trace, window_size)
    #zero_mask = trace[:] == 0
    #trace = np.convolve(trace, np.ones(5)/5, mode='valid')
    #trace = convolve_skip_zeros(trace, 5)
    
    # Where originally we have no signal, we don't want to artificially make a signal with the moving avg.
    #trace = trace * zero_mask
    
     #Normalize for DTW.
    trace = np.array(2 * (trace-np.min(trace))/(np.max(trace) - np.min(trace)) - 1 )
    for i in range(0, len(zeroes)):
        if zeroes[i]:
            trace[i] = -1

    #trace[zero_mask == 1] = 0 #Mark places where we want to skip because there is no data.
    
    return trace, modified_spectrogram, spectrogram


def apply_pcen(spectrogram, alpha=0.98, delta=2.0, r=0.5, eps=1e-6):
    pcen = librosa.pcen(spectrogram, sr=192000)
    return pcen

def create_class_templates(classes, training_dir, num_files_per_class=5, test_params=None, display=False):
    templates = {cls: [] for cls in classes}

    for cls in classes:
        label = class_mapping[cls]
        params = class_parameters[cls]
        power_threshold = params['power_threshold']
        comparison_range = params['comparison_range']
        sigma = params['sigma']
        low_threshold = params['low_threshold']
        high_threshold = params['high_threshold']
        #pcen_alpha = params['pcen_alpha']
        #pcen_delta = params['pcen_delta']
        #pcen_r = params['pcen_r']
        window_size = params['window_size']
        
        if test_params:
            power_threshold = test_params['power_threshold']
            comparison_range = test_params['comparison_range']
            sigma = params['sigma']
            low_threshold = params['low_threshold']
            high_threshold = params['high_threshold']
            #pcen_alpha = params['pcen_alpha']
            #pcen_delta = params['pcen_delta']
            #pcen_r = params['pcen_r']
            window_size = test_params['window_size']

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
            trace_result, modified_spectrogram, original_spectrogram = trace(spectrogram, power_threshold, comparison_range, window_size=window_size, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold, filter=True, normalize_to_one=True)

            if display:
                display_spectrogram_with_array(original_spectrogram, modified_spectrogram, trace_result, cls, cls)

            templates[cls].append(trace_result)

    return templates

def masked_distance(x, y):
    """Custom distance function that ignores zero values."""
    return euclidean(x, y)  # Use absolute difference otherwise


def optimize_class_parameters(classes, training_dir, num_templates_per_class=5):
    optimized_parameters = {}
    param_combinations = generate_parameter_combinations()
    
    for cls in classes:
        print(cls)
        label = class_mapping[cls]
        best_params = None
        best_score = float('inf')
        class_dir = os.path.join(training_dir, label)
        if not os.path.exists(class_dir):
            print(f"Class directory {class_dir} does not exist. Skipping.")
            continue
        class_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        test_files = class_files[num_templates_per_class:]
        if not test_files:
            print(f"Skipping class {cls} due to insufficient data.")
            continue
        for params in param_combinations:
            test_params = {
                'power_threshold': params[0],
                'comparison_range': params[1],
                'distance_threshold': params[2],
                #'pcen_alpha': best_params[3],
                #'pcen_delta': best_params[4],
                #'pcen_r': best_params[5]
                'window_size': params[3],
                'sigma': params[4],
                'low_threshold': params[5],
                'high_threshold': params[6]
            }
            new_templates = create_class_templates(classes, templates_dir, num_files_per_class=num_templates_per_class, test_params=test_params)


            #power_threshold, comparison_range, distance_threshold, pcen_alpha, pcen_delta, pcen_r = params
            power_threshold, comparison_range, distance_threshold, window_size, sigma, low_threshold, high_threshold = params
            if low_threshold > high_threshold:
                continue
            
            total_distance = None
            for filename in test_files:
                filepath = os.path.join(class_dir, filename)
                _, _, spectrogram = get_spectrogram(filepath)
                if spectrogram is None:
                    print(f"Skipping file {filename} due to invalid spectrogram.")
                    continue
                #trace_result, _ = trace(spectrogram, power_threshold, comparison_range, pcen_alpha, pcen_delta, pcen_r)

                trace_result, _, _ = trace(spectrogram, power_threshold, comparison_range, window_size=window_size, sigma=sigma, low_threshold=low_threshold, high_threshold = high_threshold)
                for template in new_templates[cls]:
                    distance, _, _, _ = accelerated_dtw(
                        np.array(trace_result).reshape(-1, 1),
                        np.array(template).reshape(-1, 1),
                        dist='euclidean',
                    )
                    if not total_distance:
                        total_distance = 0
                    total_distance += distance
            if not total_distance:
                continue
            avg_distance = total_distance / (len(test_files) * len(new_templates[cls]))
            if avg_distance < best_score:
                print(params)
                print(avg_distance)
                best_score = avg_distance
                best_params = params
        optimized_parameters[cls] = {
            'power_threshold': best_params[0],
            'comparison_range': best_params[1],
            'distance_threshold': best_params[2],
            #'pcen_alpha': best_params[3],
            #'pcen_delta': best_params[4],
            #'pcen_r': best_params[5]
            'window_size': best_params[3],
            'sigma': best_params[4],
            'low_threshold': best_params[5],
            'high_threshold': best_params[6]
        }
    return optimized_parameters

def classify_trace(spectrogram, templates, filename=""):
    best_distance = float('inf')
    best_distance_nonfiltered = float('inf')
    best_label = 'NULL'
    best_label_nonfiltered = 'NULL'
    
    best_distance_per_label = defaultdict(lambda: float('inf'))
    
    for label, class_templates in templates.items():
        params = class_parameters[label]
        power_threshold = params['power_threshold']
        comparison_range = params['comparison_range']
        distance_threshold = params['distance_threshold']
        #pcen_alpha = params['pcen_alpha']
        #pcen_delta = params['pcen_delta']
        #pcen_r = params['pcen_r']
        window_size = params['window_size']
        sigma = params['sigma']
        low_threshold = params['low_threshold']
        high_threshold = params['high_threshold']
        #trace_result, modified_spectrogram = trace(spectrogram, power_threshold, comparison_range, pcen_alpha, pcen_delta, pcen_r)
        trace_result_nonfiltered, modified_spectrogram, original_spectrogram = trace(spectrogram, power_threshold, comparison_range, window_size=window_size, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold, filter=False)
        trace_result, modified_spectrogram, original_spectrogram = trace(spectrogram, power_threshold, comparison_range, window_size=window_size, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold, filter=True)

        for template in class_templates:
            distance, _, _, _ = accelerated_dtw(
                np.array(trace_result).reshape(-1, 1),
                np.array(template).reshape(-1, 1),
                dist='euclidean',
                warp=1
            )
                
            if distance < best_distance:
                best_distance = distance
                best_label = label
            best_distance_per_label[label] = min(distance, best_distance_per_label[label])

                
        # Check if not filtering changes this.
        for template in class_templates:
            distance, _, _, _ = accelerated_dtw(
                np.array(trace_result_nonfiltered).reshape(-1, 1),
                np.array(template).reshape(-1, 1),
                dist='euclidean',
                warp=1
            )
            if distance < best_distance_nonfiltered:
                best_distance_nonfiltered = distance
                best_label_nonfiltered = label           
                
        # Experimental based on histogram.
        # Seems like when we mispredict as DEN, on avg the distance to ROP is >= 600, and if we predict DEN correctly it is < 600
        if(best_label == "DEN" and best_distance > distance_threshold):
            #best_label = "NULL"
            #print(best_label, best_distance)
            pass
    
    return best_label, best_distance, trace_result, modified_spectrogram, original_spectrogram, best_distance_per_label, best_label_nonfiltered, trace_result_nonfiltered

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generates a 2D Gaussian kernel.

    Parameters:
        size (int): The size of the kernel (must be odd).
        sigma (float): The standard deviation of the Gaussian.

    Returns:
        np.ndarray: The 2D Gaussian kernel.
    """
    # Ensure the kernel size is odd
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    # Create a coordinate grid
    radius = size // 2
    x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))

    # Compute the Gaussian function
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= 2 * np.pi * sigma**2

    # Normalize the kernel to ensure the sum is 1
    kernel /= kernel.sum()

    return kernel

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
    
def display_spectrogram_with_array_filtered(original_spectrogram: np.ndarray, modified_spectrogram: np.ndarray, array: np.ndarray, array_filtered: np.ndarray,
                                   spectrogram_title: str = "Spectrogram", 
                                   array_title: str = "1D Array", array_title_filtered: str = "1D Array Filtered", save: bool = False, title: str = "test"):
    """
    Displays a 2D numpy array (spectrogram) side by side with a 1D numpy array.

    Args:
        spectrogram (np.ndarray): 2D array representing the spectrogram.
        array (np.ndarray): 1D array to display alongside the spectrogram.
        spectrogram_title (str): Title for the spectrogram plot.
        array_title (str): Title for the 1D array plot.
    """
    fig, axes = plt.subplots(1, 4, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 3, 3, 3]})
    
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
    
    # Plot the 1D array horizontally
    ax3 = axes[3]
    ax3.plot(array_filtered, label=array_title, color='orange')
    ax3.set_title("Filtered " + array_title_filtered)
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Value')
    ax3.grid(True)
    
    plt.tight_layout()
    if save:
        plt.savefig(f"./pictures7/{title}.png", dpi=300, bbox_inches='tight')  # Save as PNG with high resolution
        plt.close()
    else:
        plt.show()

    
def show_histogram_of_distances(distances, title, bins=6):
    # Create a histogram
    plt.hist(distances, bins=bins, color='blue', edgecolor='black', alpha=0.7)

    # Add labels and title
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title(title)

    # Show the plot
    plt.show()
    
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

if __name__ == "__main__":
    classes = ['DEN', 'ROP', 'SCA', 'GRA', 'SAR']
    training_dir = "./2024DolphinDataset/training"
    testing_dir = "./2024DolphinDataset/testing"
    templates_dir = "./2024DolphinDataset/templates"
    
    print(os.path.exists("."))

    print("Optimizing parameters\n")
    #optimized_parameters = optimize_class_parameters(classes, training_dir)
    optimized_parameters = {
        'DEN': {'power_threshold': 80, 'comparison_range': 2, 'distance_threshold': 600, 'window_size': 21, 'sigma': 1, 'low_threshold': 0.9, 'high_threshold': 0.95}, 
        'ROP': {'power_threshold': 80, 'comparison_range': 2, 'distance_threshold': 600, 'window_size': 21, 'sigma': 1, 'low_threshold': 0.9, 'high_threshold': 0.95}, 
        'SCA': {'power_threshold': 80, 'comparison_range': 2, 'distance_threshold': 600, 'window_size': 21, 'sigma': 1, 'low_threshold': 0.9, 'high_threshold': 0.95}, 
        'GRA': {'power_threshold': 80, 'comparison_range': 2, 'distance_threshold': 600, 'window_size': 21, 'sigma': 1, 'low_threshold': 0.9, 'high_threshold': 0.95}, 
        'SAR': {'power_threshold': 80, 'comparison_range': 2, 'distance_threshold': 600, 'window_size': 21, 'sigma': 1, 'low_threshold': 0.9, 'high_threshold': 0.95}}
    
    print("Optimized Parameters:", optimized_parameters)
    class_parameters.update(optimized_parameters)

    #templates = create_class_templates(classes, training_dir, num_files_per_class=5, display=False)
    #class_parameters = {'DEN': {'power_threshold': 90, 'comparison_range': 2, 'distance_threshold': 100, 'window_size': 11}, 'ROP': {'power_threshold': 90, 'comparison_range': 2, 'distance_threshold': 100, 'window_size': 21}, 'SCA': {'power_threshold': 90, 'comparison_range': 2, 'distance_threshold': 100, 'window_size': 11}, 'GRA': {'power_threshold': 90, 'comparison_range': 2, 'distance_threshold': 100, 'window_size': 31}, 'SAR': {'power_threshold': 90, 'comparison_range': 2, 'distance_threshold': 100, 'window_size': 21}}
    
    print("Loading templates\n")
    templates = create_class_templates(classes, templates_dir, num_files_per_class=5, display=True)

    print("Running classification experiment\n")
    y_true = []
    y_pred = []
    
    """
    Experiment: 
    I want to know the average distance to every class when we mispredict denise and when we correctly predict denise.
    If they are wildly different, we have a good way to detect when a denise prediction is a misprediction
    """
    
    #[true class, class for distance]
    all_class_distance_mispredict_denise = defaultdict(lambda: defaultdict(lambda: []))
    all_class_distance_correct_predict_denise = defaultdict(lambda: [])
    
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
            predicted_label, best_distance, trace_result, modified_spectrogram, original_spectrogram, best_distance_per_label, predicted_label_nonfiltered, trace_result_nonfiltered = classify_trace(spectrogram, templates, filename)
            y_true.append(cls)
            y_pred.append(predicted_label)
            
            if cls != predicted_label:
                #print(i, filename, f"predicted: {predicted_label}, actual: {cls}, distance: {best_distance}")
                #try_all_threshold(spectrogram)
                #plt.show()
                print(i, filename, f"predicted: {predicted_label}, nonfiltered_predicted: {predicted_label_nonfiltered}, actual: {cls}, distance: {best_distance}, distance to actual: {best_distance_per_label[cls]}")
                #display_spectrogram_with_array(original_spectrogram, modified_spectrogram, trace_result, cls, predicted_label)
                display_spectrogram_with_array_filtered(original_spectrogram, modified_spectrogram, trace_result_nonfiltered, trace_result, cls, predicted_label_nonfiltered, predicted_label, save=True, title=filename)
                
            if predicted_label == 'DEN':
                if cls != predicted_label:
                    for curr_cls in classes:
                        all_class_distance_mispredict_denise[cls][curr_cls].append(best_distance_per_label[curr_cls])
                else:
                    for curr_cls in classes:
                        all_class_distance_correct_predict_denise[curr_cls].append(best_distance_per_label[curr_cls])
    
    for true_cls in classes:
        for cls_for_dist in classes:
            #show_histogram_of_distances(all_class_distance_mispredict_denise[true_cls][cls_for_dist], f"Distances to {cls_for_dist} when mispredicting DEN with true class {true_cls}", 20)
            pass
        
    for cls_for_dist in classes:
        #show_histogram_of_distances(all_class_distance_correct_predict_denise[cls_for_dist], f"Distances to {cls_for_dist} when we predicted DEN correctly", 20)
        pass

    print(y_pred)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=classes))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=classes))



