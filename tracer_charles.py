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
    'DEN': {'power_threshold': 75, 'comparison_range': 2, 'distance_threshold': 100, 'window_size': 5},
    'ROP': {'power_threshold': 80, 'comparison_range': 3, 'distance_threshold': 120, 'window_size': 5},
    'SCA': {'power_threshold': 70, 'comparison_range': 2, 'distance_threshold': 110, 'window_size': 5},
    'GRA': {'power_threshold': 85, 'comparison_range': 4, 'distance_threshold': 130, 'window_size': 5},
    'SAR': {'power_threshold': 75, 'comparison_range': 3, 'distance_threshold': 115, 'window_size': 5},
}

def generate_parameter_combinations():
    power_thresholds = [90]
    comparison_ranges = [2, 3, 4]
    distance_thresholds = [100, 110, 120, 130]
    window_sizes = [5, 11, 21, 31]
    power_cliff_percentiles=[]
    #pcen_alpha = [0.8, 0.85, 0.9, 0.95, 0.98, 1]
    #pcen_delta = [2, 4, 6, 8, 10]
    #pcen_r = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    #return list(product(power_thresholds, comparison_ranges, distance_thresholds, pcen_alpha, pcen_delta, pcen_r))
    return list(product(power_thresholds, comparison_ranges, distance_thresholds, window_sizes))

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
        spectrogram = (spectrogram - np.mean(spectrogram, axis=0)) / np.std(spectrogram, axis=0)
        return frequencies, times, spectrogram
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return None, None, None

def trace(spectrogram, power_threshold, comparison_range, pcen_alpha=0.9, pcen_delta=2, pcen_r=0.5, kernel_size=3, window_size = 3):
    trace = [0]
    power_threshold = np.percentile(spectrogram.flatten(), 90)
    spectrogram[spectrogram < power_threshold] = 0
    #spectrogram = apply_pcen(spectrogram[150:300, :], pcen_alpha, pcen_delta, pcen_r,)
    spectrogram = spectrogram[150:300, :]
    trace = np.argmax(spectrogram, axis=0)
    # Create a uniform kernel to calculate local density
    kernel = np.array([
        [0.1, 0.1, 0.1],
        [0.1, 1.0, 0.1],
        [0.1, 0.1, 0.1]
    ])
    # Compute local density using convolution
    #local_density = convolve((spectrogram > 0).astype(float), kernel, mode='constant', cval=0)
    #spectrogram = spectrogram * local_density
    modified_spectrogram = spectrogram
    #trace = medfilt(trace, window_size)

    #trace = np.convolve(trace, np.ones(10)/10, mode='valid')
    return trace, modified_spectrogram



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
        #pcen_alpha = params['pcen_alpha']
        #pcen_delta = params['pcen_delta']
        #pcen_r = params['pcen_r']
        window_size = params['window_size']
        
        if test_params:
            power_threshold = test_params['power_threshold']
            comparison_range = test_params['comparison_range']
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
            trace_result, modified_spectrogram = trace(spectrogram, power_threshold, comparison_range, window_size=window_size)

            if display:
                display_spectrogram_with_array(modified_spectrogram, trace_result, cls, cls)

            templates[cls].append(trace_result)

    return templates

def optimize_class_parameters(classes, training_dir):
    optimized_parameters = {}
    param_combinations = generate_parameter_combinations()
    
    for cls in classes:
        label = class_mapping[cls]
        best_params = None
        best_score = float('inf')
        class_dir = os.path.join(training_dir, label)
        if not os.path.exists(class_dir):
            print(f"Class directory {class_dir} does not exist. Skipping.")
            continue
        class_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        test_files = class_files[:3]
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
                'window_size': params[3]
            }
            new_templates = create_class_templates(classes, training_dir, num_files_per_class=5, test_params=test_params)


            #power_threshold, comparison_range, distance_threshold, pcen_alpha, pcen_delta, pcen_r = params
            power_threshold, comparison_range, distance_threshold, window_size = params

            total_distance = 0
            for filename in test_files:
                filepath = os.path.join(class_dir, filename)
                _, _, spectrogram = get_spectrogram(filepath)
                if spectrogram is None:
                    print(f"Skipping file {filename} due to invalid spectrogram.")
                    continue
                #trace_result, _ = trace(spectrogram, power_threshold, comparison_range, pcen_alpha, pcen_delta, pcen_r)
                trace_result, _ = trace(spectrogram, power_threshold, comparison_range, window_size=window_size)
                #TODO: This is technically wrong since currently the templates come from the test set,
                # but it affects the avg distance equally for all parameter combinations, so it doesn't affect anything currently
                # Fix this by making some whistles into templates(template dir).  K-fold?
                for template in new_templates[cls]:
                    distance, _, _, _ = accelerated_dtw(
                        np.array(trace_result).reshape(-1, 1),
                        np.array(template).reshape(-1, 1),
                        dist='euclidean'
                    )
                    total_distance += distance
            avg_distance = total_distance / (len(test_files) * len(new_templates[cls]))
            if avg_distance < best_score:
                best_score = avg_distance
                best_params = params
        optimized_parameters[cls] = {
            'power_threshold': best_params[0],
            'comparison_range': best_params[1],
            'distance_threshold': best_params[2],
            #'pcen_alpha': best_params[3],
            #'pcen_delta': best_params[4],
            #'pcen_r': best_params[5]
            'window_size': best_params[3]
        }
    return optimized_parameters

def classify_trace(spectrogram, templates):
    best_distance = float('inf')
    best_label = 'NULL'
    for label, class_templates in templates.items():
        params = class_parameters[label]
        power_threshold = params['power_threshold']
        comparison_range = params['comparison_range']
        distance_threshold = params['distance_threshold']
        #pcen_alpha = params['pcen_alpha']
        #pcen_delta = params['pcen_delta']
        #pcen_r = params['pcen_r']
        window_size = params['window_size']
        #trace_result, modified_spectrogram = trace(spectrogram, power_threshold, comparison_range, pcen_alpha, pcen_delta, pcen_r)
        trace_result, modified_spectrogram = trace(spectrogram, power_threshold, comparison_range, window_size=window_size)

        for template in class_templates:
            distance, _, _, _ = accelerated_dtw(
                np.array((trace_result-np.mean(trace_result)/np.std(trace_result))).reshape(-1, 1),
                np.array((template-np.mean(template)/np.std(template))).reshape(-1, 1),
                dist='euclidean'
            )
            if distance < best_distance:
                best_distance = distance
                best_label = label
    return best_label, best_distance, trace_result, modified_spectrogram


def display_spectrogram_with_array(spectrogram: np.ndarray, array: np.ndarray, 
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})
    
    # Plot the spectrogram
    ax1 = axes[0]
    im = ax1.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_title(spectrogram_title)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency')
    plt.colorbar(im, ax=ax1, orientation='vertical', label='Intensity')
    
    # Plot the 1D array horizontally
    ax2 = axes[1]
    ax2.plot(array, label=array_title, color='orange')
    ax2.set_title(array_title)
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Value')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    classes = ['DEN', 'ROP', 'SCA', 'GRA', 'SAR']
    training_dir = "./2024DolphinDataset/training"
    testing_dir = "./2024DolphinDataset/testing"
    
    print(os.path.exists("."))

    print("Optimizing parameters\n")
    optimized_parameters = optimize_class_parameters(classes, training_dir)
    print("Optimized Parameters:", optimized_parameters)
    class_parameters.update(optimized_parameters)
    
    #class_parameters = {'DEN': {'power_threshold': 90, 'comparison_range': 2, 'distance_threshold': 100, 'window_size': 11}, 'ROP': {'power_threshold': 90, 'comparison_range': 2, 'distance_threshold': 100, 'window_size': 21}, 'SCA': {'power_threshold': 90, 'comparison_range': 2, 'distance_threshold': 100, 'window_size': 11}, 'GRA': {'power_threshold': 90, 'comparison_range': 2, 'distance_threshold': 100, 'window_size': 31}, 'SAR': {'power_threshold': 90, 'comparison_range': 2, 'distance_threshold': 100, 'window_size': 21}}
    
    print("Loading templates\n")
    templates = create_class_templates(classes, training_dir, num_files_per_class=5, display=False)

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
        for filename in class_files:
            filepath = os.path.join(class_dir, filename)
            _, _, spectrogram = get_spectrogram(filepath)
            if spectrogram is None:
                print(f"Skipping file {filename} due to invalid spectrogram.")
                continue
            predicted_label, _, trace_result, modified_spectrogram = classify_trace(spectrogram, templates)
            y_true.append(cls)
            y_pred.append(predicted_label)
            
            #display_spectrogram_with_array(modified_spectrogram, trace_result, cls, predicted_label)
    print(y_pred)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=classes))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=classes))

