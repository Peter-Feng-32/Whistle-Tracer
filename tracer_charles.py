from scipy.signal import stft
import numpy as np
import os
from scipy.io import wavfile
from scipy import signal
from dtw import accelerated_dtw
from itertools import product
from sklearn.metrics import confusion_matrix, classification_report
import librosa

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
    'DEN': {'power_threshold': 75, 'comparison_range': 2, 'distance_threshold': 100},
    'ROP': {'power_threshold': 80, 'comparison_range': 3, 'distance_threshold': 120},
    'SCA': {'power_threshold': 70, 'comparison_range': 2, 'distance_threshold': 110},
    'GRA': {'power_threshold': 85, 'comparison_range': 4, 'distance_threshold': 130},
    'SAR': {'power_threshold': 75, 'comparison_range': 3, 'distance_threshold': 115},
}

def generate_parameter_combinations():
    power_thresholds = [90]
    comparison_ranges = [2, 3, 4]
    distance_thresholds = [100, 110, 120, 130]
    return list(product(power_thresholds, comparison_ranges, distance_thresholds))

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

def trace(spectrogram, power_threshold, comparison_range):
    trace = [0]
    power_threshold = np.percentile(spectrogram.flatten(), 90)
    spectrogram[spectrogram < power_threshold] = 0
    spectrogram = apply_pcen(spectrogram[150:300, :])
    trace = np.argmax(spectrogram, axis=0)
    trace = np.convolve(trace, np.ones(10)/10, mode='valid')
    return trace

def apply_pcen(spectrogram, alpha=0.98, delta=2.0, r=0.5, eps=1e-6):
    pcen = librosa.pcen(spectrogram, sr=192000)
    return pcen

def create_class_templates(classes, training_dir, num_files_per_class=5):
    templates = {cls: [] for cls in classes}

    for cls in classes:
        label = class_mapping[cls]
        params = class_parameters[cls]
        power_threshold = params['power_threshold']
        comparison_range = params['comparison_range']

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
            trace_result = trace(spectrogram, power_threshold, comparison_range)
            templates[cls].append(trace_result)

    return templates

def optimize_class_parameters(classes, training_dir, templates):
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
        if not test_files or not templates[cls]:
            print(f"Skipping class {cls} due to insufficient data.")
            continue
        for params in param_combinations:
            power_threshold, comparison_range, distance_threshold = params
            total_distance = 0
            for filename in test_files:
                filepath = os.path.join(class_dir, filename)
                _, _, spectrogram = get_spectrogram(filepath)
                if spectrogram is None:
                    print(f"Skipping file {filename} due to invalid spectrogram.")
                    continue
                trace_result = trace(spectrogram, power_threshold, comparison_range)
                for template in templates[cls]:
                    distance, _, _, _ = accelerated_dtw(
                        np.array(trace_result).reshape(-1, 1),
                        np.array(template).reshape(-1, 1),
                        dist='euclidean'
                    )
                    total_distance += distance
            avg_distance = total_distance / (len(test_files) * len(templates[cls]))
            if avg_distance < best_score:
                best_score = avg_distance
                best_params = params
        optimized_parameters[cls] = {
            'power_threshold': best_params[0],
            'comparison_range': best_params[1],
            'distance_threshold': best_params[2]
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
        trace_result = trace(spectrogram, power_threshold, comparison_range)
        for template in class_templates:
            distance, _, _, _ = accelerated_dtw(
                np.array((trace_result-np.mean(trace_result)/np.std(trace_result))).reshape(-1, 1),
                np.array((template-np.mean(template)/np.std(template))).reshape(-1, 1),
                dist='euclidean'
            )
            if distance < best_distance:
                best_distance = distance
                best_label = label
    return best_label, best_distance

if __name__ == "__main__":
    classes = ['DEN', 'ROP', 'SCA', 'GRA', 'SAR']
    training_dir = "/Users/charles/Downloads/2024DolphinDataset/training"
    testing_dir = "/Users/charles/Downloads/2024DolphinDataset/testing"

    print("Loading templates\n")
    templates = create_class_templates(classes, training_dir, num_files_per_class=2)

    print("Optimizing parameters\n")
    optimized_parameters = optimize_class_parameters(classes, training_dir, templates)
    print("Optimized Parameters:", optimized_parameters)
    class_parameters.update(optimized_parameters)

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
            predicted_label, _ = classify_trace(spectrogram, templates)
            y_true.append(cls)
            y_pred.append(predicted_label)
    print(y_pred)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=classes))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=classes))
