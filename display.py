import numpy as np
import matplotlib.pyplot as plt

def load_spectrogram(filename, delimiter=" "):
    """Reads a spectrogram from a text file, removes trailing commas, and converts it into a NumPy array."""
    with open(filename, "r") as file:
        cleaned_lines = [line.replace(",", "").strip() for line in file]  # Remove commas
    data = np.loadtxt(cleaned_lines, delimiter=delimiter)
    return data

def display_spectrogram(data):
    """Displays the spectrogram as an image."""
    plt.figure(figsize=(10, 5))
    plt.imshow(data, aspect="auto", origin="lower", cmap="inferno")  # Adjust colormap if needed
    plt.colorbar(label="Intensity")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title("Spectrogram")
    plt.show()

# Example usage
filename = "saved_templates/grass/0/0.txt"
#filename = "spectrogram_.txt"  # Replace with your file name
spectrogram = load_spectrogram(filename)
display_spectrogram(spectrogram)
