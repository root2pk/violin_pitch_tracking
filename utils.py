"""
Utility functions for pitch tracking and pitch evaluation, to support the main notebook.

Functions:
- time_to_frames: Convert time to frames
- load_files: Add file paths to a list and return it
- track_lengths: Get the length of each audio track in seconds, and total length of all tracks
- get_pitch_tracks: Extract pitch tracks from a file containing time, pitch and confidence values
- get_activation_matrix: Get the activation matrix from a file containing time, pitch and confidence values
- compute_pitch_tracks_from_activation: Compute pitch tracks from the activation matrix, using various viterbi algorithms
- save_pitch_tracks_from_activation: Compute pitch tracks from the activation matrix, using various viterbi algorithms, and save them
- apply_voicing_threshold: Apply a voicing threshold to the pitch track
- plot_pitch: Plot pitch tracking information including the fundamental frequency (F0) over time, the confidence of the estimates
- plot_pitch_comparison: Plot the pitch contour of a track and compare it to the vocal pitch annotations
- plot_activation: Plot the activation matrix
- sonify_pitch_contour: Sonify the pitch contour using a simple sine wave generator
- otsu_threshold: Compute the optimal threshold using Otsu's method
- get_threshold: Get the optimal threshold using Otsu's method, and optionally plot the histogram
- get_vocal_pitch_annotations: Get the vocal pitch annotations for a track
"""

import librosa
import os
import numpy as np
import matplotlib.pyplot as plt 
import mir_eval
from fastdtw import fastdtw
from mir_eval import display as mir_display
from scipy.signal import sawtooth
from scipy.interpolate import interp1d
from IPython.display import Audio, display
from core import to_viterbi_cents, to_local_average_cents, to_weird_viterbi_cents, output_path

def time_to_frames(start_time, end_time, step_size=10):
    """
    Convert time to frames

    Parameters:
    start_time : float
        The start time in seconds
    end_time : float
        The end time in seconds
    step_size : int
        The step size in milliseconds

    Returns:
    start_frame : int
        The start frame
    end_frame : int
        The end frame
    """
    start_frame = int(start_time * 1000 / step_size)
    end_frame = int(end_time * 1000 / step_size)
    return start_frame, end_frame

def load_files(files_path, extension=".wav"):
    """
    Add file paths to a list and return it

    Parameters:
    files_path : str
        The path to the directory containing the files
    extension : str
        The file extension to search for, ".wav" by default

    Returns:
    audio_files : list
        List containing the paths of the audio files
    """
    
    audio_files = []
    for root, dirs, files in os.walk(files_path):
        for file in files:
            if file.endswith(extension):
                audio_files.append(os.path.join(root, file))
    return audio_files

def track_lengths(audio_files):
    """
    Get the length of each audio track in seconds, and total length of all tracks

    Parameters:
    audio_files : list
        List containing the paths of the audio files

    Returns:
    track_lengths : np.ndarray
        Numpy array containing the length of each audio track in seconds
    total_length : float
        Total length of all tracks in seconds
    """

    track_lengths = []
    total_length = 0
    for file in audio_files:
        y, sr = librosa.load(file, sr=None, mono=True)
        track_lengths.append(librosa.get_duration(y=y, sr=sr))
        total_length += librosa.get_duration(y=y, sr=sr)
    return np.array(track_lengths), total_length

def get_pitch_tracks(f0_file):
    """
    Extract pitch tracks from a file containing time, pitch and confidence values

    Parameters:
    f0_file : str
        The file path to the input f0 file. Format: csv file with columns time, pitch and confidence

    Returns:
    time : np.ndarray
        Numpy array containing the time values
    pitch : np.ndarray
        Numpy array containing the pitch values
    confidence : np.ndarray
        Numpy array containing the confidence values
    """
    
    # Load f0 csv file, ignore header
    time, pitch, confidence = np.loadtxt(f0_file, delimiter=',', unpack=True, skiprows=1)
    return time, pitch, confidence

def get_activation_matrix(activation_file):
    """
    Get the activation matrix from a file containing time, pitch and confidence values

    Parameters:
    activation_file : str
        The file path to the input activation file. Format: .npy file containing the activation matrix
    
    Returns:
    activation : np.ndarray
        Numpy array containing the activation matrix
    """    
    activation = np.load(activation_file)
    return activation

def compute_pitch_tracks_from_activation(activation_file, viterbi=False, step_size=10):
    """
    Compute pitch tracks from the activation matrix, using various viterbi algorithms

    Parameters:
    activation_file : str
        The file path to the input activation file. Format: .npy file containing the activation matrix
    viterbi : 'weird' or bool
        If 'weird', use the 'weird' viterbi algorithm. If True, use the viterbi algorithm. If False, use the local average algorithm.
    step_size : int
        The step size in milliseconds

    Returns:
    time : np.ndarray
        Numpy array containing the time values
    frequency : np.ndarray
        Numpy array containing the frequency values
    confidence : np.ndarray
        Numpy array containing the confidence values
    """
    # Load activation matrix
    activation = get_activation_matrix(activation_file)

    # Compute 
    if viterbi == "weird":
        path, cents = to_weird_viterbi_cents(activation)
        confidence = np.array([activation[i, path[i]] for i in range(len(activation))])
    elif viterbi:
        # NEW!! CONFIDENCE IS NO MORE THE MAX ACTIVATION! CORRECTED TO BE CALCULATED ALONG THE PATH!
        path, cents = to_viterbi_cents(activation)
        confidence = np.array([activation[i, path[i]] for i in range(len(activation))])
    else:
        cents = to_local_average_cents(activation)
        confidence = activation.max(axis=1)

    frequency = 10 * 2 ** (cents / 1200)
    frequency[np.isnan(frequency)] = 0

    time = np.arange(confidence.shape[0]) * step_size / 1000.0

    return time, frequency, confidence

def save_pitch_tracks_from_activation(activation_file, output_path, viterbi=False, step_size=10, verbose=True):
    """
    Compute pitch tracks from the activation matrix, using various viterbi algorithms, and save them

    Parameters:
    activation_file : str
        The file path to the input activation file. Format: .npy file containing the activation matrix
    output_path : str
        The path to the directory where the pitch tracks will be saved
    viterbi : 'weird' or bool
        If 'weird', use the 'weird' viterbi algorithm. If True, use the viterbi algorithm. If False, use the local average algorithm.
    step_size : int
        The step size in milliseconds
    verbose : bool
        If True, print a message when the pitch tracks are saved

    Returns:
    None

    Note:
    The pitch tracks are saved as a csv file with columns time, frequency and confidence
    """
    time, frequency, confidence = compute_pitch_tracks_from_activation(activation_file, viterbi, step_size)

    # file name, remove everything after the first .
    f0_file = os.path.basename(activation_file)
    f0_file = os.path.split(f0_file)[1].split('.')[0]
    f0_file = os.path.join(output_path, f0_file + f"_viterbi={viterbi}.csv")

    f0_data = np.vstack([time, frequency, confidence]).transpose()
    np.savetxt(f0_file, f0_data, fmt=['%.3f', '%.3f', '%.6f'], delimiter=',',
               header='time,frequency,confidence', comments='')
    if verbose:
        print("CREPE: Saved the estimated frequencies and confidence values "
              "at {}".format(f0_file))  

def apply_voicing_threshold(pitch, confidence, threshold):
    """
    Apply a voicing threshold to the pitch track

    Parameters:
    pitch : np.ndarray
        Numpy array containing the pitch values
    confidence : np.ndarray
        Numpy array containing the confidence values
    threshold : float
        The voicing threshold

    Returns:
    pitch_voiced : np.ndarray
        Numpy array containing the voiced pitch values
    """
    
    pitch_voiced = np.where(confidence > threshold, pitch, 0)
    return pitch_voiced

def plot_pitch(time, frequency, confidence):
    """
    Plot pitch tracking information including the fundamental frequency (F0) over time, 
    the confidence of the estimates.

    Parameters
    ----------
    time : array_like
        An array of time stamps at which the frequency and confidence values are estimated.
    frequency : array_like
        An array containing estimated fundamental frequency (F0) values in Hertz (Hz) for each time stamp.
    confidence : array_like
        An array containing confidence values associated with each F0 estimate.

    """
    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(12, 8), sharex=False)
    axes[0].plot(time, frequency)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Estimated F0 (Hz)")
    axes[0].set_title("F0 Estimate Over Time")
    
    axes[1].plot(time, confidence)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Confidence")
    axes[1].set_title("Estimate Confidence Over Time")
    
    plt.tight_layout()
    plt.show()

def plot_pitch_comparison(time, pitch, time_ref, pitch_ref, title=None):
    """
    Plot the pitch contour of a track and compare it to the vocal pitch annotations

    Parameters:
    time : np.ndarray
        Numpy array containing the time values
    pitch : np.ndarray
        Numpy array containing the pitch values
    time_ref : np.ndarray
        Numpy array containing the time values of the vocal pitch annotations
    pitch_ref : np.ndarray
        Numpy array containing the pitch values of the vocal pitch annotations
    title : str
        The title of the plot, default is None
    """

    # Plot the pitch contour
    plt.figure(figsize=(16, 7))
    # mir_display.pitch(time_ref, pitch_ref, color="k", linewidth=2.1, label="Reference", ax=ax)
    mir_display.pitch(time, pitch, color="r", linewidth=0.8, label="Voilin estimated")
    mir_display.pitch(time_ref, pitch_ref, color="k", linewidth=0.8, label="Vocal pitch annotations")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    if title:
        plt.title(title)
    plt.legend()
    plt.show()

def plot_activation(activation):
    """
    Plot the activation matrix

    Parameters:
    activation : np.ndarray
        Numpy array containing the activation matrix, shape (n_frames, n_bins)
    sr : int
        Sampling rate
    hop_length : int
        Hop length
    fmin : int
        Minimum frequency
    fmax : int
        Maximum frequency
    """

    plt.figure(figsize=(16, 8))
    plt.imshow(activation.T, origin="lower", aspect="auto")
    
    c1 = 32.7 # Hz, fix for a known issue in CREPE
    c1_cent = mir_eval.melody.hz2cents(np.array([c1]))[0]
    c = np.arange(0, 360) * 20 + c1_cent
    freq = 10 * 2 ** (c / 1200)
    
    plt.yticks(np.arange(len(freq))[::35], [int(f) for f in freq[::35]])
    plt.ylim([0, 300])
    # plt.xticks((np.arange(len(activation))[::500] / 100).astype(int))
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.title("Activation Matrix: 20 Cent Bins Over Time")
    plt.colorbar()
    plt.show()

def sonify_pitch_contour(time, pitch, sr=16000, step_size=10, type = 'sine'):
    """
    Sonify the pitch contour using a simple sine wave generator

    Parameters:
    time : np.ndarray
        Numpy array containing the time values
    pitch : np.ndarray
        Numpy array containing the pitch values
    sr : int
        Sampling rate, default is 16000
    step_size : int
        The step size in milliseconds, default is 10
    type : str  
        The type of the waveform, 'sine' or 'sawtooth', default is 'sine'

    Returns:    
    y : np.ndarray
        Numpy array containing the audio signal
    """
    # Start time at 0
    time = time - time[0]
    # Audio length
    length_in_samples = int(np.ceil(time[-1]) * sr)

    if type == 'sine':

        # Convert pitch contour to audio signal
        y = mir_eval.sonify.pitch_contour(time, pitch, fs=sr, length = length_in_samples)
    
    elif type == 'sawtooth':
        # Convert pitch contour to audio signal
        y  = mir_eval.sonify.pitch_contour(time, pitch, fs=sr, length = length_in_samples, function=sawtooth)
    # Display audio
    display(Audio(y, rate=sr))

    # Return audio signal
    return y


def otsu_threshold(hist):
    """
    Compute the optimal threshold using Otsu's method

    Parameters:
    hist : np.ndarray
        Numpy array containing the histogram of the confidence values

    Returns:
    optimal_threshold : int
        The optimal threshold
    """
    total_pixels = np.sum(hist)
    sum_total = 0
    for t in range(len(hist)):
        sum_total += t * hist[t]

    sum_background = 0
    weight_background = 0
    weight_foreground = 0

    var_max = 0
    optimal_threshold = 0

    for t in range(len(hist)):
        weight_background += hist[t]
        if weight_background == 0:
            continue

        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * hist[t]

        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        var_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if var_between > var_max:
            var_max = var_between
            optimal_threshold = t

    return optimal_threshold

def get_threshold(confidence, title = None, plot=True, bins = 256):
    """
    Get the optimal threshold using Otsu's method, and optionally plot the histogram

    Parameters:
    confidence : np.ndarray
        Numpy array containing the confidence values
    title : str 
        The title of the histogram
    plot : bool
        If True, plot the histogram
    bins : int
        The number of bins in the histogram, default is 256

    Returns:
    threshold : float
        The optimal threshold
    """
    # Calculate optimal threshold using Otsu's method
    hist, _ = np.histogram(confidence, bins=256, range=(0, 1))
    threshold = otsu_threshold(hist)/256

    if plot:
        plt.figure(figsize=(12, 6))
        plt.hist(confidence, bins=256, color='r', alpha=0.7, range=(0, 1))
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        plt.axvline(x=threshold, color='k', linestyle='--', label=f"Otsu's Threshold: {threshold:.2f}")
        if title:
            plt.title(title)
        plt.show()

    return threshold

def get_vocal_pitch_annotations(track_name, vocal_pitch_path):
    """
    Get the vocal pitch annotations for a track

    Parameters:
    track_name : str
        The name of the track
    vocal_pitch_path : str
        The path to the directory containing the vocal pitch annotations

    Returns:
    time_ref : np.ndarray
        Numpy array containing the time values
    pitch_ref : np.ndarray
        Numpy array containing the pitch values
    """
    vocal_pitch_file = None  

    # In given directory, find the file whch starts with the track name
    for root, dirs, files in os.walk(vocal_pitch_path):
        for file in files:
            if file.startswith(track_name):
                vocal_pitch_file = os.path.join(root, file)
                break
    
    if vocal_pitch_file:
        # Load the vocal pitch annotations, the file is in format time, pitch separated by tab, no header
        time, pitch = np.loadtxt(vocal_pitch_file, delimiter='\t', unpack=True)
        print(f"Vocal pitch annotations found for track {track_name}")
        return time, pitch
    else:
        print("No track found")
        return None, None

