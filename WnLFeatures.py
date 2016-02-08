import librosa as lib
import numpy as np
import math
import scipy.fftpack as fft


def beat_variance_ratio(rate, data, tolerance=0.1):
    tempo, beat_frames = lib.beat.beat_track(y=data, sr=rate)

    beat_times = lib.frames_to_time(beat_frames, sr=rate)

    beat_diffs = np.diff(beat_times)

    mean_diff = np.mean(beat_diffs)
    upper_lim = mean_diff * 1 + tolerance
    lower_lim = mean_diff * 1 - tolerance

    outside_lim = len([x for x in beat_diffs if lower_lim < x < upper_lim])
    beat_ratio = 100 - ((outside_lim / len(beat_diffs)) * 100)

    return beat_ratio


def silence_ratio(rate, data, tolerance=0.05):
    num_samples = int(math.ceil(rate/1000))
    num_frames = int(math.ceil(len(data)/num_samples))

    # get the maximum absolute number to use for normalisation of the data
    norm_num = abs(data.max()) if abs(data.max()) > abs(data.min()) else abs(data.min())

    # Normalise the data
    norm_data = [(ele / norm_num) for ele in data]

    # Get the frame averages of the fft transforms and return these as a list
    frame_averages = [abs(np.mean(fft.fft(norm_data[i*num_samples:i*num_samples+(num_samples-1)]))) for i in range(0, num_frames)]
    
    num_silence_frames = len([x for x in frame_averages if x < tolerance])

    s_ratio = 100-((num_silence_frames/num_frames)*100)
    return s_ratio
