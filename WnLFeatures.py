import librosa as lib
import numpy as np
import math
import scipy.fftpack as fft


def beat_variance_ratio(rate, data, tolerance=0.1):
    try:
        tempo, beat_frames = lib.beat.beat_track(y=data, sr=rate)

        if len(beat_frames) == 0:
            return -1

        beat_times = lib.frames_to_time(beat_frames, sr=rate)

        beat_diffs = np.diff(beat_times)

        mean_diff = np.mean(beat_diffs)
        upper_lim = mean_diff * 1 + tolerance
        lower_lim = mean_diff * 1 - tolerance

        outside_lim = len([x for x in beat_diffs if lower_lim < x < upper_lim])
        beat_ratio = 100 - ((outside_lim / len(beat_diffs)) * 100)

        return beat_ratio
    except ZeroDivisionError:
        return -1


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


def librosa_mfcc_delta(rate, data):
    mfcc = lib.feature.mfcc(data, rate)

    mfcc_delta = lib.feature.delta(mfcc)

    return mfcc, mfcc_delta


def librosa_chromagram(rate, data):
    stft = lib.core.stft(data)
    decom_h, decom_p = lib.decompose.hpss(stft)
    istft_h = lib.core.istft(decom_h)
    istft_p = lib.core.istft(decom_p)

    return lib.feature.chroma_cqt(y=istft_h, sr=rate)


def spec_centroid(rate, data):
    return lib.feature.spectral_centroid(y=data, sr = rate)


def lib_to_mono(data):
    return lib.to_mono(data)


def get_windowed_fft(data, block_length):
    # how many blocks have to be processed?
    num_blocks = int(np.ceil(len(data) / block_length))

    w_fft = []

    if num_blocks == 1:
        w_fft = fft.rfft(data)
    else:
        for i in range(0, num_blocks):
            start = i * block_length
            stop = np.min([(start + block_length - 1), len(data)])

            f = fft.rfft(data[start:stop])
            w_fft.append(f.var())

    return np.asarray(w_fft)


def get_windowed_zcr(data, block_length):
    num_blocks = int(np.ceil(len(data) / block_length))

    w_zcr = []

    for i in range (0, num_blocks - 1):
        start = i * block_length
        stop = np.min([(start + block_length - 1), len(data)])

        zcr = lib.zero_crossings(data[start:stop])
        w_zcr.append(len(zcr))

    return np.asarray(w_zcr)












