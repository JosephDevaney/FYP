import os
import scipy.io.wavfile as wav
import WnLFeatures as wnl


class VideoFeatures:
    def __init__(self, rate, data):
        self.rate = rate
        self.data = data.T[0]

        self.bvratio = wnl.beat_variance_ratio(rate, data)

        self.silence_ratio = wnl.silence_ratio(rate, data)

        self.mfcc, self.mfcc_delta = wnl.librosa_mfcc_delta(rate, data)

        self.chromagram = wnl.librosa_chromagram(rate, data)

        self.spectroid = wnl.spec_centroid(rate, data)


def main():
    file = input("Enter the filepath here: \n")

    rate, data = wav.read(file)



if __name__ == "__main__":
    main()

    # D:\Documents\DT228_4\FYP\Datasets\080327\0_Audio\Autos & Vehicles_7n3jD-kxb1U_310.wav