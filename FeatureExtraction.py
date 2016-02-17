import os
import scipy.io.wavfile as wav
import WnLFeatures as wnl
import pickle as pkl


class VideoFeatures:
    def __init__(self, file_data):
        self.rate = file_data[0]
        self.data = file_data[1].T[0]
        rate = file_data[0]
        data = file_data[1].T[0]

        self.bvratio = wnl.beat_variance_ratio(rate, data)

        self.silence_ratio = wnl.silence_ratio(rate, data)

        self.mfcc, self.mfcc_delta = wnl.librosa_mfcc_delta(rate, data)

        self.chromagram = wnl.librosa_chromagram(rate, data)

        self.spectroid = wnl.spec_centroid(rate, data)

    def write_to_file(self, pickler=None):
        if pickler is None:
            file = "features.ftr"
            with open(file) as output:
                pkl.dump(self, output, -1)
        else:
            pickler.dump(self)

def main():
    path = input("Enter the filepath here: \n")

    # rate, data = wav.read(file)
    with open(path + "\\features.ftr", "wb+") as output:
        pickler = pkl.Pickler(output, -1)
        videos = [VideoFeatures(wav.read(path + file)).write_to_file(pickler) for file in os.listdir(path)
                  if file.endswith('.wav')]


if __name__ == "__main__":
    main()

    # D:\Documents\DT228_4\FYP\Datasets\080327\0_Audio\Autos & Vehicles_7n3jD-kxb1U_310.wav