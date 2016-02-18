import os
import scipy.io.wavfile as wav
import WnLFeatures as wnl
import pickle as pkl


class VideoFeatures:
    def __init__(self, rate, data, filename):
        self.filename = filename

        # self.rate = file_data[0]
        # self.data = file_data[1]
        # rate = file_data[0]
        # data = file_data[1]
        #
        # data = data[0]
        data = data.T[0]
        self.rate = rate
        self.data = data

        self.bvratio = wnl.beat_variance_ratio(rate, data)

        self.silence_ratio = wnl.silence_ratio(rate, data)

        self.mfcc, self.mfcc_delta = wnl.librosa_mfcc_delta(rate, data)

        self.chromagram = wnl.librosa_chromagram(rate, data)

        self.spectroid = wnl.spec_centroid(rate, data)

    def write_to_file(self, pickler=None):
        print("***Writing {0} to disk***".format(self.filename))
        if pickler is None:
            file = "features.ftr"
            with open(file, 'a+') as output:
                pkl.dump(self, output, -1)
        else:
            pickler.dump(self)
        print("Finished writing {0} to file".format(self.filename))


def main():
    path = input("Enter the filepath here: \n")
    videos = {}
    try:
        with open(path + "features.ftr", "rb") as inp:
            while True:
                try:
                    vid = pkl.load(inp)
                    videos[vid.filename] = vid
                except:
                    print("EOF")
                    break
    except:
        print("No file")

    # rate, data = wav.read(file)
    with open(path + "features.ftr", "ab") as output:
        pickler = pkl.Pickler(output, -1)
        # [videos.update({file: VideoFeatures(wav.read(path + file), file).write_to_file(pickler)})
        #  for file in os.listdir(path) if file.endswith('.wav') and file not in videos]
        for file in os.listdir(path):
            if file.endswith('.wav') and file not in videos:
                rate, data = wav.read(path + file)
                vid = VideoFeatures(rate, data, file)
                vid.write_to_file(pickler)
                videos[file] = vid



    print("Dunzo!")


if __name__ == "__main__":
    main()

    # D:\Documents\DT228_4\FYP\Datasets\080327\0_Audio\Autos & Vehicles_7n3jD-kxb1U_310.wav