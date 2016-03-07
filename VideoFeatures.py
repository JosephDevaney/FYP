import WnLFeatures as wnl
import pickle as pkl


class VideoFeatures:
    def __init__(self, rate, data, filename):
        self.filename = filename

        shp = data.shape

        if len(shp) > 1 and shp[0] > shp[1]:
            data = data.T

        data = wnl.lib_to_mono(data)
        data = data[:rate*30]

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

    def get_category_from_name(self):
        return self.filename[:self.filename.index('_')]

    def get_length_from_name(self):
        return self.filename[self.filename.rindex('_')+1:self.filename.rindex('.wav')]

    def get_windowed_fft(self, block_length):
        return wnl.get_windowed_fft(self.data, block_length)
