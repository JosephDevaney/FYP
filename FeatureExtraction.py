import os
import scipy.io.wavfile as wav
import pickle as pkl
from VideoFeatures import VideoFeatures

FTR_NAME = "features30sec.ftr"


# Checks for existing Features file and loads the filename of any existing objects. These will not be duplicated
# Read all .wav files and get the filename, rate and data for each
# This information is passed to the constructor of a VideoFeatures object to calculate all features.
#  Each object is then pickled to the Features file
def main():
    path = input("Enter the filepath here: \n")
    videos = {}
    try:
        with open(path + FTR_NAME, "rb") as inp:
            unpickle = pkl.Unpickler(inp)
            while True:
                try:
                    vid = unpickle.load()
                    videos[vid.filename] = vid.rate
                except EOFError:
                    print("EOF")
                    break
    except Exception as e:
        print(type(e))
        print(e.args)

    cls = ["Entertainment", "Music", "Comedy", "Film & Animation", "News & Politics", "Sports", "People & Blogs",
           "Howto & Style", "Pets & Animals"]
    # rate, data = wav.read(file)
    with open(path + FTR_NAME, "ab") as output:
        pickler = pkl.Pickler(output, -1)
        # [videos.update({file: VideoFeatures(wav.read(path + file), file).write_to_file(pickler)})
        #  for file in os.listdir(path) if file.endswith('.wav') and file not in videos]
        for file in os.listdir(path):
            if file.endswith('.wav') and file not in videos:
                cat = file[:file.index('_')]
                if cat in cls:
                    rate, data = wav.read(path + file)
                    vid = VideoFeatures(rate, data, file)
                    vid.write_to_file(pickler)
                    videos[file] = vid.rate

    print("Dunzo!")


if __name__ == "__main__":
    main()

# D:\Documents\DT228_4\FYP\Datasets\080327\0_Audio\Autos & Vehicles_7n3jD-kxb1U_310.wav
