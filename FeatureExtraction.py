import os
import scipy.io.wavfile as wav
import pickle as pkl
from VideoFeatures import VideoFeatures


def main():
    path = input("Enter the filepath here: \n")
    videos = {}
    try:
        with open(path + "features.ftr", "rb") as inp:
            unpickle = pkl.Unpickler(inp)
            while True:
                try:
                    vid = unpickle.load()
                    videos[vid.filename] = vid
                except EOFError:
                    print("EOF")
                    break
    except Exception as e:
        print(type(e))
        print(e.args)

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