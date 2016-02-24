import os
import scipy.io.wavfile as wav
import pickle as pkl
from VideoFeatures import VideoFeatures


def analyse_features():
    path = input("Enter the filepath here: \n")
    videos = {}
    try:
        with open(path + "features.ftr", "rb") as inp:
            unpickle = pkl.Unpickler(inp)
            while True:
                try:
                    vid = unpickle.load()
                    cat = vid.get_category_from_name()
                    len = vid.get_length_from_name()
                    if cat in videos:
                        videos[cat][0] += 1
                        videos[cat][1] += int(len)
                    else:
                        videos[cat] = [1, int(len)]
                except EOFError:
                    print("EOF")
                    break
    except Exception as e:
        print(type(e))
        print(e.args)

    for cat, val in videos.items():
        print(cat + "\t|\t" + str(val[0]) + "\t|\t" + str(int(val[1]) / val[0]))


def main():
    analyse_features()

if __name__ == "__main__":
    main()

    # D:\Documents\DT228_4\FYP\Datasets\080327\0_Audio\Autos & Vehicles_7n3jD-kxb1U_310.wav