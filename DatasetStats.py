import os
import scipy.io.wavfile as wav
import pickle as pkl
from VideoFeatures import VideoFeatures


def analyse_features():
    path = input("Enter the filepath here: \n")
    videos = {}
    try:
        with open(path + "features30sec.ftr", "rb") as inp:
            unpickle = pkl.Unpickler(inp)
            while True:
                try:
                    vid = unpickle.load()
                    cat = vid.get_category_from_name()
                    vid_len = vid.get_length_from_name()
                    if cat in videos:
                        videos[cat][0] += 1
                        videos[cat][1] += int(vid_len)
                    else:
                        videos[cat] = [1, int(vid_len)]
                except EOFError:
                    print("EOF")
                    break
                except TypeError:
                    print("Unable to load object")
                except pkl.UnpicklingError:
                    print("Unable to load object2")
    except Exception as e:
        print(type(e))
        print(e.args)

    return videos


def analyse_videos_file():
    vid_stats = {}
    while True:
        datafile = input("Please enter the location of the datafile: \n")
        # videodata = [[val for val in line.split('\t')] for line in open(datafile) if line]

        with open(datafile) as f:
            for line in f:
                cols = line.split('\t')
                if len(cols) >= 4:
                    vid_link = cols[0]
                    vid_cat = cols[3]
                    vid_len = int(cols[4])

                    if vid_cat in vid_stats:
                        vid_stats[vid_cat][0] += 1
                        vid_stats[vid_cat][1] += vid_len
                    else:
                        vid_stats[vid_cat] = [1, vid_len]

        more_files = input("Press Y(y) to enter another file to be analysed: \n")
        if more_files != 'Y' and more_files != 'y':
            break

    return vid_stats


def print_stats(stats):

    for cat, val in stats.items():
        print(cat + "\t|\t" + str(val[0]) + "\t|\t" + str(val[1] / val[0]))


def main():
    print_stats(analyse_features())

    # print_stats(analyse_videos_file())


if __name__ == "__main__":
    main()

    # D:\Documents\DT228_4\FYP\Datasets\080327\0_Audio\Autos & Vehicles_7n3jD-kxb1U_310.wav
    # D:\Documents\DT228_4\FYP\Datasets\080327\1.txt
