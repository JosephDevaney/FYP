import youtube_dl
import numpy as np
import DatasetStats as dss
import os
from VideoFeatures import VideoFeatures


YT_PREFIX = "https://www.youtube.com/watch?v="


def load_classes():
    return [cl.strip('\n') for cl in open("classes.txt") if cl]


def download(v_id, cat, time, dir, options):
    print("Starting Download of " + v_id + "***")
    options['outtmpl'] = dir + '\\' + cat + '_' + v_id + '_' + time + '.%(ext)s'
    options['-w'] = True
    try:
        with youtube_dl.YoutubeDL(options) as ydl:
            ydl.download([v_id])
        print("***Download Complete***")
        return True
    except:
        print("***Unable to Download***")
        return False


def main():
    classes = load_classes()

    datafile = input("Please enter the location of the datafile: \n")
    num_vids_per_cat = int(input("Required number of videos per category : \n"))

    print("Loading statistics from existing dataset")
    stats = dss.analyse_features()
    dss.print_stats(stats)
    # stats = {}
    videodata = [[val for val in line.split('\t')] for line in open(datafile) if line]

    videolinks = np.array([[v[0], v[3], v[4]] for v in videodata if len(v) > 4])

    savedir = input("Please enter the directory that will store the downloads: \n")
    f_names = [f for f in os.listdir(savedir) if f.endswith('.wav')]

    yt_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }]
    }

    for video in videolinks:
        vid_link = video[0]
        vid_cat = video[1]
        vid_len = video[2]

        v_name = vid_cat + '_' + vid_link + '_' + vid_len + '.wav'

        if vid_cat in classes:
            if vid_cat not in stats:
                stats[vid_cat] = [0, 0]

            if stats[vid_cat][0] < num_vids_per_cat and v_name not in f_names and 30 <= int(vid_len) <= 240:
                if download(vid_link, vid_cat, vid_len, savedir, yt_opts):
                    stats[vid_cat][0] += 1
                    stats[vid_cat][1] += int(vid_len)

    dss.print_stats(stats)

if __name__ == "__main__":
    main()
# D:\Documents\DT228_4\FYP\Datasets\080327\1.txt
# D:\Documents\DT228_4\FYP\Datasets\080327\0_Audio
# D:\Documents\DT228_4\FYP\Datasets\080327\0_Audio2
