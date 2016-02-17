import youtube_dl
import numpy as np


YT_PREFIX = "https://www.youtube.com/watch?v="


def download(v_id, cat, time, dir, options):
    print("Starting Download of " + v_id + "***")
    options['outtmpl'] = dir + '\\' + cat + '_' + v_id + '_' + time + '.%(ext)s'
    try:
        with youtube_dl.YoutubeDL(options) as ydl:
            ydl.download([v_id])
        print("***Download Complete***")
    except:
        print("***Unable to Download***")


def main():
    datafile = input("Please enter the location of the datafile: \n")
    videodata = [[val for val in line.split('\t')] for line in open(datafile) if line]

    videolinks = np.array([[v[0], v[3], v[4]] for v in videodata if len(v) > 4])

    savedir = input("Please enter the directory that will store the downloads: \n")
    yt_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }]
    }

    for video in videolinks:
        download(video[0], video[1], video[2], savedir, yt_opts)


if __name__ == "__main__":
    main()
# D:\Documents\DT228_4\FYP\Datasets\080327\0.txt
