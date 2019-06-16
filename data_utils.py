import youtube_dl
from moviepy.editor import VideoFileClip

import os
import shutil
import csv
import random
import ssl


def prep_directory():
    def make_dir(dir_name):
        if os.path.exists(dir_name):
            os.rename(dir_name, dir_name + '_backup')
            shutil.rmtree(dir_name + '_backup')

        os.mkdir(dir_name)

    make_dir('data')
    make_dir('data/train')
    make_dir('data/test')
    make_dir('output')


def download_data(filename, output_dir, limit=0):
    output_dir = output_dir.rstrip('/')

    clip_dictionary = {}
    with open(filename) as fp:
        reader = csv.reader(fp, delimiter=',')
        for row in reader:
            if row[0] not in clip_dictionary:
                clip_dictionary[row[0]] = []

            clip_dictionary[row[0]].append((float(row[1]), float(row[2])))

    if limit:
        limited_keys = random.sample(list(clip_dictionary.keys()), limit)
        clip_dictionary = {key: clip_dictionary[key] for key in limited_keys}

    for clip_id, params in clip_dictionary.items():
        download_opts = {
            'format': 'mp4',
            'outtmpl': 'data/{}.mp4'.format(clip_id),
            'nocheckcertificate': True
        }
        try:
            with youtube_dl.YoutubeDL(download_opts) as ydl:
                ydl.download(['https://www.youtube.com/watch?v={}'.format(clip_id)])

            main_clip = VideoFileClip('data/{}.mp4'.format(clip_id))

            for idx, param in enumerate(params):
                sub_clip = main_clip.subclip(param[0], param[1])
                print(type(sub_clip))
                sub_clip.write_videofile('{}/{}_{}.mp4'.format(output_dir, clip_id, idx), codec="libx264", audio_codec="aac")
                sub_clip.close()

            main_clip.close()
            os.remove('data/{}.mp4'.format(clip_id))
        except:
            print('Could not download 1 video')


def main():
    prep_directory()

    if not os.path.exists('avspeech_train.csv') or not os.path.exists('avspeech_test.csv'):
        print('No training and Testing data found')
    else:
        download_data(filename='avspeech_train.csv', output_dir='data/train', limit=10)
        download_data(filename='avspeech_test.csv', output_dir='data/test', limit=2)


if __name__ == "__main__":
    main()
