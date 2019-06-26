import youtube_dl
from moviepy.editor import VideoFileClip

import os
import csv
import random
import json


def prep_directory():
    def make_dir(dir_name):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

    make_dir('data')
    make_dir('data/train')
    make_dir('data/test')
    make_dir('output')
    make_dir('logs')


def build_meta(output_dir, clip_dictionary):
    meta_file = '{}/meta.json'.format(output_dir)

    samples_in_meta = set()
    meta_data = dict()
    if os.path.exists(meta_file):
        with open('{}/meta.json'.format(output_dir), 'r') as fp:
            meta_data = json.load(fp)
            samples_in_meta = set(list(meta_data.keys()))

    dir_samples = os.listdir(output_dir)
    sample_in_directory = set([sample[:-4] for sample in dir_samples if not sample == 'meta.json'])
    samples_not_in_meta = sample_in_directory - samples_in_meta

    for sample in samples_not_in_meta:
        sample_split = sample.split('_')
        clip_key = '_'.join(sample_split[0:-1])
        param_key = int(sample_split[-1])
        params = clip_dictionary[clip_key][param_key]
        meta_data[sample] = (params[2], params[3])

    used_clips = set()
    for clip_id, _ in meta_data.items():
        clip_key = '_'.join(clip_id.split('_')[0:-1])
        used_clips.add(clip_key)

    with open('{}/meta.json'.format(output_dir), 'w') as fp:
        fp.write(json.dumps(meta_data))

    return meta_data, used_clips


def download_data(filename, output_dir, limit=0):
    output_dir = output_dir.rstrip('/')

    clip_dictionary = {}
    with open(filename) as fp:
        reader = csv.reader(fp, delimiter=',')
        for row in reader:
            if row[0] not in clip_dictionary:
                clip_dictionary[row[0]] = []

            clip_dictionary[row[0]].append((float(row[1]), float(row[2]), float(row[3]), float(row[4])))

    meta_data, used_clips = build_meta(output_dir, clip_dictionary)

    if limit:
        limit = limit - len(used_clips)
        limited_keys = random.sample(list(set(list(clip_dictionary.keys())) - used_clips), limit)
        clip_dictionary = {key: clip_dictionary[key] for key in limited_keys}

    items_to_download = len(clip_dictionary)
    downloading = 0
    for clip_id, params in clip_dictionary.items():
        downloading += 1
        print(
            '[IMP] Downloading {} of {}'.format(downloading, items_to_download))

        download_opts = {
            'format': 'mp4',
            'outtmpl': 'data/{}.mp4'.format(clip_id),
            'nocheckcertificate': True
        }

        try:
            with youtube_dl.YoutubeDL(download_opts) as ydl:
                ydl.download(['https://www.youtube.com/watch?v={}'.format(clip_id)])

            for idx, param in enumerate(params):
                main_clip = VideoFileClip('data/{}.mp4'.format(clip_id))
                print(
                    '[IMP] {}_{} -> {} to {} from max {}'.format(clip_id, idx, param[0], param[1], main_clip.duration))

                sub_clip = main_clip.subclip(param[0], param[1])

                sub_clip_id = '{}_{}'.format(clip_id, idx)
                sub_clip_file_name = '{}/{}.mp4'.format(output_dir, sub_clip_id)
                sub_clip.write_videofile(sub_clip_file_name, codec="libx264", audio_codec="aac")

                sub_clip.close()
                main_clip.close()

                meta_data[sub_clip_id] = (param[2], param[3])

            os.remove('data/{}.mp4'.format(clip_id))

        except youtube_dl.utils.DownloadError as error:
            print(error)

    with open('{}/meta.json'.format(output_dir), 'w') as fp:
        fp.write(json.dumps(meta_data))


def main():
    prep_directory()

    if not os.path.exists('avspeech_train.csv') or not os.path.exists('avspeech_test.csv'):
        print('No training and Testing data found')
    else:
        download_data(filename='avspeech_train.csv', output_dir='data/train', limit=100)
        download_data(filename='avspeech_test.csv', output_dir='data/test', limit=10)


if __name__ == "__main__":
    main()
