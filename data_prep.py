import json
import os
import csv

import numpy as np
from moviepy.editor import VideoFileClip

from video_processor import process_video
from audio_processor import process_audio
from data_utils import build_meta


def remake_meta(data_dir, filename):
    clip_dictionary = {}
    with open(filename) as fp:
        reader = csv.reader(fp, delimiter=',')
        for row in reader:
            if row[0] not in clip_dictionary:
                clip_dictionary[row[0]] = []

            clip_dictionary[row[0]].append((float(row[1]), float(row[2]), float(row[3]), float(row[4])))

    build_meta(data_dir, clip_dictionary)


def build_sample_slices(feed_dir, speaker_key):
    filename = '{}/{}.mp4'.format(feed_dir, speaker_key)

    video = VideoFileClip(filename)
    duration = video.duration
    video.close()

    sample_video_slices = []
    sample_audio_slices = []

    end = 3.0
    while end < duration:
        start = end - 3.0
        video = VideoFileClip(filename)
        sub_video = video.subclip(start, end)

        frames = [frame for frame in sub_video.iter_frames(with_times=False, fps=25)]
        sample_video_slices.append(frames)

        # For debugging only
        # sub_video.write_videofile('test{}.mp4'.format(int(start)), codec="libx264", audio_codec="aac")

        waveform = sub_video.audio.to_soundarray(fps=16000)[:, 0]  # We only care about the left channel
        sample_audio_slices.append(waveform)

        sub_video.close()
        video.close()
        end += 2.0

    os.remove(filename)

    return sample_video_slices, sample_audio_slices


def save_slice(data_dir, filename, data, mode):
    with open('{}/{}.csv'.format(data_dir, filename), mode) as fp:
        if isinstance(data, np.ndarray):
            np.savetxt(fp, data, delimiter=',')
        else:
            fp.writelines(data)


def build_sample_file(data_dir, sample, true_speaker_location, mode):
    video_slices = []
    audio_slices = []
    slices_key = []

    sample_video_slices, sample_audio_slices = build_sample_slices(data_dir, sample)

    num_slices = len(sample_audio_slices)
    slices_made = 0
    for slice_idx, _ in enumerate(sample_audio_slices):
        waveform = process_audio(sample_audio_slices[slice_idx])
        assert(waveform.shape[0] == 298 and waveform.shape[1] == 257 and waveform.shape[2] == 2)
        embeddings = process_video(frames=sample_video_slices[slice_idx], ground_truth=true_speaker_location)
        for speaker_idx, embedding in enumerate(embeddings):
            # Wavefrom -- shape 298, 257, 2
            audio_slices.append(waveform.numpy().flatten())
            # Emedding shape -- shape 75, 1, 1792
            assert (embedding.shape[0] == 75 and embedding.shape[1] == 1 and embedding.shape[2] == 1792)
            video_slices.append(embedding.flatten())

            slice_key = '{}_slice{}_speaker{}\n'.format(sample, slice_idx, speaker_idx)
            slices_key.append(slice_key)
            slices_made += 1
        print('-----Completed Pre-processing {} of {}'.format(slice_idx + 1, num_slices))

    save_slice(data_dir, 'video', np.array(video_slices), mode)
    save_slice(data_dir, 'audio', np.array(audio_slices), mode)
    save_slice(data_dir, 'label', slices_key, mode)


def iter_samples(data_dir, should_filter_ground_truth):
    with open('{}/meta.json'.format(data_dir), 'r') as fp:
        feed_dict = json.load(fp)
        samples = list(feed_dict.keys())

    sample_count = len(samples)
    for idx, sample in enumerate(samples):
        true_speaker_location = feed_dict[sample] if should_filter_ground_truth else None
        mode = 'w' if idx == 0 else 'a'

        print('Making Sample {} -> {} of {}'.format(sample, idx+1, sample_count))
        build_sample_file(data_dir, sample, true_speaker_location, mode)
        print('----Prepped sample')


def main():
    print('Prepping Training Data')
    iter_samples('data/train', should_filter_ground_truth=True)

    # print('Prepping Testing Data')
    # iter_samples('data/test', should_filter_ground_truth=True)


if __name__ == '__main__':
    main()