import json

import numpy as np
from moviepy.editor import VideoFileClip

from video_processor import process_video
from audio_processor import process_audio


def build_sample_slices(feed_dir, speaker_key):
    filename = '{}/{}.mp4'.format(feed_dir, speaker_key)

    video = VideoFileClip(filename).set_fps(25)
    duration = video.duration
    video.close()

    sample_video_slices = []
    sample_audio_slices = []

    end = 3.0
    while end < duration:
        start = end - 3.0
        video = VideoFileClip(filename).set_fps(25)

        video = video.subclip(start, end).set_fps(25)
        frames = [frame for frame in video.iter_frames(with_times=False)]
        sample_video_slices.append(frames)

        audio = video.audio.set_fps(16000)
        waveform = audio.to_soundarray()[:, 0]  # We only care about the left channel
        sample_audio_slices.append(waveform)

        video.close()
        end += 3.0

    return sample_video_slices, sample_audio_slices


def build_sample_file(data_dir, should_filter_ground_truth):
    with open('{}/meta.json'.format(data_dir), 'r') as fp:
        feed_dict = json.load(fp)
        samples = list(feed_dict.keys())

    video_slices = []
    audio_slices = []
    slices_key = []

    num_samples = len(samples)-1
    for idx, sample in enumerate(samples):
        sample_video_slices, sample_audio_slices = build_sample_slices(data_dir, sample)
        print('Built Sample {} of {}'.format(idx, num_samples))

        true_speaker_location = feed_dict[sample] if should_filter_ground_truth else None
        for slice_idx, _ in enumerate(sample_audio_slices):
            waveform = process_audio(sample_audio_slices[slice_idx])
            embeddings = process_video(frames=sample_video_slices[slice_idx], ground_truth=true_speaker_location)
            print('Completed Pre-processing {} of {}'.format(idx, num_samples))
            for speaker_idx, embedding in enumerate(embeddings):
                # Wavefrom -- shape 298, 257, 2
                audio_slices.append(waveform.numpy().flatten())
                # Emedding shape -- shape 75, 1, 1792
                video_slices.append(embedding.flatten())

                slice_key = '{}_slice{}_speaker{}'.format(sample, slice_idx, speaker_idx)
                slices_key.append(slice_key)
        print('Processed sample {} of {}: {}'.format(idx, num_samples, sample))

    video_slices = np.array(video_slices)
    audio_slices = np.array(audio_slices)

    np.savetxt('{}/x1.csv'.format(data_dir), video_slices, delimiter=',')
    np.savetxt('{}/x2.csv'.format(data_dir), audio_slices, delimiter=',')

    with open('{}/label.csv'.format(data_dir), 'w') as fp:
        fp.writelines(slices_key)

    print('Prepped data for training')


def main():
    build_sample_file('data/train', should_filter_ground_truth=True)
    build_sample_file('data/test', should_filter_ground_truth=False)


if __name__ == '__main__':
    main()