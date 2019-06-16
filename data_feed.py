import numpy as np
import json
import random

from moviepy.editor import VideoFileClip


class Feed:
    def __init__(self, feed_dir, filter_ground_truth):
        self.should_filter_ground_truth = filter_ground_truth
        self.feed_dir = feed_dir.rstrip('/')

        self.samples = []
        self.feed_dict = {}

        self.pre_process_ops = []

        with open('{}/meta.json'.format(feed_dir), 'r') as fp:
            self.feed_dict = json.load(fp)
            self.samples = list(self.feed_dict.keys())

        print('Loaded meta with {} samples'.format(len(self.samples)))

    def build_sample(self, speaker_key):
        filename = '{}/{}.mp4'.format(self.feed_dir, speaker_key)

        video = VideoFileClip(filename).set_fps(25)
        audio = video.audio.set_fps(16000)
        waveform = audio.to_soundarray()[:, 0]  # We only care about the left channel

        frames = [frame for frame in video.iter_frames(with_times=False)]
        frames = np.array(frames)

        return frames, waveform

    def process_audio(self, waveform):
        processed_waveform = np.copy(waveform)
        return processed_waveform

    def process_video(self, frames):
        processed_frames = np.copy(frames)
        return processed_frames

    def filter_ground_truth(self, frames):
        filtered_frames = np.copy(frames)
        return filtered_frames

    def request_sample(self):
        sample_key = self.samples[random.randint(0, len(self.samples))]

        frames, wavefrom = self.build_sample(sample_key)

        frames = self.process_video(frames)
        wavefrom = self.process_audio(wavefrom)

        if self.should_filter_ground_truth:
            frames = self.filter_ground_truth(frames)

        return sample_key, frames, wavefrom


"""
THIS SHOULD BE USED FOR TESTING ONLY
"""
def main():
    """ Main method """
    training_feed = Feed('data/train', filter_ground_truth=True)
    testing_feed = Feed('data/test', filter_ground_truth=False)

    key, frames, wavefrom = training_feed.request_sample()
    print('Train Requested {}: got {} frames of video and {} timeslices of audio'.format(key, frames.shape, wavefrom.shape))

    key, frames, wavefrom = training_feed.request_sample()
    print('Train Requested {}: got {} frames of video and {} timeslices of audio'.format(key, frames.shape, wavefrom.shape))

    key, frames, wavefrom = testing_feed.request_sample()
    print('Test Requested {}: got {} frames of video and {} timeslices of audio'.format(key, frames.shape, wavefrom.shape))


if __name__ == '__main__':
    main()
