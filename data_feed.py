import numpy as np
import json
import random

from moviepy.editor import VideoFileClip

from video_processor import process_video
from audio_processor import process_audio


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

        return frames, waveform

    def request_sample(self):
        sample_key = self.samples[random.randint(0, len(self.samples)-1)]
        sample_key = '1yo45HeVCDE_26'
        print('Picked Key: {}'.format(sample_key))

        frames, wavefrom = self.build_sample(sample_key)

        true_speaker_location = self.feed_dict[sample_key] if self.should_filter_ground_truth else None
        frames = process_video(frames=frames, ground_truth=true_speaker_location)
        wavefrom = process_audio(wavefrom)

        return sample_key, frames, wavefrom


"""
THIS SHOULD BE USED FOR TESTING ONLY
"""
def main():
    """ Main method """
    training_feed = Feed('data/train', filter_ground_truth=True)
    testing_feed = Feed('data/test', filter_ground_truth=False)

    key, frames, wavefrom = training_feed.request_sample()
    print('Got {} frames of video and {} timeslices of audio'.format(len(frames), wavefrom.shape))

    """
    key, frames, wavefrom = training_feed.request_sample()
    print('Train Requested {}: got {} frames of video and {} timeslices of audio'.format(key, len(frames), wavefrom.shape))

    key, frames, wavefrom = testing_feed.request_sample()
    print('Test Requested {}: got {} frames of video and {} timeslices of audio'.format(key, len(frames), wavefrom.shape))
    """


if __name__ == '__main__':
    main()
