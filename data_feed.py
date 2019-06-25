import json
import random
import math

import numpy as np
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
        start = math.floor(random.uniform(0.0, video.duration - 3.0))
        start = 0.0
        end = start + 3.0

        video = video.subclip(start, end).set_fps(25)
        audio = video.audio.set_fps(16000)
        waveform = audio.to_soundarray()[:, 0]  # We only care about the left channel

        frames = [frame for frame in video.iter_frames(with_times=False)]

        assert(len(frames) == 75)

        return start, frames, waveform

    def request_sample(self):
        sample_key = self.samples[random.randint(0, len(self.samples)-1)]
        # sample_key = '1yo45HeVCDE_26'
        # sample_key = '1yo45HeVCDE_17'
        # sample_key = 'tNdxD5kcvjU_0'
        start, frames, waveform = self.build_sample(sample_key)

        print('Picked Key: {} at start {}'.format(sample_key, start))

        true_speaker_location = self.feed_dict[sample_key] if self.should_filter_ground_truth else None
        waveform = process_audio(waveform)
        embeddings = process_video(frames=frames, ground_truth=true_speaker_location)

        samples = []
        if embeddings is not None:
            for embedding in embeddings:
                samples.append((np.expand_dims(embedding, axis=0), np.expand_dims(waveform, axis=0)))

        return sample_key, start, samples


class DatasetIterator:
    def __init__(self, num_validations=1000, val_interval=500):
        self.training_feed = Feed('data/train', filter_ground_truth=True)
        self.testing_feed = Feed('data/test', filter_ground_truth=False)

        self.global_step = 0
        self.validation_step = 0

        self.total_iterations = num_validations * val_interval
        self.val_interval = val_interval

    def get_batch_feed(self, data_type):
        if data_type == 1:
            sample_key, start, samples = self.training_feed.request_sample()
            while not samples:
                sample_key, start, samples = self.training_feed.request_sample()
        else:
            sample_key, start, samples = self.testing_feed.request_sample()
            while not samples:
                sample_key, start, samples = self.testing_feed.request_sample()

        return sample_key, start, samples

    def step_train(self):
        self.global_step += 1
        self.validation_step = self.global_step // self.val_interval

        run_validation = (self.global_step % self.val_interval == 0)
        end_of_training = (self.global_step == self.total_iterations)

        return run_validation, end_of_training



"""
THIS SHOULD BE USED FOR TESTING ONLY
"""
def main():
    """ Main method """
    training_feed = Feed('data/train', filter_ground_truth=True)
    #testing_feed = Feed('data/test', filter_ground_truth=False)

    key, start, sample = training_feed.request_sample()

    """
    key, frames, wavefrom = training_feed.request_sample()
    print('Train Requested {}: got {} frames of video and {} timeslices of audio'.format(key, len(frames), wavefrom.shape))

    key, frames, wavefrom = testing_feed.request_sample()
    print('Test Requested {}: got {} frames of video and {} timeslices of audio'.format(key, len(frames), wavefrom.shape))
    """


if __name__ == '__main__':
    main()
