import numpy as np
from os import listdir

from moviepy.editor import VideoFileClip


class Feed:
    def __init__(self, feed_dir):
        self.audio_sample = []
        self.video_sample = []

        for sample in listdir(feed_dir):
            filename = '{}/{}'.format(feed_dir, sample)

            video = VideoFileClip(filename).set_fps(25)
            audio = video.audio.set_fps(16000)
            audio = audio.to_soundarray()[:, 0]  # We only care about the left channel

            frames = []
            for t, frame in video.iter_frames(with_times=True):
                frames.append(frame)

            frames = np.array(frames)

            self.video_sample.append(frames)
            self.audio_sample.append(audio)

        print('Collected {} audio clips and {} video clips'.format(len(self.audio_sample), len(self.video_sample)))


def main():
    """ Main method """
    training_feed = Feed('data/train')


if __name__ == '__main__':
    main()
