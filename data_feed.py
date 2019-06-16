import numpy as np
from scipy import signal

from moviepy.editor import VideoFileClip


def main():
    """ Main method """
    filename = "data/train/xNXfdJEcvYc_0.mp4"
    video = VideoFileClip(filename)
    audio = video.audio.to_soundarray()
    print(signal.stft(audio)[2].shape)
    


if __name__ == '__main__':
    main()
