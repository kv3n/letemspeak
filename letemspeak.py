from data_feed import Feed
from model_utils import stitch_model

import tensorflow as tf


def main():
    training_feed = Feed('data/train', filter_ground_truth=True)
    key, frames, wavefrom = training_feed.request_sample()

    video_input = tf.keras.layers.Input(shape=[75, 1, 1792])
    audio_input = tf.keras.layers.Input(shape=[298, 257, 2])

    model = stitch_model(inputs=[video_input, audio_input])
    model.summary()


if __name__ == '__main__':
    main()