import time
import os
import json

import numpy as np
import tensorflow as tf

from data_feed import DatasetIterator
from model_utils import stitch_model, build_output_functor
from video_processor import save_val_results


def commit_to_log(log, results, step):
    with log.as_default():
        tf.summary.scalar('loss', results[0], step=step)
        tf.summary.scalar('accuracy', results[1], step=step)


def load_model_weights(model):
    if os.path.exists('output/weights.h5'):
        model.load_weights('output/weights.h5')

    accuracy = 0.0
    if os.path.exists('output/meta.json'):
        with open('output/meta.json', 'r') as fp:
            model_meta = json.load(fp)
            accuracy = model_meta['accuracy']

    return accuracy


def main():
    # Logging and Output setup
    run_time = time.time()
    output_dir = 'run_{}'.format(run_time)
    training_log = tf.summary.create_file_writer('logs/{}_train/'.format(run_time))
    validation_log = tf.summary.create_file_writer('logs/{}_val/'.format(run_time))

    # The model
    letemspeak_network = stitch_model()
    model_accuracy = load_model_weights(letemspeak_network)

    video_input_slices = np.loadtxt('data/train/x1.csv', dtype=np.float32, delimiter=',')
    audio_input_slices = np.loadtxt('data/train/x2.csv', dtype=np.float32, delimiter=',')

    letemspeak_network.fit([video_input_slices, audio_input_slices], audio_input_slices,
                           validation_split=0.2, epochs=150, batch_size=10)

    # letemspeak_network.save_weights('output/weights.h5')
    # model_meta = {
    #     'accuracy': val_results[1],
    #     'loss': val_results[0],
    #     'file_loc': 'output/weights.h5'
    # }
    #
    # with open('output/meta.json', 'w') as fp:
    #     fp.write(json.dumps(model_meta))
    #
    # save_val_results(test_key, test_start, val_outputs, output_dir=output_dir)


if __name__ == '__main__':
    main()
