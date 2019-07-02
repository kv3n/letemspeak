import time
import os
import json

import tensorflow as tf
from numpy import np

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
    keras_loop()


def keras_loop():
    run_time = time.time()
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(run_time))
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='output/{}'.format(run_time),
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    load_weights_on_restart=True)

    letemspeak_network = stitch_model()
    video_input_slices = np.loadtxt('data/train/x1.csv', dtype=np.float32, delimiter=',')
    audio_input_slices = np.loadtxt('data/train/x2.csv', dtype=np.float32, delimiter=',')

    print('Training on {} samples'.format(video_input_slices.shape[0]))

    letemspeak_network.fit([video_input_slices, audio_input_slices], audio_input_slices,
                           validation_split=0.2, epochs=150, batch_size=4,
                           callbacks=[tensorboard, checkpoint])


def custom_training_loop():
    run_time = time.time()
    output_dir = 'run_{}'.format(run_time)
    training_log = tf.summary.create_file_writer('logs/{}_train/'.format(run_time))
    validation_log = tf.summary.create_file_writer('logs/{}_val/'.format(run_time))

    # The model
    letemspeak_network = stitch_model()
    audio_output = build_output_functor(letemspeak_network)

    model_accuracy = load_model_weights(letemspeak_network)

    # The data
    data_iter = DatasetIterator(num_validations=500, val_interval=1000)
    end_of_training = False
    while not end_of_training:
        # Run mini-batch
        train_key, train_start, train_sample = data_iter.get_batch_feed(data_type=1)

        print('Training on {}'.format(train_key))
        train_result = letemspeak_network.train_on_batch([train_sample[0][0], train_sample[0][1]], train_sample[0][1])
        run_validation, end_of_training = data_iter.step_train()
        print('Ran Batch: {} -> {}'.format(data_iter.global_step, train_result))
        commit_to_log(training_log, train_result, data_iter.global_step - 1)

        if run_validation:
            val_outputs = []
            val_results = []

            test_key, test_start, test_sample = data_iter.get_batch_feed(data_type=2)
            print('Validating on {}'.format(test_key))
            for speaker in test_sample:
                result = letemspeak_network.test_on_batch([speaker[0], speaker[1]], speaker[1])
                output = audio_output([speaker[0], speaker[1]])[0]
                val_outputs.append(output)
                if not val_results:
                    val_results = [metric for metric in result]
                else:
                    for idx, _ in enumerate(val_results):
                        val_results[idx] += result[idx]

            for idx, _ in enumerate(val_results):
                val_results[idx] = val_results[idx] / len(test_sample)
            commit_to_log(validation_log, val_results, data_iter.validation_step - 1)

            print('Ran Validation: ' + str(data_iter.validation_step))
            if val_results[1] > model_accuracy:
                model_accuracy = val_results[1]
                letemspeak_network.save_weights('output/weights.h5')
                model_meta = {
                    'accuracy': val_results[1],
                    'loss': val_results[0],
                    'file_loc': 'output/weights.h5'
                }
                with open('output/meta.json', 'w') as fp:
                    fp.write(json.dumps(model_meta))

            # save_val_results(test_key, test_start, val_outputs, output_dir=output_dir)


if __name__ == '__main__':
    main()
