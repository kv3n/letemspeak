import time

import tensorflow as tf

from data_feed import DatasetIterator
from model_utils import stitch_model, build_output_functor
from video_processor import save_val_results


def commit_to_log(log, results, step):
    with log.as_default():
        tf.summary.scalar('loss', results[0], step=step)
        tf.summary.scalar('accuracy', results[1], step=step)


def main():
    # Logging and Output setup
    run_time = time.time()
    output_dir = 'run_{}'.format(run_time)
    training_log = tf.summary.create_file_writer('logs/{}_train/'.format(run_time))
    validation_log = tf.summary.create_file_writer('logs/{}_val/'.format(run_time))

    # The model
    video_input = tf.keras.layers.Input(shape=[75, 1, 1792])
    audio_input = tf.keras.layers.Input(shape=[298, 257, 2])
    letemspeak_network = stitch_model(inputs=[video_input, audio_input])
    audio_output = build_output_functor(letemspeak_network)

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

        commit_to_log(training_log, train_result, data_iter.global_step-1)

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

            commit_to_log(validation_log, val_results, data_iter.validation_step-1)

            print('Ran Validation: ' + str(data_iter.validation_step))
            letemspeak_network.save_weights('output/weight_{}.hd5'.format(data_iter.validation_step))

            save_val_results(test_key, test_start, val_outputs, output_dir=output_dir)


if __name__ == '__main__':
    main()
