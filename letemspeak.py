from data_feed import DatasetIterator
from model_utils import stitch_model

import tensorflow as tf


def main():
    # The model
    video_input = tf.keras.layers.Input(shape=[75, 1, 1792])
    audio_input = tf.keras.layers.Input(shape=[298, 257, 2])
    letemspeak_network = stitch_model(inputs=[video_input, audio_input])

    # The data
    data_iter = DatasetIterator(num_validations=1000, val_interval=500)

    end_of_training = False
    while not end_of_training:
        # Run mini-batch
        train_key, train_start, train_sample = data_iter.get_batch_feed(data_type=1)

        print('Training on {}'.format(train_key))

        train_result = letemspeak_network.train_on_batch([train_sample[0][0], train_sample[0][1]], train_sample[0][1])
        run_validation, end_of_training = data_iter.step_train()
        print('Ran Batch: ' + str(data_iter.global_step))

        if run_validation:
            val_results = []
            test_key, test_start, test_sample = data_iter.get_batch_feed(data_type=2)
            print('Validating on {}'.format(test_key))

            for speaker in test_sample:
                result = letemspeak_network.test_on_batch([speaker[0], speaker[1]], speaker[1])
                val_results.append(result)

            print('Ran Validation: ' + str(data_iter.validation_step))
            letemspeak_network.save_weights('output/weight_{}.hd5'.format(data_iter.validation_step))


if __name__ == '__main__':
    main()
