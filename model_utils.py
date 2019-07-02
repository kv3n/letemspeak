import tensorflow as tf

from audio_processor import break_complex_spectrogram, apply_mask


def create_conv_layer(name, filters, dilation=(1, 1), in_shape=None, size=(5, 5), stride=(1, 1), padding='same', activation='relu'):
    layer_name = 'Conv' + str(name)  # + '-' + str(size) + 'x' + str(size) + 'x' + str(filters) + '-' + str(stride)

    if in_shape is None:
        return tf.keras.layers.Conv2D(filters=filters,
                                      kernel_size=size,
                                      dilation_rate=dilation,
                                      strides=stride,
                                      padding=padding,
                                      activation=activation,
                                      use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer='glorot_uniform',
                                      name=layer_name)
    else:
        return tf.keras.layers.Conv2D(filters=filters,
                                      kernel_size=size,
                                      dilation_rate=dilation,
                                      strides=stride,
                                      padding=padding,
                                      activation=activation,
                                      use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer='glorot_uniform',
                                      input_shape=in_shape,
                                      name=layer_name)


def create_deconv_layer(name, filters, size=5, stride=1, padding='same'):
    layer_name = 'DeConv' + str(name) + '-' + str(size) + 'x' + str(size) + 'x' + str(filters) + '-' + str(stride)

    return tf.keras.layers.Conv2DTranspose(filters=filters,
                                           kernel_size=size,
                                           strides=stride,
                                           padding=padding,
                                           use_bias=True,
                                           kernel_initializer='glorot_uniform',
                                           bias_initializer='glorot_uniform',
                                           name=layer_name)


def create_pooling_layer(name, size=2, stride=2, padding='same'):
    layer_name = 'Pool' + str(name) + '-' + str(size) + 'x' + str(size) + '-' + str(stride)
    return tf.keras.layers.MaxPooling2D(pool_size=size,
                                        strides=stride,
                                        padding=padding,
                                        name=layer_name)


def video_dilation_network():
    video_network = tf.keras.Sequential([
        create_conv_layer(name='Vid01', filters=256, size=(7, 1), dilation=(1, 1)),
        create_conv_layer(name='Vid02', filters=256, size=(5, 1), dilation=(1, 1)),
        create_conv_layer(name='Vid03', filters=256, size=(5, 1), dilation=(2, 1)),
        create_conv_layer(name='Vid04', filters=256, size=(5, 1), dilation=(4, 1)),
        create_conv_layer(name='Vid05', filters=256, size=(5, 1), dilation=(8, 1)),
        create_conv_layer(name='Vid06', filters=256, size=(5, 1), dilation=(16, 1)),

        # Upsampling to match audio
        tf.keras.layers.UpSampling2D(name='VidUpsample', size=(4, 1), interpolation='nearest'),
        tf.keras.layers.Cropping2D(cropping=((1, 1), (0, 0))),

        tf.keras.layers.Permute((1, 3, 2)),
        tf.keras.layers.Reshape((298, 256))
    ])

    return video_network


def audio_dilation_network():
    audio_network = tf.keras.Sequential([
        create_conv_layer(name='Aud01', filters=96, size=(1, 7), dilation=(1, 1)),
        create_conv_layer(name='Aud02', filters=96, size=(5, 1), dilation=(1, 1)),
        create_conv_layer(name='Aud03', filters=96, size=(5, 5), dilation=(1, 1)),
        create_conv_layer(name='Aud04', filters=96, size=(5, 5), dilation=(2, 1)),
        create_conv_layer(name='Aud05', filters=96, size=(5, 5), dilation=(4, 1)),
        create_conv_layer(name='Aud06', filters=96, size=(5, 5), dilation=(8, 1)),
        create_conv_layer(name='Aud07', filters=96, size=(5, 5), dilation=(16, 1)),
        create_conv_layer(name='Aud08', filters=96, size=(5, 5), dilation=(32, 1)),
        create_conv_layer(name='Aud09', filters=96, size=(5, 5), dilation=(1, 1)),
        create_conv_layer(name='Aud10', filters=96, size=(5, 5), dilation=(2, 2)),
        create_conv_layer(name='Aud11', filters=96, size=(5, 5), dilation=(4, 4)),
        create_conv_layer(name='Aud12', filters=96, size=(5, 5), dilation=(8, 8)),
        create_conv_layer(name='Aud13', filters=96, size=(5, 5), dilation=(16, 16)),
        create_conv_layer(name='Aud14', filters=96, size=(5, 5), dilation=(32, 32)),
        create_conv_layer(name='Aud15', filters=8, size=(1, 1), dilation=(1, 1)),

        tf.keras.layers.Reshape((298, 1, 257 * 8)),  # Line up all elements across the channels,
        tf.keras.layers.Permute((1, 3, 2)),
        tf.keras.layers.Reshape((298, 257 * 8))
    ])

    return audio_network


def power_loss(true_spectrogram, mask_spectrogram):
    true_spectrogram = tf.reshape(true_spectrogram, (-1, 298, 257, 2))
    mask_spectrogram = tf.reshape(mask_spectrogram, (-1, 298, 257, 2))

    reconstructed_spectrogram = break_complex_spectrogram(apply_mask(true_spectrogram, mask_spectrogram))

    loss = tf.math.sqrt(tf.nn.l2_loss(reconstructed_spectrogram[:, :, :] - true_spectrogram[:, :, :]))
    return loss


def build_output_functor(model):
    # with a Sequential model
    get_output = tf.keras.backend.function([model.layers[0].input, model.layers[1].input],
                                           [model.layers[-1].output])
    return get_output


def stitch_model():
    video_input = tf.keras.layers.Input(shape=[134400])
    audio_input = tf.keras.layers.Input(shape=[153172])

    video_stream = video_dilation_network()
    audio_stream = audio_dilation_network()

    fusion = tf.keras.layers.Concatenate(axis=2)
    fusion = fusion([video_stream(video_input), audio_stream(audio_input)])

    bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100))(fusion)

    fc1 = tf.keras.layers.Dense(200)(bidirectional)
    fc2 = tf.keras.layers.Dense(200)(fc1)
    fc3 = tf.keras.layers.Dense(298*257*2)(fc2)

    final_model = tf.keras.Model(inputs=[video_input, audio_input], outputs=fc3)

    final_model.compile(loss=power_loss,
                        optimizer='adam',
                        metrics=['accuracy'])

    return final_model


def build_output_functor(model):
    # with a Sequential model
    get_output = tf.keras.backend.function([model.layers[0].input, model.layers[1].input],
                                           [model.layers[-1].output])
    return get_output

"""
USE THIS SPACE FOR TESTING ONLY
"""
def main():
    print("Testing only")

if __name__ == '__main__':
    main()
