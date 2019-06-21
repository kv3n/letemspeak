import tensorflow as tf
from scipy.signal import welch


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


def video_dilation_network(input_shape):
    video_network = tf.keras.Sequential([
        create_conv_layer(name='Vid01', filters=256, size=(7, 1), dilation=(1, 1), in_shape=input_shape),
        create_conv_layer(name='Vid02', filters=256, size=(5, 1), dilation=(1, 1)),
        create_conv_layer(name='Vid03', filters=256, size=(5, 1), dilation=(2, 1)),
        create_conv_layer(name='Vid04', filters=256, size=(5, 1), dilation=(4, 1)),
        create_conv_layer(name='Vid05', filters=256, size=(5, 1), dilation=(8, 1)),
        create_conv_layer(name='Vid06', filters=256, size=(5, 1), dilation=(16, 1)),

        # Upsampling to match audio
        tf.keras.layers.UpSampling2D(name='VidUpsample', size=(4, 1), interpolation='nearest')
    ]
    )

    return video_network


def audio_dilation_network(input_shape):
    audio_network = tf.keras.Sequential([
        create_conv_layer(name='Aud01', filters=96, size=(1, 7), dilation=(1, 1), in_shape=input_shape),
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
        create_conv_layer(name='Aud15', filters=8, size=(1, 1), dilation=(1, 1))
    ]
    )

    return audio_network


def power_loss(true_spectrum, prediction_spectrum):
    true_frequency, true_density = welch(true_spectrum)
    prediction_frequency, prediction_density = welch(prediction_spectrum)

    return tf.keras.losses.MSE(true_density, prediction_density)

"""
USE THIS SPACE FOR TESTING ONLY
"""
def main():
    video_stream = video_dilation_network(input_shape=(75, 1, 1790))
    video_stream.summary()

    audio_stream = audio_dilation_network(input_shape=(300, 257, 2))
    audio_stream.summary()


if __name__ == '__main__':
    main()
