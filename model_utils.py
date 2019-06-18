import tensorflow as tf


def create_conv_layer(name, filters, in_shape=None, size=5, stride=1, padding='same', activation='relu'):
    layer_name = 'Conv' + str(name) + '-' + str(size) + 'x' + str(size) + 'x' + str(filters) + '-' + str(stride)

    if in_shape is None:
        return tf.keras.layers.Conv2D(filters=filters,
                                      kernel_size=size,
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
