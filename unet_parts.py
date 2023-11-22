from tensorflow.keras import layers


def decoder_block(inputs, skip_features, num_filters):
    """ Function to create a decoder block """
    x = layers.UpSampling2D((2, 2))(inputs)
    x = layers.Conv2D(num_filters, (2, 2), padding='same')(x)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def conv_block(inputs, num_filters):
    """ Function to create a convolutional block """
    x = layers.Conv2D(num_filters, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x
