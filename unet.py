import tensorflow as tf
from tensorflow.keras import layers, models
from unet_parts import decoder_block
from tensorflow.keras.applications import DenseNet121


def create_unet_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    base_model = DenseNet121(weights="imagenet", include_top=False, input_tensor=inputs)

    for layer in base_model.layers:
        layer.trainable = False

    """ Encoder layers """
    s1 = base_model.layers[0].output
    s2 = base_model.get_layer("conv1/relu").output
    s3 = base_model.get_layer("pool2_relu").output
    s4 = base_model.get_layer("pool3_relu").output

    """ Bridge """
    b1 = base_model.get_layer("pool4_relu").output

    """ Decoder layers """
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')(d4)
    model = models.Model(inputs, outputs)
    return model
