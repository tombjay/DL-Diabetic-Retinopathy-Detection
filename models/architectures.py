import gin
import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization,GlobalAveragePooling2D
from models.layers import fetaure_extractor, vgg_block

    
@gin.configurable
def customModel(input_shape, n_classes, dense_units, dropout_rate):
    """Defines a custom architecture.
    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        dense_units (int): number of dense units
    Returns:
        (keras.Model): keras model object
    """

    inputs = tf.keras.Input(input_shape)
    #out = tf.keras.layers.Conv2D(16, kernel_size=7, strides=2, padding='same',kernel_initializer="glorot_normal",kernel_regularizer=tf.keras.regularizers.L2(0.01))(inputs)
    out = fetaure_extractor(inputs,32)
    out = fetaure_extractor(out,64)
    out = fetaure_extractor(out,128)
    out = fetaure_extractor(out,256)
    out = fetaure_extractor(out,512)
    out = GlobalAveragePooling2D()(out) # Instead of dense layer
    out = Dropout(dropout_rate)(out)
    out = Dense(dense_units, activation='relu')(out)
    #out = Dense(dense_units/2, activation='relu')(out)
    #out = Dropout(dropout_rate)(out)
    outputs = Dense(n_classes, activation = 'softmax')(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='custom')

@gin.configurable
def vgg_like(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    """Defines a VGG-like architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled for every VGG block
        n_blocks (int): number of VGG blocks
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate

    Returns:
        (keras.Model): keras model object
    """

    assert n_blocks > 0, 'Number of blocks has to be at least 1.'

    inputs = tf.keras.Input(input_shape)
    out = vgg_block(inputs, base_filters)
    for i in range(1, n_blocks):
        out = vgg_block(out, base_filters * 2 ** (i))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')


@gin.configurable
def inception_resnet_v2(input_shape, n_classes, dense_units, base_trainable,dropout_rate):
    """Defines inception resnet architecture.
    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        dense_units (int): number of dense units
        base_trainables(bool): Defines if the base model should be trained or if the pretrained weights
                                should be used.
        dropout_rate (int) : dropout rate used in the model
    Returns:
        (keras.Model): keras model object
    """
    base_model =tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', pooling= 'avg')
    base_model.trainable = base_trainable
    out = base_model.output
    out = Dense(dense_units, activation='relu')(out)
    out = Dense(dense_units/2, activation='relu')(out)
    out = Dropout(dropout_rate)(out)
    outputs = Dense(n_classes, activation = 'softmax')(out)

    return tf.keras.Model(inputs=base_model.input, outputs=outputs, name='inception_resnet_v2')

@gin.configurable
def resnet_v2(input_shape, n_classes, dense_units, base_trainable,dropout_rate):
    """Defines resnet_v2 architecture.
    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        dense_units (int): number of dense units
        base_trainables(bool): Defines if the base model should be trained or if the pretrained weights
                                should be used.
        dropout_rate (int) : dropout rate used in the model
    Returns:
        (keras.Model): keras model object
    """
    base_model =tf.keras.applications.resnet_v2.ResNet50V2(input_shape = input_shape, include_top=False, weights='imagenet', pooling= 'avg')
    base_model.trainable = base_trainable
    out = base_model.output 
    out = Dense(dense_units, activation='relu')(out)
    out = Dropout(dropout_rate)(out)
    outputs = Dense(n_classes, activation = 'softmax')(out)

    return tf.keras.Model(inputs=base_model.input, outputs=outputs, name='resnet_v2')

@gin.configurable
def inception_v3(input_shape, n_classes, dense_units, base_trainable,dropout_rate):
    """Defines inception_v3 architecture.
    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        dense_units (int): number of dense units
        base_trainables(bool): Defines if the base model should be trained or if the pretrained weights
                                should be used.
        dropout_rate (int) : dropout rate used in the model
    Returns:
        (keras.Model): keras model object
    """
    base_model =tf.keras.applications.inception_v3.InceptionV3(input_shape = input_shape, include_top=False, weights='imagenet', pooling= 'avg')
    base_model.trainable = base_trainable
    out = base_model.output
    out = Dropout(dropout_rate)(out)
    out = Dense(dense_units, activation='relu')(out)
    out = Dropout(dropout_rate)(out)
    outputs = Dense(n_classes, activation = 'softmax')(out)

    return tf.keras.Model(inputs=base_model.input, outputs=outputs, name='inception_v3')

@gin.configurable
def xception(input_shape, n_classes, dense_units, base_trainable,dropout_rate):
    """Defines xception architecture.
    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        dense_units (int): number of dense units
        base_trainables(bool): Defines if the base model should be trained or if the pretrained weights
                                should be used.
        dropout_rate (int) : dropout rate used in the model
    Returns:
        (keras.Model): keras model object
    """
    base_model =tf.keras.applications.xception.Xception(input_shape = input_shape, include_top=False, weights='imagenet', pooling= 'avg')
    base_model.trainable = base_trainable
    out = base_model.output
    out = Dense(dense_units, activation='relu')(out)
    out = Dense(dense_units/2, activation='relu')(out)
    out = Dropout(dropout_rate)(out)
    outputs = Dense(n_classes, activation = 'softmax')(out)

    return tf.keras.Model(inputs=base_model.input, outputs=outputs, name='xception')