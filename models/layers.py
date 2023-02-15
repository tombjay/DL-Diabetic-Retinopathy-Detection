import gin
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization

@gin.configurable
def fetaure_extractor(inputs, filtrs):
      """Defines a single feature extractor block.
      Parameters:
          input: Input to the feature extractor block from the previous layer
          filtrs : No. of filters in the convolutional layers
      Cannot be used as the first layer, an input layer/other layer should be present above this blocl

      Returns:
          (keras.Model): keras model object
      """
      out = Conv2D(filtrs, (3, 3), padding='same', activation='relu',kernel_initializer="he_normal",kernel_regularizer=tf.keras.regularizers.L2(0.01))(inputs)
      out = Conv2D(filtrs, (3, 3), activation='relu',kernel_initializer="he_normal",kernel_regularizer=tf.keras.regularizers.L2(0.01))(out)
      out = MaxPooling2D(pool_size=(2, 2))(out)
      out = BatchNormalization()(out)

      return out
 
@gin.configurable
def vgg_block(inputs, filters, kernel_size):
    """A single VGG block consisting of two convolutional layers, followed by a max-pooling layer.

    Parameters:
        inputs (Tensor): input of the VGG block
        filters (int): number of filters used for the convolutional layers
        kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)

    Returns:
        (Tensor): output of the VGG block
    """

    out = tf.keras.layers.Conv2D(
        filters, kernel_size, padding='same', activation=tf.nn.relu)(inputs)
    out = tf.keras.layers.Conv2D(
        filters, kernel_size, padding='same', activation=tf.nn.relu)(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    return out
 