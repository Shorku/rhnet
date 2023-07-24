"""RhNet construction module"""

import tensorflow as tf

from model.blocks import ConvBlockExp, DenseBlock
from model.blocks import dense_layers_funcapi, cnn_layers_funcapi
from model.blocks import dense_cnn_functional, dense_multker_functional
from model.params import image_dim, macro_dim


def model_choice(params):
    """Choose architecture based on params.model and return model instance.

    Args:
        params (munch.Munch): Command line parameters

    Return:
        (tf.keras.Model): RhNet model instance

    """
    index = params.model
    dnn_dropout = params.dnn_dropout
    cnn_dropout = params.cnn_dropout
    dnn_l2 = params.dnn_l2
    cnn_l2 = params.cnn_l2
    precision = tf.float16 if params.use_amp else tf.float32
    activation = {'relu': 'relu',
                  'sigmoid': 'sigmoid',
                  'tanh': 'tanh',
                  'leaky_relu': tf.nn.leaky_relu,
                  'elu': 'elu'}[params.activation]
    cube_dim = image_dim()
    macr_dim = macro_dim(params)
    if index == 'linear':
        return LR()
    elif index == 'densenet':
        return densenet(dnn_dropout, cnn_dropout, dnn_l2, cnn_l2,
                        precision, activation, cube_dim, macr_dim)
    elif index == 'multikernel':
        return multikernel(dnn_dropout, cnn_dropout, dnn_l2, cnn_l2,
                           precision, activation, cube_dim, macr_dim)
    else:
        raise ValueError('Invalid model choice')


def densenet(dnn_dropout, cnn_dropout, dnn_l2, cnn_l2, precision, activation,
             cube_dim, macr_dim):
    """Define a model with DenseNet-style convolutional neural network using
    Functional API

    Args:
        dnn_dropout (float): dropout rate for the dense part of the model
        cnn_dropout (float): dropout rate for the convolutional part of the
            model
        dnn_l2 (float): l2 regularization factor for the dense part of
            the model
        cnn_l2 (float): l2 regularization factor for the convolutional part of
            the model
        precision (tensorflow.python.framework.dtypes.DType): input data
            precision
        activation (str or function): activation function
        cube_dim (Tuple[int]): density image dimensions
        macr_dim (int): number of macroscopic features

    Return:
        (tf.keras.Model): RhNet model instance

    """
    # Define layers
    conv_unit_list = [14, 16, 32, 64, 256, 512]
    conv_kern_list = [3, 3, 3, 3, 3, 4]
    conv_pad_list = ['SAME', 'SAME', 'SAME', 'SAME', 'SAME', 'VALID']
    dense_unit_list = [512, 512, 512, 256, 256, 256, 128]
    conv_pool_list = [0, 1, 1, 1, 1, 1]

    return dense_cnn_functional(conv_unit_list, conv_kern_list,
                                conv_pad_list, dense_unit_list, conv_pool_list,
                                dnn_dropout, cnn_dropout, dnn_l2, cnn_l2,
                                precision, activation, cube_dim, macr_dim)


def multikernel(dnn_dropout, cnn_dropout, dnn_l2, cnn_l2, precision,
                activation, cube_dim, macr_dim):
    """Define a model with multikernel convolutional neural network using
    Functional API

    Args:
        dnn_dropout (float): dropout rate for the dense part of the model
        cnn_dropout (float): dropout rate for the convolutional part of the
            model
        dnn_l2 (float): l2 regularization factor for the dense part of
            the model
        cnn_l2 (float): l2 regularization factor for the convolutional part of
            the model
        precision (tensorflow.python.framework.dtypes.DType): input data
            precision
        activation (str or function): activation function
        cube_dim (Tuple[int]): density image dimensions
        macr_dim (int): number of macroscopic features

    Return:
        (tf.keras.Model): RhNet model instance

    """
    # Define layers
    conv3_unit_list = [16, 16, 32, 64, 128, 128]
    conv5_unit_list = [8, 8, 16, 32, 128]
    conv7_unit_list = [4, 4, 8, 16]
    conv3_pad_list = ['SAME', 'SAME', 'SAME', 'SAME', 'VALID', 'VALID']
    conv5_pad_list = ['SAME', 'SAME', 'SAME', 'SAME', 'VALID']
    conv7_pad_list = ['SAME', 'SAME', 'SAME', 'SAME']
    dense_unit_list = [512, 512, 512, 256, 256, 256, 128]
    conv_pool_list = [1, 1, 1, 1, 0, 0]

    return dense_multker_functional(conv3_unit_list, conv3_pad_list,
                                    conv5_unit_list, conv5_pad_list,
                                    conv7_unit_list, conv7_pad_list,
                                    conv_pool_list, dense_unit_list,
                                    dnn_dropout, cnn_dropout, dnn_l2, cnn_l2,
                                    precision, activation, cube_dim, macr_dim,
                                    True)


class LR(tf.keras.Model):
    """
    Linear model

    """
    def __init__(self):
        super().__init__(self)
        self.dense = tf.keras.layers.Dense(1, activation=None)

    def call(self, features, training=False):
        x, y, z = features
        out1 = tf.keras.layers.Flatten()(x)
        out2 = tf.keras.layers.Flatten()(y)
        out = tf.keras.layers.Concatenate(axis=-1)([out1, out2, z])
        out = self.dense(out)

        return out


# Further are a couple of examples of a model built both using functional API
# and as a subclass of the tf.keras.Model class, they are left here for fun
class LargeRegularCNNSubclassed(tf.keras.Model):
    """
    Define "large" net

    """
    def __init__(self, dnn_dropout, cnn_dropout):
        super().__init__(self)
        self.conv_block = ConvBlockExp(cnn_dropout, pooling='av_pool')
        self.dense_block = DenseBlock([2048, 2048, 1024, 512,
                                       256, 256, 256, 128], dnn_dropout)

    def call(self, features, training=False):
        x, y, z = features
        out1 = self.conv_block(x, training)
        out2 = self.conv_block(y, training)
        out1 = tf.keras.layers.Flatten()(out1)
        out2 = tf.keras.layers.Flatten()(out2)
        out = tf.keras.layers.Concatenate(axis=-1)([out1, out2, z])
        out = self.dense_block(out, training)

        return out


def large_regular_cnn_functional(dnn_dropout, cnn_dropout, dnn_l2, cnn_l2,
                                 precision, activation, cube_dim, macr_dim):
    """Define "large" net using Functional API

    Args:
        dnn_dropout (float): dropout rate for the dense part of the model
        cnn_dropout (float): dropout rate for the convolutional part of the
            model
        dnn_l2 (float): l2 regularization factor for the dense part of
            the model
        cnn_l2 (float): l2 regularization factor for the convolutional part of
            the model
        precision (tensorflow.python.framework.dtypes.DType): input data
            precision
        activation (str or function): activation function
        cube_dim (Tuple[int]): density image dimensions
        macr_dim (int): number of macroscopic features

    Return:
        (tf.keras.Model): RhNet model instance

    """
    # Define inputs
    poly_input = tf.keras.Input(shape=cube_dim, dtype=precision)
    solv_input = tf.keras.Input(shape=cube_dim, dtype=precision)
    feat_input = tf.keras.Input(shape=macr_dim, dtype=precision)
    poly_out = poly_input
    solv_out = solv_input
    # Define layers
    pool_index = [0, 3, 5, 6]
    n_conv_layers = 7
    conv_unit_list = [64, 64, 128, 128, 128, 256, 256]
    conv_kern_list = [3 for i in range(7)]
    conv_padd_list = ['VALID' for i in range(7)]
    dense_unit_list = [2048, 2048, 1024, 512, 256, 256, 256, 128]
    conv_layers = cnn_layers_funcapi(conv_unit_list, conv_kern_list,
                                     conv_padd_list, cnn_l2, activation)
    dense_layers = dense_layers_funcapi(dense_unit_list, dnn_l2, activation)

    # Define forward pass
    for i in range(n_conv_layers):
        poly_out = conv_layers[i](poly_out)
        solv_out = conv_layers[i](solv_out)
        if i in pool_index:
            poly_out = tf.keras.layers. \
                    AveragePooling3D(pool_size=2, strides=2)(poly_out)
            poly_out = tf.keras.layers.experimental.preprocessing.\
                Rescaling(8)(poly_out)
            solv_out = tf.keras.layers. \
                AveragePooling3D(pool_size=2, strides=2)(solv_out)
            solv_out = tf.keras.layers.experimental.preprocessing. \
                Rescaling(8)(solv_out)
        if cnn_dropout:
            poly_out = tf.keras.layers. \
                SpatialDropout3D(rate=cnn_dropout)(poly_out)
            solv_out = tf.keras.layers. \
                SpatialDropout3D(rate=cnn_dropout)(solv_out)
    poly_out = tf.keras.layers.Flatten()(poly_out)
    solv_out = tf.keras.layers.Flatten()(solv_out)
    out = tf.keras.layers.Concatenate(axis=-1)([poly_out,
                                                solv_out,
                                                feat_input])
    for dense in dense_layers:
        out = dense(out)
        if dnn_dropout:
            out = tf.keras.layers.Dropout(rate=dnn_dropout)(out)
    out = tf.keras.layers.Dense(1, activation=None)(out)

    return tf.keras.Model([poly_input, solv_input, feat_input], out)
