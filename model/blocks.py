""" Building blocks for RhNet models"""

import tensorflow as tf


def dense_layers_funcapi(dense_unit_list, dnn_l2, activation):
    """Define fully connected layers for networks using Functional API

    Args:
        dense_unit_list (List[int]): number of units in dense layers
        dnn_l2 (float): l2 regularization factor for the dense part of
            the model
        activation (str or function): activation function

    Return:
        List[tf.keras.layers.Layer]: a list of dense layers

    """
    # Define inputs
    return [tf.keras.layers.Dense(unit,
                                  activation=activation,
                                  kernel_regularizer=tf.keras.
                                  regularizers.l2(dnn_l2))
            for unit in dense_unit_list]


def cnn_layers_funcapi(conv_unit_list, conv_kern_list, conv_padd_list, cnn_l2,
                       activation):
    """Define convolutional layers for networks using Functional API

    Args:
        conv_unit_list (List[int]): number of filters in convolutional layers
        conv_kern_list (List[int]): kernel dimensions of convolutional layers
        conv_padd_list (List[str]): padding options for convolutional layers
        cnn_l2 (float): l2 regularization factor for the convolutional part of
            the model
        activation (str or function): activation function

    Return:
        List[tf.keras.layers.Layer]: a list of convolutional layers

    """
    # Define inputs
    n_conv_layers = len(conv_unit_list)

    if (n_conv_layers != len(conv_kern_list)
            or n_conv_layers != len(conv_padd_list)):
        raise RuntimeError('Check convolutional layers definition, ' +
                           'kernels list length is incompatible')

    return [tf.keras.layers.Conv3D(conv_unit_list[i],
                                   conv_kern_list[i],
                                   activation=activation,
                                   padding=conv_padd_list[i],
                                   kernel_regularizer=tf.keras.
                                   regularizers.l2(cnn_l2))
            for i in range(n_conv_layers)]


# TODO make the code more compact
def dense_cnn_functional(conv_unit_list, conv_kern_list, conv_padd_list,
                         dense_unit_list, conv_pool_list,
                         dnn_dropout, cnn_dropout, dnn_l2, cnn_l2,
                         precision, activation, cube_dim, macr_dim):
    """Define a net with DenseNet-style convolutional neural network
    using Functional API

    Args:
        conv_unit_list (List[int]): number of filters in convolutional layers
        conv_kern_list (List[int]): kernel dimensions of convolutional layers
        conv_padd_list (List[str]): padding options for convolutional layers
        dense_unit_list (List[int]): number of units in dense layers
        conv_pool_list (List[int]): whether to apple pooling after ith layer
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
    conv_layers = cnn_layers_funcapi(conv_unit_list, conv_kern_list,
                                     conv_padd_list, cnn_l2, activation)
    dense_layers = dense_layers_funcapi(dense_unit_list, dnn_l2, activation)

    n_conv_layers = len(conv_unit_list)
    # Define forward pass
    for i in range(n_conv_layers):
        if i < n_conv_layers - 1:
            poly_out = tf.keras.layers.\
                Concatenate(axis=-1)([conv_layers[i](poly_out), poly_out])
            solv_out = tf.keras.layers. \
                Concatenate(axis=-1)([conv_layers[i](solv_out), solv_out])
        else:
            poly_out = conv_layers[i](poly_out)
            solv_out = conv_layers[i](solv_out)
        if conv_pool_list[i]:
            poly_out = tf.keras.layers. \
                AveragePooling3D(pool_size=2, strides=2)(poly_out)
            poly_out = tf.keras.layers.experimental.preprocessing.\
                Rescaling(8)(poly_out)
            solv_out = tf.keras.layers. \
                AveragePooling3D(pool_size=2, strides=2)(solv_out)
            solv_out = tf.keras.layers.experimental.preprocessing. \
                Rescaling(8)(solv_out)
        if cnn_dropout and i < n_conv_layers - 1:
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


def dense_multker_functional(conv3_unit_list, conv3_padd_list,
                             conv5_unit_list, conv5_padd_list,
                             conv7_unit_list, conv7_padd_list,
                             conv_pool_list, dense_unit_list,
                             dnn_dropout, cnn_dropout, dnn_l2, cnn_l2,
                             precision, activation, cube_dim, macr_dim,
                             linear_comb=False):
    """Define a net with DenseNet-style multikernel convolutional neural
    network using Functional API

    Args:
        conv3_unit_list (List[int]): number of filters in convolutional layers
        conv3_padd_list (List[str]): padding options for convolutional layers
        conv5_unit_list (List[int]): number of filters in convolutional layers
        conv5_padd_list (List[str]): padding options for convolutional layers
        conv7_unit_list (List[int]): number of filters in convolutional layers
        conv7_padd_list (List[str]): padding options for convolutional layers
        conv_pool_list (List[int]): whether to apple pooling after ith layer
        dense_unit_list (List[int]): number of units in dense layers
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
        linear_comb (bool): whether to use 1x1 convolutions

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
    dense_layers = dense_layers_funcapi(dense_unit_list, dnn_l2, activation)
    n_conv3_layers = len(conv3_unit_list)
    n_conv5_layers = len(conv5_unit_list)
    n_conv7_layers = len(conv7_unit_list)
    conv3_layers = cnn_layers_funcapi(conv3_unit_list,
                                      [3 for i in range(n_conv3_layers)],
                                      conv3_padd_list, cnn_l2, activation)
    conv5_layers = cnn_layers_funcapi(conv5_unit_list,
                                      [5 for i in range(n_conv5_layers)],
                                      conv5_padd_list, cnn_l2, activation)
    conv7_layers = cnn_layers_funcapi(conv7_unit_list,
                                      [7 for i in range(n_conv7_layers)],
                                      conv7_padd_list, cnn_l2, activation)
    if linear_comb:
        lin_comb_filters = [16, 32, 64, 128]
        conv1_layers = [tf.keras.layers.Conv3D(i, 1,
                                               activation=None,
                                               kernel_regularizer=tf.keras.
                                               regularizers.l2(cnn_l2))
                        for i in lin_comb_filters]
    # Define forward pass
    for i in range(n_conv7_layers):
        conv3_poly = conv3_layers[i](poly_out)
        conv5_poly = conv5_layers[i](poly_out)
        conv7_poly = conv7_layers[i](poly_out)
        conv3_solv = conv3_layers[i](solv_out)
        conv5_solv = conv5_layers[i](solv_out)
        conv7_solv = conv7_layers[i](solv_out)
        poly_out = tf.keras.layers. \
            Concatenate(axis=-1)([conv3_poly, conv5_poly, conv7_poly,
                                  poly_out])
        solv_out = tf.keras.layers. \
            Concatenate(axis=-1)([conv3_solv, conv5_solv, conv7_solv,
                                  solv_out])
        if linear_comb:
            solv_out = conv1_layers[i](solv_out)
            poly_out = conv1_layers[i](poly_out)
        if conv_pool_list[i]:
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
    conv5_poly = conv5_layers[-1](poly_out)
    conv5_solv = conv5_layers[-1](solv_out)
    conv3_poly = conv3_layers[-2](poly_out)
    conv3_solv = conv3_layers[-2](solv_out)
    conv3_poly = conv3_layers[-1](conv3_poly)
    conv3_solv = conv3_layers[-1](conv3_solv)

    poly_out = tf.keras.layers.Concatenate(axis=-1)([conv3_poly, conv5_poly])
    solv_out = tf.keras.layers.Concatenate(axis=-1)([conv3_solv, conv5_solv])
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


# Further are the building blocks for a tf.keras.Model subclassed model,
# they are left here for fun
class DenseBlock(tf.keras.Model):
    def __init__(self, units_list, dnn_dropout):
        """ Dense net to be placed on the top of CNN part of RhNet

        Args:
            units_list (List[int]): Dense layers output spaces

        Return:
            RhNet prediction

        """
        super().__init__(self)
        self.dropout = dnn_dropout
        self.depth = len(units_list)
        with tf.name_scope('DenseBlock'):

            self.dense = [tf.keras.layers.Dense(unit,
                                                activation=tf.nn.relu)
                          for unit in units_list]
            if self.dropout:
                self.drop = [tf.keras.layers.Dropout(rate=self.dropout)
                             for unit in units_list]
            self.dense_output = tf.keras.layers.Dense(1, activation=None)

    def __call__(self, inputs, training):
        out = inputs

        for i in range(self.depth):
            out = self.dense[i](out)
            if self.dropout:
                if training:
                    out = self.drop[i](out, training=training)
        out = self.dense_output(out)

        return out


class ConvBlockExp(tf.keras.Model):
    def __init__(self, cnn_dropout, pooling='max_pool'):
        """ Naive convolutional part of RhNet: 4 unpadded convolutional layers
         with 3x3 kernels and ReLu activation interspersed with max-pooling

        Args:
            filters (List[int]): Number of convolution layers filters
            pooling (str): 'max_pool' or 'av_pool' for pooling choice

        Return:
            Tuple of convolved ``inputs``

        """
        super().__init__(self)
        self.drop = cnn_dropout
        with tf.name_scope('ConvBlockDown'):
            self.pooling = pooling
            self.conv1 = tf.keras.layers.Conv3D(filters=64,
                                                kernel_size=(3, 3, 3),
                                                activation=tf.nn.relu)
            self.conv2 = tf.keras.layers.Conv3D(filters=64,
                                                kernel_size=(3, 3, 3),
                                                activation=tf.nn.relu)
            self.conv3 = tf.keras.layers.Conv3D(filters=128,
                                                kernel_size=(3, 3, 3),
                                                activation=tf.nn.relu)
            self.conv4 = tf.keras.layers.Conv3D(filters=128,
                                                kernel_size=(4, 4, 4),
                                                activation=tf.nn.relu)
            self.conv5 = tf.keras.layers.Conv3D(filters=128,
                                                kernel_size=(3, 3, 3),
                                                activation=tf.nn.relu)
            self.conv6 = tf.keras.layers.Conv3D(filters=256,
                                                kernel_size=(3, 3, 3),
                                                activation=tf.nn.relu)
            self.conv7 = tf.keras.layers.Conv3D(filters=256,
                                                kernel_size=(3, 3, 3),
                                                activation=tf.nn.relu)
            if pooling == 'max_pool':
                self.pool = tf.keras.layers.\
                    MaxPool3D(pool_size=(2, 2, 2), strides=2)
            elif pooling == 'av_pool':
                self.pool = tf.keras.layers.\
                    AveragePooling3D(pool_size=(2, 2, 2), strides=2)
            else:
                raise ValueError('ConvBlockDown got incorrect pooling option')
            if self.drop:
                self.drop1 = tf.keras.layers.SpatialDropout3D(rate=self.drop)
                self.drop2 = tf.keras.layers.SpatialDropout3D(rate=self.drop)
                self.drop3 = tf.keras.layers.SpatialDropout3D(rate=self.drop)
                self.drop4 = tf.keras.layers.SpatialDropout3D(rate=self.drop)
                self.drop5 = tf.keras.layers.SpatialDropout3D(rate=self.drop)
                self.drop6 = tf.keras.layers.SpatialDropout3D(rate=self.drop)
                self.drop7 = tf.keras.layers.SpatialDropout3D(rate=self.drop)

    def __call__(self, inputs, training):
        out = self.conv1(inputs)
        out = self.pool(out)
        if self.pooling == 'av_pool':
            out = tf.math.scalar_mul(8, out)
        if self.drop:
            if training:
                out = self.drop1(out, training=training)
        out = self.conv2(out)
        if self.drop:
            if training:
                out = self.drop2(out, training=training)
        out = self.conv3(out)
        if self.drop:
            if training:
                out = self.drop3(out, training=training)
        out = self.conv4(out)
        out = self.pool(out)
        if self.pooling == 'av_pool':
            out = tf.math.scalar_mul(8, out)
        if self.drop:
            if training:
                out = self.drop4(out, training=training)
        out = self.conv5(out)
        if self.drop:
            if training:
                out = self.drop5(out, training=training)
        out = self.conv6(out)
        out = self.pool(out)
        if self.pooling == 'av_pool':
            out = tf.math.scalar_mul(8, out)
        if self.drop:
            if training:
                out = self.drop6(out, training=training)
        out = self.conv7(out)
        out = self.pool(out)
        if self.pooling == 'av_pool':
            out = tf.math.scalar_mul(8, out)
        if self.drop:
            if training:
                out = self.drop7(out, training=training)
        return out
