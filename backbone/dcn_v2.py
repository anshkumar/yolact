import tensorflow as tf
import tensorflow_addons.utils.keras_utils as conv_utils
from tensorflow_addons.layers.deformable_conv2d import DeformableConv2D

class DCN(tf.keras.layers.Layer):
    def __init__(self, filters,
                 kernel_size, strides=(1, 1), padding='valid',
                 data_format="channels_first",
                 dilation_rate=1, weight_groups=1,
                 offset_groups=1, use_mask=True,
                 use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        super(DCN, self).__init__()
        # super(DCN, self).__init__(filters, kernel_size, strides, padding, 
        #     data_format, dilation_rate, weight_groups, offset_groups, use_mask, 
        #     use_bias, kernel_initializer, bias_initializer, kernel_regularizer, 
        #     bias_regularizer, kernel_constraint, bias_constraint, **kwargs) 

        if weight_groups != 1 or offset_groups != 1:
            raise ValueError("groups greater than 1 is not supported.")

        self.use_mask = use_mask
        self.null_mask = tf.zeros((0, 0, 0, 0))
        if use_mask:
            out_chan = 3
        else:
            out_chan = 2
        if isinstance(kernel_size, tuple):
            channels_ = out_chan * self.kernel_size[0] * self.kernel_size[1]
        else:
            channels_ = out_chan * kernel_size * kernel_size
        self.conv_offset_mask = tf.keras.layers.Conv2D(channels_,
                                          kernel_size=kernel_size,
                                          strides=strides,
                                          padding=padding,
                                          use_bias=True,
                                          kernel_initializer='zeros',
                                          bias_initializer='zeros',
                                          name='conv_offset_mask')

        self.conv = DeformableConv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            weight_groups=weight_groups,
            offset_groups=offset_groups,
            use_mask=use_mask,
            use_bias=True,
        )

    def call(self, input_tensor, **kwargs):
        out = self.conv_offset_mask(input_tensor)
        if self.use_mask:
            o1, o2, mask = tf.split(out, num_or_size_splits=3, axis=-1)
            mask_tensor = tf.keras.activations.sigmoid(mask)
        else:
            o1, o2 = tf.split(out, num_or_size_splits=2, axis=-1)
        offset_tensor = tf.concat((o1, o2), axis=-1)
        
        # change into channels_first format
        input_tensor = tf.transpose(input_tensor, (0, 3, 1, 2))
        offset_tensor = tf.transpose(offset_tensor, (0, 3, 1, 2))
        if self.use_mask:
            mask_tensor = tf.transpose(mask_tensor, (0, 3, 1, 2))
        else:
            mask_tensor = self.null_mask

        out = self.conv([input_tensor, offset_tensor, mask_tensor])

        # change into channels_last format
        return tf.transpose(out, (0, 2, 3, 1))


