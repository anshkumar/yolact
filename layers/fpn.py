import tensorflow as tf


class FeaturePyramidNeck(tf.keras.layers.Layer):
    """
        Creating the backbone component of feature Pyramid Network
        Arguments:
            num_fpn_filters
    """

    def __init__(self, num_fpn_filters):
        super(FeaturePyramidNeck, self).__init__()
        self.upSample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

        # no Relu for downsample layer
        # Pytorch and tf differs in conv2d when stride > 1
        # https://dmolony3.github.io/Pytorch-to-Tensorflow.html
        # Hence, manually adding padding
        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.downSample1 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 2, padding="valid",
                                                  kernel_initializer=tf.keras.initializers.glorot_uniform())

        self.pad2 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.downSample2 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 2, padding="valid",
                                                  kernel_initializer=tf.keras.initializers.glorot_uniform())

        self.lateralCov1 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same",
                                                  kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.lateralCov2 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same",
                                                  kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.lateralCov3 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same",
                                                  kernel_initializer=tf.keras.initializers.glorot_uniform())

        # predict layer for FPN
        self.predictP5 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 1, padding="same",
                                                kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                activation="relu")
        self.predictP4 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 1, padding="same",
                                                kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                activation="relu")
        self.predictP3 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 1, padding="same",
                                                kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                activation="relu")

    def call(self, c3, c4, c5):
        # lateral conv for c3 c4 c5
        # pytorch input  (N,Cin,Hin,Win) 
        # tf input (N,Hin,Win,Cin) 
        p5 = self.lateralCov1(c5)
        # _, h, w, _ = tf.shape(c4)
        p4 = tf.add(tf.image.resize(p5, [tf.shape(c4)[1],tf.shape(c4)[2]]), self.lateralCov2(c4))
        # _, h, w, _ = tf.shape(c3)
        p3 = tf.add(tf.image.resize(p4, [tf.shape(c3)[1],tf.shape(c3)[2]]), self.lateralCov3(c3))
        # print("p3: ", p3.shape)

        # smooth pred layer for p3, p4, p5
        p3 = self.predictP3(p3)
        p4 = self.predictP4(p4)
        p5 = self.predictP5(p5)

        # downsample conv to get p6, p7
        p6 = self.downSample1(self.pad1(p5))
        p7 = self.downSample2(self.pad2(p6))

        return [p3, p4, p5, p6, p7]

