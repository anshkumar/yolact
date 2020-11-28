import tensorflow as tf

from layers.fpn import FeaturePyramidNeck
from layers.head import PredictionModule
from layers.protonet import ProtoNet
from utils.create_prior import make_priors
import numpy as np
assert tf.__version__.startswith('2')
from detection import Detect

class Yolact(tf.keras.Model):
    """
        Creating the YOLCAT Architecture
        Arguments:

    """

    def __init__(self, img_h, img_w, fpn_channels, num_class, num_mask, aspect_ratio, scales):
        super(Yolact, self).__init__()
        out = ['conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
        # use pre-trained ResNet50
        # Todo figure out how pre-trained can be train again
        base_model = tf.keras.applications.ResNet50(input_shape=(img_h, img_w, 3),
                                                    include_top=False,
                                                    layers=tf.keras.layers,
                                                    weights='imagenet')
        # extract certain feature maps for FPN
        self.backbone_resnet = tf.keras.Model(inputs=base_model.input,
                                              outputs=[base_model.get_layer(x).output for x in out])
        
        # Calculating feature map size
        # https://stackoverflow.com/a/44242277/4582711
        # https://github.com/tensorflow/tensorflow/issues/4297#issuecomment-246080982
        self.feature_map_size = np.array([list(base_model.get_layer(x).output.shape[1:3]) for x in out])
        out_height_p6 = np.ceil((self.feature_map_size[-1, 0]).astype(np.float32) / float(2))
        out_width_p6  = np.ceil((self.feature_map_size[-1, 1]).astype(np.float32) / float(2))
        out_height_p7 = np.ceil(out_height_p6 / float(2))
        out_width_p7  = np.ceil(out_width_p6/ float(2))
        self.feature_map_size = np.concatenate((self.feature_map_size, [[out_height_p6, out_width_p6], [out_height_p7, out_width_p7]]), axis=0)
        self.protonet_out_size = self.feature_map_size[0]*2 # Only one upsampling on p3 

        self.backbone_fpn = FeaturePyramidNeck(fpn_channels)
        self.protonet = ProtoNet(num_mask)

        # semantic segmentation branch to boost feature richness
        self.semantic_segmentation = tf.keras.layers.Conv2D(num_class-1, (1, 1), 1, padding="same",
                                                            kernel_initializer=tf.keras.initializers.glorot_uniform())

        self.num_anchor, self.priors = make_priors(img_h, img_w, self.feature_map_size, aspect_ratio, scales)
        # print("prior shape:", self.priors.shape)
        # print("num anchor per feature map: ", self.num_anchor)

        # shared prediction head
        # Here, len(aspect_ratio) is passed as during prior calculations, individula scale is selected for each layer.
        # So, when scale are [24, 48, 96, 130, 192] that means 24 is for p3; 48 is for p4 and so on.
        # So, number of priors for that layer will be HxWxlen(aspect_ratio)
        # Hence, passing len(aspect_ratio)
        # This implementation differs from the original used in yolact
        self.predictionHead = PredictionModule(256, len(aspect_ratio), num_class, num_mask)

        # post-processing for evaluation
        self.detect = Detect(num_class, bkg_label=0, top_k=200,
                conf_thresh=0.05, nms_thresh=0.5)

    def set_bn(self, mode='train'):
        if mode == 'train':
            for layer in self.backbone_resnet.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
        else:
            for layer in self.backbone_resnet.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True

    @tf.function(input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32), tf.TensorSpec([None], tf.bool)])
    def call(self, inputs, training):
        # backbone(ResNet + FPN)
        c3, c4, c5 = self.backbone_resnet(inputs)
        fpn_out = self.backbone_fpn(c3, c4, c5)

        # Protonet branch
        p3 = fpn_out[0]
        protonet_out = self.protonet(p3)
        # print("protonet: ", protonet_out.shape)

        # semantic segmentation branch
        seg = self.semantic_segmentation(p3)

        # Prediction Head branch
        pred_cls = []
        pred_offset = []
        pred_mask_coef = []

        # all output from FPN use same prediction head
        for f_map in fpn_out:
            cls, offset, coef = self.predictionHead(f_map)
            pred_cls.append(cls)
            pred_offset.append(offset)
            pred_mask_coef.append(coef)
            
        pred_cls = tf.concat(pred_cls, axis=1)
        pred_offset = tf.concat(pred_offset, axis=1)
        pred_mask_coef = tf.concat(pred_mask_coef, axis=1)

        if training:
            pred = {
                'pred_cls': pred_cls,
                'pred_offset': pred_offset,
                'pred_mask_coef': pred_mask_coef,
                'proto_out': protonet_out,
                'seg': seg,
                'priors': self.priors
            }
        else:
            pred = {
                'pred_cls': tf.nn.softmax(pred_cls, axis=-1),
                'pred_offset': pred_offset,
                'pred_mask_coef': pred_mask_coef,
                'proto_out': protonet_out,
                'seg': seg,
                'priors': self.priors
            }

            pred.update(self.detect(pred))

        return pred
