import tensorflow as tf

from data import tfrecord_decoder
from utils import augmentation
from utils.utils import normalize_image
from functools import partial

class Parser(object):

    def __init__(self,
                 output_size,
                 anchor_instance,
                 match_threshold=0.5,
                 unmatched_threshold=0.4,
                 num_max_fix_padding=100,
                 proto_output_size=[138,138],
                 skip_crow_during_training=True,
                 use_bfloat16=True,
                 mode=None):

        self._mode = mode
        self._skip_crowd_during_training = skip_crow_during_training
        self._is_training = (mode == "train")

        self._example_decoder = tfrecord_decoder.TfExampleDecoder()

        self._output_size_h = output_size[0]
        self._output_size_w = output_size[1]
        self._anchor_instance = anchor_instance
        self._match_threshold = match_threshold
        self._unmatched_threshold = unmatched_threshold

        # output related
        # for classes and mask to be padded to fix length
        self._num_max_fix_padding = num_max_fix_padding
        # resize the mask to proto output size in advance (always 138, from paper's figure)
        self._proto_output_size = proto_output_size

        # Device.
        self._use_bfloat16 = use_bfloat16
        self.count = 0

        # Data is parsed depending on the model.
        if mode == "train":
            self._parse_fn = partial(self._parse, augment=True)
        elif mode == "val":
            self._parse_fn = partial(self._parse, augment=False)
        elif mode == "test":
            self._parse_fn = self._parse_predict_data
        else:
            raise ValueError('mode is not defined.')

    def __call__(self, value):
        with tf.name_scope('parser'):
            data = self._example_decoder.decode(value)
            return self._parse_fn(data)

    def _parse(self, data, augment):
        classes = data['gt_classes']
        boxes = data['gt_bboxes']
        masks = data['gt_masks']
        image_height = data['height']
        image_width = data['width']

        # read and normalize the image
        image = data['image']
        
        #########################
        # _mean = tf.constant([103.94, 116.78, 123.68])
        # _std = tf.constant([57.38, 57.12, 58.40])
        # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # image = (image - _mean) / _std

        # convert image to range [0, 1]
        # https://github.com/tensorflow/tensorflow/issues/33892
        # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # image = tf.image.per_image_standardization(tf.image.convert_image_dtype(image, dtype=tf.float32))
        # image = image / 255.0
        # image = normalize_image(image)
        # tf.io.write_file('out.jpg', tf.image.encode_jpeg(tf.cast(image, tf.uint8)))

        # resize mask
        masks = tf.cast(masks, tf.bool)
        masks = tf.cast(masks, tf.float32)

        # Mask values should only be either 0 or 1
        masks = tf.cast(masks + 0.5, tf.uint8)

        # Todo: SSD data augmentation (Photometrics, expand, sample_crop, mirroring)
        # data augmentation randomly
        if augment:
            image, boxes, masks, classes = augmentation.random_augmentation(image, boxes, masks, [self._output_size_h, self._output_size_w],
                                                                        self._proto_output_size, classes)
        masks = tf.expand_dims(masks, axis=-1)
        image = tf.image.resize(image, [self._output_size_h, self._output_size_w])

        masks = tf.image.resize(masks, [self._proto_output_size[0], self._proto_output_size[1]],
                                        method=tf.image.ResizeMethod.BILINEAR)
        masks = tf.squeeze(masks)
        masks = tf.cast(masks + 0.5, tf.uint8)
        masks = tf.cast(masks, tf.float32)
        
        # matching anchors
        all_offsets, conf_gt, prior_max_box, prior_max_index = self._anchor_instance.matching(
            self._match_threshold, self._unmatched_threshold, boxes, classes)

        boxes_norm = boxes
        # remember to unnormalized the bbox
        # [ymin, xmin, ymax, xmax ]
        boxes = boxes * [self._output_size_h, self._output_size_w, self._output_size_h , self._output_size_w]

        # number of object in training sample
        num_obj = tf.size(classes)

        # resized boxes for proto output size
        # boxes_norm = boxes * [self._proto_output_size[0] / self._output_size_h, self._proto_output_size[1] / self._output_size_w, self._proto_output_size[0] / self._output_size_h, self._proto_output_size[1] / self._output_size_w]


        # Padding classes and mask to fix length [None, num_max_fix_padding, ...]
        num_padding = self._num_max_fix_padding - tf.shape(classes)[0]
        pad_classes = tf.zeros([num_padding], dtype=tf.int64)
        pad_boxes = tf.zeros([num_padding, 4])
        pad_masks = tf.zeros([num_padding, self._proto_output_size[0], self._proto_output_size[1]])

        if tf.shape(classes)[0] == 1:
            masks = tf.expand_dims(masks, axis=0)

        masks = tf.concat([masks, pad_masks], axis=0, name="parser_concat_masks")
        classes = tf.concat([classes, pad_classes], axis=0, name="parser_concat_classes")
        boxes = tf.concat([boxes, pad_boxes], axis=0, name="parser_concat_boxes")
        boxes_norm = tf.concat([boxes_norm, pad_boxes], axis=0, name="parser_concat_boxes_norm")

        labels = {
            'all_offsets': all_offsets,
            'conf_gt': conf_gt,
            'prior_max_box': prior_max_box,
            'prior_max_index': prior_max_index,
            'boxes_norm': boxes_norm,
            'classes': classes,
            'num_obj': num_obj,
            'mask_target': masks,
        }
        return image, labels

    def _parse_predict_data(self, data):
        pass
