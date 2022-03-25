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
                 mode=None,
                 label_map=None):

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
        # resize the mask to proto output size in advance 
        self._proto_output_size = proto_output_size

        # Device.
        self._use_bfloat16 = use_bfloat16
        self.count = 0

        self.id_mapping = {}
        for i, itm in zip(range(1, len(label_map)+1), label_map):
            self.id_mapping[itm['id']] = i

        # https://stackoverflow.com/a/59414295/4582711
        init = tf.lookup.KeyValueTensorInitializer(
            keys=list(self.id_mapping.keys()),
            values=list(self.id_mapping.values()),
            key_dtype=tf.int64,
            value_dtype=tf.int64)
        self.table_id_mapping = tf.lookup.StaticVocabularyTable(
            init,
            num_oov_buckets=1)


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
        classes = self.table_id_mapping.lookup(classes)
        boxes = data['gt_bboxes']
        masks = data['gt_masks']
        image_height = data['height']
        image_width = data['width']

        # read and normalize the image
        image = data['image']
        
        # resize mask
        masks = tf.cast(masks, tf.bool)
        masks = tf.cast(masks, tf.float32)

        # Mask values should only be either 0 or 1
        masks = tf.cast(masks + 0.5, tf.uint8)

        # data augmentation randomly
        if augment:
            image, boxes, masks, classes = augmentation.random_augmentation(
                image, boxes, masks, [self._output_size_h, self._output_size_w],
                self._proto_output_size, classes)

        masks = tf.expand_dims(masks, axis=-1)

        image = tf.image.resize(image, 
            [self._output_size_h, self._output_size_w])

        masks = tf.image.resize(masks, 
            [self._proto_output_size[0], self._proto_output_size[1]],
            method=tf.image.ResizeMethod.BILINEAR)

        masks = tf.squeeze(masks)
        masks = tf.cast(masks + 0.5, tf.uint8)
        masks = tf.cast(masks, tf.float32)

        # remember to unnormalized the bbox
        # [ymin, xmin, ymax, xmax ]
        boxes_norm = boxes
        boxes = boxes * [self._output_size_h, self._output_size_w, self._output_size_h , self._output_size_w]
        
        # matching anchors
        # all_offsets, conf_gt, prior_max_box, prior_max_index = \
        # self._anchor_instance.matching(
        #     self._match_threshold, self._unmatched_threshold, boxes, classes)
        all_offsets, conf_gt, prior_max_box, prior_max_index = \
        self._anchor_instance.matching(
            self._match_threshold, self._unmatched_threshold, boxes_norm, classes)

        # number of object in training sample
        num_obj = tf.size(classes)

        # Padding classes and mask to fix length [None, num_max_fix_padding,...]
        num_padding = self._num_max_fix_padding - tf.shape(classes)[0]
        pad_classes = tf.zeros([num_padding], dtype=tf.int64)
        pad_boxes = tf.zeros([num_padding, 4])
        pad_masks = tf.zeros([num_padding, self._proto_output_size[0], 
            self._proto_output_size[1]])
        boxes_norm = tf.concat([boxes_norm, pad_boxes], axis=0)

        if tf.shape(classes)[0] == 1:
            masks = tf.expand_dims(masks, axis=0)

        masks = tf.concat([masks, pad_masks], axis=0)
        classes = tf.concat([classes, pad_classes], axis=0)

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
