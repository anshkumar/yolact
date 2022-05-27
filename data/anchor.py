from itertools import product
from math import sqrt
from utils import utils
import tensorflow as tf

# Can generate one instance only when creating the model
class Anchor(object):

    def __init__(self, img_size_h, img_size_w, feature_map_size, aspect_ratio, 
        scale):
        """
        :param img_size:
        :param feature_map_size:
        :param aspect_ratio:
        :param scale:
        """
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.num_anchors, self.anchors_norm = self._generate_anchors(
            feature_map_size, aspect_ratio, scale)
        self.anchors = self.get_anchors()

    def _generate_anchors(self, feature_map_size, aspect_ratio, scale):
        """
        :param img_size:
        :param feature_map_size:
        :param aspect_ratio:
        :param scale:
        :return:
        """
        prior_boxes = []
        num_anchors = 0
        for idx, f_size in enumerate(feature_map_size):
            count_anchor = 0
            for j, i in product(range(int(f_size[0])), range(int(f_size[1]))):
                # i,j are pixels values in feature map
                # + 0.5 because priors are in center
                x = (i + 0.5) / f_size[1] * self.img_size_w # normalize the pixel values
                y = (j + 0.5) / f_size[0] * self.img_size_h
                for ars in aspect_ratio:
                    a = sqrt(ars)
                    w = scale[idx] * a #/ self.img_size_w
                    h = scale[idx] / a #/ self.img_size_h
                    # directly use point form here => [cx, cy, w, h]
                    prior_boxes += [x, y, w, h]
                    count_anchor += 1
            num_anchors += count_anchor
            print("Create priors for f_size: ", f_size, 
                " aspect_ratio: ",aspect_ratio, 
                " scale: ", scale[idx],
                " anchors count: ", count_anchor)
        output = tf.reshape(tf.convert_to_tensor(prior_boxes), [-1, 4])
        output = tf.cast(output, tf.float32)
        return num_anchors, output

    def get_anchors(self):
        # Convert anchors from [cx, cy, w, h] to [ymin, xmin, ymax, xmax ] 
        # for IOU calculations
        w = self.anchors_norm[:, 2]
        h = self.anchors_norm[:, 3]
        anchors_yxyx = tf.cast(tf.stack(
            [(self.anchors_norm[:, 1] - (h / 2)), 
            (self.anchors_norm[:, 0] - (w / 2)), 
            (self.anchors_norm[:, 1] + (h / 2)), 
            (self.anchors_norm[:, 0] + (w / 2))], 
            axis=-1), tf.float32)

        return anchors_yxyx

    def matching(self, pos_thresh, neg_thresh, gt_bbox, gt_labels):
        # size: [num_objects, num_priors]; anchors along the row and 
        # ground_truth clong the columns

        # anchors and gt_bbox in [y1, x1, y2, x2]
        pairwise_iou = utils._iou(self.anchors, gt_bbox) 

        # size [num_priors]; iou with ground truth with the anchors
        each_prior_max = tf.reduce_max(pairwise_iou, axis=-1) 

        if tf.shape(pairwise_iou)[-1] == 0: # No positive ground-truth boxes
            return (self.anchors*0, tf.cast(self.anchors[:, 0]*0, dtype=tf.int64), 
                self.anchors*0, tf.cast(self.anchors[:, 0]*0, dtype=tf.int64))

        # size [num_priors]; id of groud truth having max iou with the anchors
        each_prior_index = tf.math.argmax(pairwise_iou, axis=-1) 

        each_box_max = tf.reduce_max(pairwise_iou, axis=0)
        each_box_index = tf.math.argmax(pairwise_iou, axis=0)

        # For the max IoU prior for each gt box, set its IoU to 2. This ensures 
        # that it won't be filtered in the threshold step even if the IoU is 
        # under the negative threshold. This is because that we want
        # at least one prior to match with each gt box or else we'd be wasting 
        # training data.

        indices = tf.expand_dims(each_box_index,axis=-1)

        updates = tf.cast(tf.tile(tf.constant([2]), tf.shape(each_box_index)), 
            dtype=tf.float32)
        each_prior_max = tf.tensor_scatter_nd_update(each_prior_max, indices, 
            updates)

        # Set the index of the pair (prior, gt) we set the overlap for above.
        updates = tf.cast(tf.range(0,tf.shape(each_box_index)[0]),
            dtype=tf.int64)
        each_prior_index = tf.tensor_scatter_nd_update(each_prior_index, 
            indices, updates)

        # size: [num_priors, 4]; each_prior_box in [y1, x1, y2, x2]
        each_prior_box = tf.gather(gt_bbox, each_prior_index) 

        # the class of the max IoU gt box for each prior, size: [num_priors]
        conf = tf.squeeze(tf.gather(gt_labels, each_prior_index)) 

        neutral_label_index = tf.where(each_prior_max < pos_thresh)
        background_label_index = tf.where(each_prior_max < neg_thresh)

        conf = tf.tensor_scatter_nd_update(conf, 
            neutral_label_index, 
            -1*tf.ones(tf.size(neutral_label_index), dtype=tf.int64))
        conf = tf.tensor_scatter_nd_update(conf, 
            background_label_index, 
            tf.zeros(tf.size(background_label_index), dtype=tf.int64))

        # anchors and each_prior_box in [y1, x1, y2, x2]
        offsets = utils._encode(each_prior_box, self.anchors)

        return offsets, conf, each_prior_box, each_prior_index
