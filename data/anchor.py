from itertools import product
from math import sqrt

import tensorflow as tf


# Can generate one instance only when creating the model
class Anchor(object):

    def __init__(self, img_size_h, img_size_w, feature_map_size, aspect_ratio, scale):
        """
        :param img_size:
        :param feature_map_size:
        :param aspect_ratio:
        :param scale:
        """
        self.num_anchors, self.anchors = self._generate_anchors(img_size_h, img_size_w, feature_map_size, aspect_ratio, scale)

    def _generate_anchors(self, img_size_h, img_size_w, feature_map_size, aspect_ratio, scale):
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
            # print("Create priors for f_size:%s", f_size)
            count_anchor = 0
            for j, i in product(range(int(f_size[0])), range(int(f_size[1]))):
                # i,j are pixels values in feature map
                # + 0.5 because priors are in center
                x = (i + 0.5) / f_size[1] # normalize the pixel values
                y = (j + 0.5) / f_size[0]
                for ars in aspect_ratio:
                    a = sqrt(ars)
                    w = scale[idx] * a / img_size_w # normalize the width value
                    h = scale[idx] / a / img_size_h
                    # directly use point form here => [cx, cy, w, h]
                    prior_boxes += [x, y, w, h]
                count_anchor += 1
            num_anchors += count_anchor
        output = tf.reshape(tf.convert_to_tensor(prior_boxes), [-1, 4])
        output = tf.cast(output, tf.float32)
        return num_anchors, output

    def _encode(self, map_loc, center_anchors, include_variances=False):
        # center_gt = tf.map_fn(lambda x: map_to_center_form(x), map_loc)
        h = map_loc[:, 2] - map_loc[:, 0]
        w = map_loc[:, 3] - map_loc[:, 1]
        center_gt = tf.cast(tf.stack([map_loc[:, 1] + (w / 2), map_loc[:, 0] + (h / 2), w, h], axis=-1), tf.float32)
        variances = [0.1, 0.2]

        # calculate offset
        if include_variances:
            g_hat_cx = (center_gt[:, 0] - center_anchors[:, 0]) / center_anchors[:, 2] / variances[0]
            g_hat_cy = (center_gt[:, 1] - center_anchors[:, 1]) / center_anchors[:, 3] / variances[0]
        else:
            g_hat_cx = (center_gt[:, 0] - center_anchors[:, 0]) / center_anchors[:, 2]
            g_hat_cy = (center_gt[:, 1] - center_anchors[:, 1]) / center_anchors[:, 3]
        tf.debugging.assert_non_negative(center_anchors[:, 2] / center_gt[:, 2])
        tf.debugging.assert_non_negative(center_anchors[:, 3] / center_gt[:, 3])
        if include_variances:
            g_hat_w = tf.math.log(center_gt[:, 2] / center_anchors[:, 2]) / variances[1]
            g_hat_h = tf.math.log(center_gt[:, 3] / center_anchors[:, 3]) / variances[1]
        else:
            g_hat_w = tf.math.log(center_gt[:, 2] / center_anchors[:, 2])
            g_hat_h = tf.math.log(center_gt[:, 3] / center_anchors[:, 3])
        offsets = tf.stack([g_hat_cx, g_hat_cy, g_hat_w, g_hat_h], axis=-1)
        return offsets

    def _area(self, boxlist, scope=None):
        # https://github.com/tensorflow/models/blob/831281cedfc8a4a0ad7c0c37173963fafb99da37/official/vision/detection/utils/object_detection/box_list_ops.py#L48
        """Computes area of boxes.
        Args:
        boxlist: BoxList holding N boxes
        scope: name scope.
        Returns:
        a tensor with shape [N] representing box areas.
        """
        y_min, x_min, y_max, x_max = tf.split(
            value=boxlist, num_or_size_splits=4, axis=1)
        return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

    def _intersection(self, boxlist1, boxlist2, scope=None):
        # https://github.com/tensorflow/models/blob/831281cedfc8a4a0ad7c0c37173963fafb99da37/official/vision/detection/utils/object_detection/box_list_ops.py#L209
        """Compute pairwise intersection areas between boxes.
        Args:
        boxlist1: BoxList holding N boxes
        boxlist2: BoxList holding M boxes
        scope: name scope.
        Returns:
        a tensor with shape [N, M] representing pairwise intersections
        """
        y_min1, x_min1, y_max1, x_max1 = tf.split(
            value=boxlist1, num_or_size_splits=4, axis=1)
        y_min2, x_min2, y_max2, x_max2 = tf.split(
            value=boxlist2, num_or_size_splits=4, axis=1)
        all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
        all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
        intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
        all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
        intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
        return intersect_heights * intersect_widths

    def _iou(self, boxlist1, boxlist2, scope=None):
        # https://github.com/tensorflow/models/blob/831281cedfc8a4a0ad7c0c37173963fafb99da37/official/vision/detection/utils/object_detection/box_list_ops.py#L259
        """Computes pairwise intersection-over-union between box collections.
        Args:
        boxlist1: BoxList holding N boxes
        boxlist2: BoxList holding M boxes
        scope: name scope.
        Returns:
        a tensor with shape [N, M] representing pairwise iou scores.
        """
        intersections = self._intersection(boxlist1, boxlist2)
        areas1 = self._area(boxlist1)
        areas2 = self._area(boxlist2)
        unions = (
            tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
        return tf.where(
            tf.equal(intersections, 0.0),
            tf.zeros_like(intersections), tf.truediv(intersections, unions))

    def get_anchors(self):
        return self.anchors

    def matching(self, pos_thresh, neg_thresh, gt_bbox, gt_labels):
        pairwise_iou = self._iou(self.anchors, gt_bbox) # # size: [num_objects, num_priors]; anchors along the row and ground_truth clong the columns

        each_prior_max = tf.reduce_max(pairwise_iou, axis=-1) # size [num_priors]; iou with ground truth with the anchors
        each_prior_index = tf.math.argmax(pairwise_iou, axis=-1) # size [num_priors]; id of groud truth having max iou with the anchors

        each_box_max = tf.reduce_max(pairwise_iou, axis=0)
        each_box_index = tf.math.argmax(pairwise_iou, axis=0)

        # For the max IoU prior for each gt box, set its IoU to 2. This ensures that it won't be filtered
        # in the threshold step even if the IoU is under the negative threshold. This is because that we want
        # at least one prior to match with each gt box or else we'd be wasting training data.

        indices = tf.expand_dims(each_box_index,axis=-1)

        updates = tf.cast(tf.tile(tf.constant([2]), tf.shape(each_box_index)), dtype=tf.float32)
        each_prior_max = tf.tensor_scatter_nd_update(each_prior_max, indices, updates)

        # Set the index of the pair (prior, gt) we set the overlap for above.
        updates = tf.cast(tf.range(0,tf.shape(each_box_index)[0]),dtype=tf.int64)
        each_prior_index = tf.tensor_scatter_nd_update(each_prior_index, indices, updates)

        each_prior_box = tf.gather(gt_bbox, each_prior_index) # size: [num_priors, 4]
        conf = tf.squeeze(tf.gather(gt_labels, each_prior_index) + 1) # the class of the max IoU gt box for each prior, size: [num_priors]


        neutral_label_index = tf.where(each_prior_max < pos_thresh)
        background_label_index = tf.where(each_prior_max < neg_thresh)

        conf = tf.tensor_scatter_nd_update(conf, neutral_label_index, -1*tf.ones(tf.size(neutral_label_index), dtype=tf.int64))
        conf = tf.tensor_scatter_nd_update(conf, background_label_index, tf.zeros(tf.size(background_label_index), dtype=tf.int64))

        offsets = self._encode(each_prior_box, self.anchors)

        return offsets, conf, each_prior_box, each_prior_index
