import numpy as np
import tensorflow as tf
from utils import standard_fields
from utils import utils

class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """

    def __init__(self, num_classes, max_output_size, per_class_max_output_size, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.use_fast_nms = False
        self.max_output_size = 300
        self.per_class_max_output_size = 100

    def __call__(self, net_outs, img_shape, trad_nms=True, use_cropped_mask=True):
        """
        Args:
             pred_offset: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            pred_cls: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            pred_mask_coef: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            priors: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_out: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]
        
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.
            Note that the outputs are sorted only if cross_class_nms is False
        """

        box_p = net_outs['pred_offset']  # [1, 27429, 4]
        class_p = net_outs['pred_cls']  # [1, 27429, 2]
        coef_p = net_outs['pred_mask_coef']  # [1, 27429, 32]
        anchors = net_outs['priors']  # [27429, 4] [cx, cy, w, h] format. Unnormalized.
        proto_p = net_outs['proto_out']  # [1, 90, 302, 32]
        
        proto_h = tf.shape(proto_p)[1]
        proto_w = tf.shape(proto_p)[2]

        num_class = tf.shape(class_p)[2] - 1

        # Apply softmax to the prediction class
        class_p = tf.nn.softmax(class_p, axis=-1)
        # exclude the background class
        class_p = class_p[:, :, 1:]

        # get the max score class of 27429 predicted boxes
        class_p_max = tf.reduce_max(class_p, axis=-1)  # [1, 27429]
        batch_size = tf.shape(class_p_max)[0]

        detection_boxes = tf.zeros((batch_size, self.max_output_size, 4), tf.float32)
        detection_classes = tf.zeros((batch_size, self.max_output_size), tf.float32)
        detection_scores = tf.zeros((batch_size, self.max_output_size), tf.float32)
        detection_masks = tf.zeros((batch_size, self.max_output_size, proto_h, proto_w), tf.float32)
        num_detections = tf.zeros((batch_size), tf.int32)

        for b in range(batch_size):
            # filter predicted boxes according the class score
            class_thre = tf.boolean_mask(class_p[b], class_p_max[b] > self.conf_thresh)
            coef_thre = tf.boolean_mask(coef_p[b], class_p_max[b] > self.conf_thresh)
            raw_boxes = tf.boolean_mask(box_p[b], class_p_max[b] > self.conf_thresh)
            raw_anchors = tf.boolean_mask(anchors, class_p_max[b] > self.conf_thresh)

            # decode only selected boxes
            boxes = self._decode(raw_boxes, raw_anchors)  # [27429, 4]

            if tf.size(class_thre) != 0:
                if not trad_nms:
                    boxes, coef_thre, class_ids, class_thre = _fast_nms(boxes, coef_thre, class_thre)
                else:
                    boxes, coef_thre, class_ids, class_thre = self._traditional_nms(boxes, coef_thre, class_thre, score_threshold=self.conf_thresh, iou_threshold=self.nms_thresh, max_class_output_size=self.per_class_max_output_size)

                num_detection = [tf.shape(boxes)[0]]

                masks = tf.matmul(proto_p[b], tf.transpose(coef_thre))
                masks = tf.sigmoid(masks) # [138, 138, NUM_BOX]

                # boxes = self._sanitize(boxes, width=img_shape[2], height=img_shape[1])
                # boxes = tf.stack([
                #     boxes[:, 0]/tf.cast(img_shape[1], tf.float32), 
                #     boxes[:, 1]/tf.cast(img_shape[2], tf.float32),
                #     boxes[:, 2]/tf.cast(img_shape[1], tf.float32),
                #     boxes[:, 3]/tf.cast(img_shape[2], tf.float32)
                #     ], axis=-1)

                boxes = self._sanitize(boxes, width=1, height=1)

                if use_cropped_mask:
                    masks = utils.crop(masks, boxes)

                masks = tf.clip_by_value(masks, clip_value_min=0.0, 
                    clip_value_max=1.0)
                masks = tf.transpose(masks, (2,0,1))

                _ind_boxes = tf.stack((tf.tile([b], num_detection), tf.range(0, tf.shape(boxes)[0])), axis=-1) # Shape: (Number of updates, index of update)
                detection_boxes = tf.tensor_scatter_nd_update(detection_boxes, _ind_boxes, boxes)
                detection_classes = tf.tensor_scatter_nd_update(detection_classes, _ind_boxes, class_ids)
                detection_scores = tf.tensor_scatter_nd_update(detection_scores, _ind_boxes, class_thre)
                detection_masks = tf.tensor_scatter_nd_update(detection_masks, _ind_boxes, masks)
                num_detections = tf.tensor_scatter_nd_update(num_detections, [[b]], num_detection)

        result = {'detection_boxes': detection_boxes,'detection_classes': detection_classes, 'detection_scores': detection_scores, 'detection_masks': detection_masks, 'num_detections': num_detections}
        return result

    def _batch_decode(self, box_p, priors, include_variances=False):
        # https://github.com/feiyuhuahuo/Yolact_minimal/blob/9299a0cf346e455d672fadd796ac748871ba85e4/utils/box_utils.py#L151
        """
        Decode predicted bbox coordinates using the scheme
        employed at https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html
            b_x = prior_w*loc_x + prior_x
            b_y = prior_h*loc_y + prior_y
            b_w = prior_w * exp(loc_w)
            b_h = prior_h * exp(loc_h)
        
        Note that loc is inputed as [c_x, x_y, w, h]
        while priors are inputed as [c_x, c_y, w, h] where each coordinate
        is relative to size of the image.
        
        Also note that prior_x and prior_y are center coordinates.
        """
        variances = [0.1, 0.2]
        box_p = tf.cast(box_p, tf.float32)
        priors = tf.cast(priors, tf.float32)
        if include_variances:
            b_x_y = priors[:, :2] + box_p[:, :, :2] * priors[:, 2:]* variances[0]
            b_w_h = priors[:, 2:] * tf.math.exp(box_p[:, :, 2:]* variances[1])
        else:
            b_x_y = priors[:, :2] + box_p[:, :, :2] * priors[:, 2:]
            b_w_h = priors[:, 2:] * tf.math.exp(box_p[:, :, 2:])
        
        boxes = tf.concat([b_x_y, b_w_h], axis=-1)
        
        # [x_min, y_min, x_max, y_max]
        boxes = tf.concat([boxes[:, :, :2] - boxes[:, :, 2:] / 2, boxes[:, :, 2:] / 2 + boxes[:, :, :2]], axis=-1)
        
        # [y_min, x_min, y_max, x_max]
        return tf.stack([boxes[:, :, 1], boxes[:, :, 0],boxes[:, :, 3], boxes[:, :, 2]], axis=-1)

    def _decode(self, box_p, priors, include_variances=False):
        # https://github.com/feiyuhuahuo/Yolact_minimal/blob/9299a0cf346e455d672fadd796ac748871ba85e4/utils/box_utils.py#L151
        """
        Decode predicted bbox coordinates using the scheme
        employed at https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html
            b_x = prior_w*loc_x + prior_x
            b_y = prior_h*loc_y + prior_y
            b_w = prior_w * exp(loc_w)
            b_h = prior_h * exp(loc_h)
        
        Note that loc is inputed as [c_x, x_y, w, h]
        while priors are inputed as [c_x, c_y, w, h] where each coordinate
        is relative to size of the image.
        
        Also note that prior_x and prior_y are center coordinates.
        """
        variances = [0.1, 0.2]
        box_p = tf.cast(box_p, tf.float32)
        priors = tf.cast(priors, tf.float32)

        ph = priors[:, 2] - priors[:, 0]
        pw = priors[:, 3] - priors[:, 1]
        priors = tf.cast(tf.stack(
            [priors[:, 1] + (pw / 2), 
            priors[:, 0] + (ph / 2), pw, ph], 
            axis=-1), tf.float32)

        if include_variances:
            b_x_y = priors[:, :2] + box_p[:, :2] * priors[:, 2:]* variances[0]
            b_w_h = priors[:, 2:] * tf.math.exp(box_p[:, 2:]* variances[1])
        else:
            b_x_y = priors[:, :2] + box_p[:, :2] * priors[:, 2:]
            b_w_h = priors[:, 2:] * tf.math.exp(box_p[:, 2:])
        
        boxes = tf.concat([b_x_y, b_w_h], axis=-1)
        
        # [x_min, y_min, x_max, y_max]
        boxes = tf.concat([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, 2:] / 2 + boxes[:, :2]], axis=-1)
        
        # [y_min, x_min, y_max, x_max]
        return tf.stack([boxes[:, 1], boxes[:, 0],boxes[:, 3], boxes[:, 2]], axis=-1)

    def _sanitize_coordinates(self, _x1, _x2, size, padding: int = 0):
        """
        Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
        Also converts from relative to absolute coordinates and casts the results to long tensors.
        Warning: this does things in-place behind the scenes so copy if necessary.
        """
        x1 = tf.math.minimum(_x1, _x2)
        x2 = tf.math.maximum(_x1, _x2)
        x1 = tf.clip_by_value(x1 - padding, clip_value_min=0.0, clip_value_max=tf.cast(size,tf.float32))
        x2 = tf.clip_by_value(x2 + padding, clip_value_min=0.0, clip_value_max=tf.cast(size,tf.float32))

        # Normalize the coordinates
        return x1, x2

    def _sanitize(self, boxes, width, height,  padding: int = 0, crop_size=(30,30)):
        """
        "Crop" predicted masks by zeroing out everything not in the predicted bbox.
        Args:
            - masks should be a size [h, w, n] tensor of masks
            - boxes should be a size [n, 4] tensor of bbox coords in relative point form
        """        
        x1, x2 = self._sanitize_coordinates(boxes[:, 1], boxes[:, 3], width, padding)
        y1, y2 = self._sanitize_coordinates(boxes[:, 0], boxes[:, 2], height, padding)

        boxes = tf.stack((y1, x1, y2, x2), axis=1)

        return boxes

    def _traditional_nms(self, boxes, mask_coef, scores, iou_threshold=0.5, score_threshold=0.15, max_class_output_size=100, max_output_size=300, soft_nms_sigma=0.5):
        num_classes = tf.shape(scores)[1]

        _num_coef = tf.shape(mask_coef)[1]
        _boxes = tf.zeros((max_class_output_size*num_classes, 4), tf.float32)
        _coefs = tf.zeros((max_class_output_size*num_classes, _num_coef), tf.float32)
        _classes = tf.zeros((max_class_output_size*num_classes), tf.float32)
        _scores = tf.zeros((max_class_output_size*num_classes), tf.float32)

        for _cls in range(num_classes):
            cls_scores = scores[:, _cls]
            selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
                boxes, 
                cls_scores, 
                max_output_size=max_class_output_size, 
                iou_threshold=iou_threshold, 
                score_threshold=score_threshold,
                soft_nms_sigma=soft_nms_sigma)

            _update_boxes = tf.gather(boxes, selected_indices)
            _num_boxes = tf.shape(_update_boxes)[0]
            _ind_boxes = tf.range(_cls*max_class_output_size, _cls*max_class_output_size+_num_boxes)

            _boxes = tf.tensor_scatter_nd_update(_boxes, tf.expand_dims(_ind_boxes, axis=-1), _update_boxes)
            _coefs = tf.tensor_scatter_nd_update(_coefs, tf.expand_dims(_ind_boxes, axis=-1), tf.gather(mask_coef, selected_indices))
            _classes = tf.tensor_scatter_nd_update(_classes, tf.expand_dims(_ind_boxes, axis=-1), tf.gather(cls_scores, selected_indices) * 0.0 + tf.cast(_cls, dtype=tf.float32) + 1.0)
            _scores = tf.tensor_scatter_nd_update(_scores, tf.expand_dims(_ind_boxes, axis=-1), tf.gather(cls_scores, selected_indices))

        _ids = tf.argsort(_scores, direction='DESCENDING')
        scores = tf.gather(_scores, _ids)[:max_output_size]
        boxes = tf.gather(_boxes, _ids)[:max_output_size]
        mask_coef = tf.gather(_coefs, _ids)[:max_output_size]
        classes = tf.gather(_classes, _ids)[:max_output_size]

        return boxes, mask_coef, classes, scores

