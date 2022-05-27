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

    def __call__(self, net_outs, img_shape, trad_nms=False, use_cropped_mask=True):
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
            boxes = utils._decode(raw_boxes, raw_anchors)  # [27429, 4]

            if tf.size(class_thre) > 0:
                if not trad_nms:
                    boxes, coef_thre, class_ids, class_thre = self._cc_fast_nms(boxes, coef_thre, class_thre)
                else:
                    boxes, coef_thre, class_ids, class_thre = self._traditional_nms_v2(boxes, coef_thre, class_thre, score_threshold=self.conf_thresh, iou_threshold=self.nms_thresh)

                num_detection = [tf.shape(boxes)[0]]

                masks = tf.matmul(proto_p[b], tf.transpose(coef_thre))
                masks = tf.sigmoid(masks) # [138, 138, NUM_BOX]

                boxes = self._sanitize(boxes, width=img_shape[2], height=img_shape[1])
                boxes = tf.stack([
                    boxes[:, 0]/tf.cast(img_shape[1], tf.float32), 
                    boxes[:, 1]/tf.cast(img_shape[2], tf.float32),
                    boxes[:, 2]/tf.cast(img_shape[1], tf.float32),
                    boxes[:, 3]/tf.cast(img_shape[2], tf.float32)
                    ], axis=-1)

                # boxes = self._sanitize(boxes, width=1, height=1)

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

    def _sanitize(self, boxes, width, height,  padding: int = 0):
        """
        "Crop" predicted masks by zeroing out everything not in the predicted bbox.
        Args:
            - masks should be a size [h, w, n] tensor of masks
            - boxes should be a size [n, 4] tensor of bbox coords in relative point form
        """        
        x1, x2 = utils._sanitize_coordinates(boxes[:, 1], boxes[:, 3], width, normalized=False)
        y1, y2 = utils._sanitize_coordinates(boxes[:, 0], boxes[:, 2], height, normalized=False)

        boxes = tf.stack((y1, x1, y2, x2), axis=1)

        return boxes

    def _traditional_nms(self, boxes, mask_coef, scores, iou_threshold=0.5, score_threshold=0.05, max_class_output_size=100, max_output_size=300):
        num_classes = tf.shape(scores)[1]

        _num_coef = tf.shape(mask_coef)[1]
        _boxes = tf.zeros((max_class_output_size*num_classes, 4), tf.float32)
        _coefs = tf.zeros((max_class_output_size*num_classes, _num_coef), tf.float32)
        _classes = tf.zeros((max_class_output_size*num_classes), tf.float32)
        _scores = tf.zeros((max_class_output_size*num_classes), tf.float32)

        for _cls in range(num_classes):
            cls_scores = scores[:, _cls]
            selected_indices = tf.image.non_max_suppression(
                boxes, 
                cls_scores, 
                max_output_size=max_class_output_size,
                iou_threshold=iou_threshold, 
                score_threshold=score_threshold)

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

    def _traditional_nms_v2(self, boxes, mask_coef, scores, iou_threshold=0.5, score_threshold=0.05, max_output_size=300):
        selected_indices = tf.image.non_max_suppression(boxes, 
            tf.reduce_max(scores, axis=-1), 
            max_output_size=max_output_size, 
            iou_threshold=iou_threshold, 
            score_threshold=score_threshold)

        classes = tf.argmax(scores, axis=-1)+1
        boxes = tf.gather(boxes, selected_indices)
        scores = tf.gather(tf.reduce_max(scores, axis=-1), selected_indices)
        mask_coef = tf.gather(mask_coef, selected_indices)
        classes = tf.cast(tf.gather(classes, selected_indices), dtype=tf.float32)
        return boxes, mask_coef, classes, scores

    def _cc_fast_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=15):
        # Cross Class NMS
        # Collapse all the classes into 1 
        classes = tf.argmax(scores, axis=-1)+1
        scores = tf.reduce_max(scores, axis=-1)
        _, idx = tf.math.top_k(scores, k=tf.math.minimum(top_k, tf.shape(scores)[0]))
        boxes_idx = tf.gather(boxes, idx, axis=0)

        # Compute the pairwise IoU between the boxes
        iou = utils._iou(boxes_idx, boxes_idx)

        # Zero out the lower triangle of the cosine similarity matrix and diagonal
        iou = tf.linalg.band_part(iou, 0, -1) - tf.linalg.band_part(iou, 0, 0)

        # Now that everything in the diagonal and below is zeroed out, if we take the max
        # of the IoU matrix along the columns, each column will represent the maximum IoU
        # between this element and every element with a higher score than this element.
        iou_max = tf.reduce_max(iou, axis=0)

        # Now just filter out the ones greater than the threshold, i.e., only keep boxes that
        # don't have a higher scoring box that would supress it in normal NMS.
        idx_det = (iou_max <= iou_threshold)
        idx_det = tf.where(idx_det == True)

        classes = tf.gather_nd(classes, idx_det)
        boxes = tf.gather_nd(boxes, idx_det)
        masks = tf.gather_nd(masks, idx_det)
        scores = tf.gather_nd(scores, idx_det)

        return boxes, masks, classes, scores
    
    def _fast_nms(self, boxes, masks, scores, iou_threshold=0.5, top_k=100):
        if tf.rank(scores) == 1:
            scores = tf.expand_dims(scores, axis=-1)
            boxes = tf.expand_dims(boxes, axis=0)
            masks = tf.expand_dims(masks, axis=0)

        scores, idx = tf.math.top_k(scores, k=top_k)
        num_classes, num_dets = tf.shape(idx)[0], tf.shape(idx)[1]
        boxes = tf.gather(boxes, idx, axis=0)
        masks = tf.gather(masks, idx, axis=0)
        iou = utils._iou(boxes, boxes)
        # upper trangular matrix - diagnoal
        upper_triangular = tf.linalg.band_part(iou, 0, -1)
        diag = tf.linalg.band_part(iou, 0, 0)
        iou = upper_triangular - diag

        # fitler out the unwanted ROI
        iou_max = tf.reduce_max(iou, axis=1)
        idx_det = (iou_max <= iou_threshold)

        # second threshold
        # second_threshold = (iou_max <= self.conf_threshold)
        second_threshold = (scores > self.conf_threshold)
        idx_det = tf.where(tf.logical_and(idx_det, second_threshold) == True)
        classes = tf.broadcast_to(tf.expand_dims(tf.range(num_classes), axis=-1), tf.shape(iou_max))
        classes = tf.gather_nd(classes, idx_det)
        boxes = tf.gather_nd(boxes, idx_det)
        masks = tf.gather_nd(masks, idx_det)
        scores = tf.gather_nd(scores, idx_det)

        # number of max detection = 100 (u can choose whatever u want)
        max_num_detection = tf.math.minimum(self.max_num_detection, tf.size(scores))
        scores, idx = tf.math.top_k(scores, k=max_num_detection)

        # second threshold
        classes = tf.gather(classes, idx)
        boxes = tf.gather(boxes, idx)
        masks = tf.gather(masks, idx)

        return boxes, masks, classes, scores
