import numpy as np
import tensorflow as tf
from utils import standard_fields

class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        
        self.use_cross_class_nms = False
        self.use_fast_nms = False
        self.max_output_size = 300

    def __call__(self, net_outs, trad_nms=True):
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
        anchors = net_outs['priors']  # [27429, 4]
        proto_p = net_outs['proto_out']  # [1, 90, 302, 32]
        
        proto_h = tf.shape(proto_p)[1]
        proto_w = tf.shape(proto_p)[2]

        box_decode = self._decode(box_p, anchors)  # [1, 27429, 4]
        
        num_class = tf.shape(class_p)[2] - 1

        # Apply softmax to the prediction class
        class_p = tf.nn.softmax(class_p, axis=-1)
        # exclude the background class
        class_p = class_p[:, :, 1:]
        # get the max score class of 27429 predicted boxes
        class_p_max = tf.reduce_max(class_p, axis=-1)  # [1, 27429]
        batch_size = tf.shape(class_p_max)[0]

        # Not using python list here, as tf Autograph has some issues with it
        # https://github.com/tensorflow/tensorflow/issues/37512#issuecomment-600776581
        detection_boxes = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        detection_classes = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        detection_scores = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        detection_masks = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        num_detections = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

        for b in range(batch_size):
            # filter predicted boxes according the class score
            class_thre = tf.boolean_mask(class_p[b], class_p_max[b] > 0.3)
            box_thre = tf.boolean_mask(box_decode[b], class_p_max[b] > 0.3) 
            coef_thre = tf.boolean_mask(coef_p[b], class_p_max[b] > 0.3)

            if tf.size(class_thre) == 0:
                # TODO: Check this
                detection_boxes = detection_boxes.write(detection_boxes.size(), tf.zeros((self.max_output_size, 4)))
                detection_classes = detection_classes.write(detection_classes.size(), tf.zeros((self.max_output_size)))
                detection_scores = detection_scores.write(detection_scores.size(),  tf.zeros((self.max_output_size)))
                detection_masks = detection_masks.write(detection_masks.size(), tf.zeros((self.max_output_size, proto_h, proto_w)))
                num_detections = num_detections.write(num_detections.size(), tf.constant(0))
            else:
                if not trad_nms:
                    box_thre, coef_thre, class_ids, class_thre = _fast_nms(box_thre, coef_thre, class_thre)
                else:
                    box_thre, coef_thre, class_ids, class_thre = self._traditional_nms(box_thre, coef_thre, class_thre)

                # Padding with zeroes to reach max_output_size
                class_ids = tf.concat([class_ids, tf.zeros(self.max_output_size - tf.shape(box_thre)[0])], 0)
                class_thre = tf.concat([class_thre, tf.zeros(self.max_output_size - tf.shape(box_thre)[0])], 0)
                num_detection = tf.shape(box_thre)[0]
                pad_num_detection = self.max_output_size - num_detection

                _masks_coef = tf.matmul(proto_p[b], tf.transpose(coef_thre))
                _masks_coef = tf.sigmoid(_masks_coef) # [138, 138, NUM_BOX]

                boxes, masks = self._sanitize(_masks_coef, box_thre)
                masks = tf.transpose(masks, (2,0,1))
                paddings = tf.convert_to_tensor( [[0, pad_num_detection], [0,0], [0, 0]])
                masks = tf.pad(masks, paddings, "CONSTANT")
                
                paddings = tf.convert_to_tensor( [[0, pad_num_detection], [0, 0]])
                boxes = tf.pad(boxes, paddings, "CONSTANT")

                detection_boxes = detection_boxes.write(detection_boxes.size(), boxes)
                detection_classes = detection_classes.write(detection_classes.size(), class_ids)
                detection_scores = detection_scores.write(detection_scores.size(), class_thre)
                detection_masks = detection_masks.write(detection_masks.size(), masks)
                num_detections = num_detections.write(num_detections.size(), num_detection)

        detection_boxes = detection_boxes.stack()
        detection_classes = detection_classes.stack()
        detection_scores = detection_scores.stack()
        detection_masks = detection_masks.stack()
        num_detections = num_detections.stack()
        
        result = {'detection_boxes': detection_boxes,'detection_classes': detection_classes, 'detection_scores': detection_scores, 'detection_masks': detection_masks, 'num_detections': num_detections}
        return result

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

    def _sanitize_coordinates(self, _x1, _x2, padding: int = 0):
        """
        Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
        Also converts from relative to absolute coordinates and casts the results to long tensors.
        Warning: this does things in-place behind the scenes so copy if necessary.
        """
        x1 = tf.math.minimum(_x1, _x2)
        x2 = tf.math.maximum(_x1, _x2)
        x1 = tf.clip_by_value(x1 - padding, clip_value_min=0.0, clip_value_max=tf.cast(1.0,tf.float32))
        x2 = tf.clip_by_value(x2 + padding, clip_value_min=0.0, clip_value_max=tf.cast(1.0,tf.float32))

        # Normalize the coordinates
        return x1, x2

    def _sanitize(self, masks, boxes, padding: int = 0, crop_size=(30,30)):
        """
        "Crop" predicted masks by zeroing out everything not in the predicted bbox.
        Args:
            - masks should be a size [h, w, n] tensor of masks
            - boxes should be a size [n, 4] tensor of bbox coords in relative point form
        """
        # h, w, n = masks.shape
        
        x1, x2 = self._sanitize_coordinates(boxes[:, 1], boxes[:, 3], padding)
        y1, y2 = self._sanitize_coordinates(boxes[:, 0], boxes[:, 2], padding)

        # Making adjustments for tf.image.crop_and_resize
        boxes = tf.stack((y1, x1, y2, x2), axis=1)

        # box_indices = tf.zeros(tf.shape(boxes)[0], dtype=tf.int32) # All the boxes belong to a single batch
        # masks = tf.expand_dims(tf.transpose(masks, (2,0,1)), axis=-1)
        # masks = tf.image.crop_and_resize(masks, boxes, box_indices, crop_size)

        return boxes, masks

    def _traditional_nms(self, boxes, masks, scores, iou_threshold=0.5, score_threshold=0.3, max_class_output_size=100, max_output_size=300, soft_nms_sigma=0.5):
        num_classes = tf.shape(scores)[1]
        # List won't work as for now
        # https://github.com/tensorflow/tensorflow/issues/37512#issuecomment-600776581
        box_lst_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        mask_lst_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        cls_lst_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        scr_lst_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        for _cls in range(num_classes):
            cls_scores = scores[:, _cls]
            selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
                boxes, 
                cls_scores, 
                max_output_size=max_class_output_size, 
                iou_threshold=iou_threshold, 
                score_threshold=score_threshold,
                soft_nms_sigma=soft_nms_sigma)
            
            box_lst_arr = box_lst_arr.write(box_lst_arr.size(), tf.gather(boxes, selected_indices))
            mask_lst_arr = mask_lst_arr.write(mask_lst_arr.size(), tf.gather(masks, selected_indices))
            cls_lst_arr = cls_lst_arr.write(cls_lst_arr.size(), tf.gather(cls_scores, selected_indices) * 0.0 + tf.cast(_cls, dtype=tf.float32) + 1.0) # class ID starting from 1
            scr_lst_arr = scr_lst_arr.write(scr_lst_arr.size(), tf.gather(cls_scores, selected_indices))

        boxes = box_lst_arr.stack()[0]
        masks = mask_lst_arr.stack()[0]
        classes = cls_lst_arr.stack()[0]
        scores = scr_lst_arr.stack()[0]

        _ids = tf.argsort(scores, direction='DESCENDING')
        scores = tf.gather(scores, _ids)[:max_output_size]
        boxes = tf.gather(boxes, _ids)[:max_output_size]
        masks = tf.gather(masks, _ids)[:max_output_size]
        classes = tf.gather(classes, _ids)[:max_output_size]

        return boxes, masks, classes, scores

