"""
Arthor: Vedanshu
"""

import tensorflow as tf

def _area(boxlist, scope=None):
    # https://github.com/tensorflow/models/blob/831281cedfc8a4a0ad7c0c37173
    # 963fafb99da37/official/vision/detection/utils/object_detection/
    # box_list_ops.py#L48

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

def _intersection(boxlist1, boxlist2, scope=None):
    # https://github.com/tensorflow/models/blob/831281cedfc8a4a0ad7c0c37173
    # 963fafb99da37/official/vision/detection/utils/object_detection/
    # box_list_ops.py#L209

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
    intersect_heights = tf.maximum(0.0, 
        all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, 
        all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths

def _iou(boxlist1, boxlist2, scope=None):
    # https://github.com/tensorflow/models/blob/831281cedfc8a4a0ad7c0c37173
    # 963fafb99da37/official/vision/detection/utils/object_detection/
    # box_list_ops.py#L259

    """Computes pairwise intersection-over-union between box collections.
    Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.
    Returns:
    a tensor with shape [N, M] representing pairwise iou scores.
    """
    intersections = _intersection(boxlist1, boxlist2)
    areas1 = _area(boxlist1)
    areas2 = _area(boxlist2)
    unions = (tf.expand_dims(areas1, 1) + tf.expand_dims(
        areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))

# mapping from [ymin, xmin, ymax, xmax] to [cx, cy, w, h]
def map_to_center_form(x):
    h = x[:, 2] - x[:, 0]
    w = x[:, 3] - x[:, 1]
    cy = x[:, 0] + (h / 2)
    cx = x[:, 1] + (w / 2)
    return tf.stack([cx, cy, w, h], axis=-1)


# encode the gt and anchors to offset
def map_to_offset(x):
    g_hat_cx = (x[0, 0] - x[0, 1]) / x[2, 1]
    g_hat_cy = (x[1, 0] - x[1, 1]) / x[3, 1]
    g_hat_w = tf.math.log(x[2, 0] / x[2, 1])
    g_hat_h = tf.math.log(x[3, 0] / x[3, 1])
    return tf.stack([g_hat_cx, g_hat_cy, g_hat_w, g_hat_h])

def sanitize_coordinates(_x1, _x2, img_size, normalized, padding = 0.0):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    if normalized:
        _x1 = _x1 * img_size
        _x2 = _x2 * img_size

    x1 = tf.math.minimum(_x1, _x2)
    x2 = tf.math.maximum(_x1, _x2)
    x1 = tf.clip_by_value(x1 - padding, clip_value_min=0.0, clip_value_max=img_size)
    x2 = tf.clip_by_value(x2 + padding, clip_value_min=0.0, clip_value_max=img_size)

    return x1, x2


# crop the prediction of mask so as to calculate the linear combination mask loss
def crop(mask_p, boxes, padding = 1, normalized=True):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Args:
        - mask_p should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    x1, x2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], tf.cast(tf.shape(mask_p)[1], tf.float32), padding, normalized)
    y1, y2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], tf.cast(tf.shape(mask_p)[0], tf.float32), padding, normalized)

    rows = tf.reshape(tf.range(tf.shape(mask_p)[0], dtype=x1.dtype), (-1, 1, 1))
    cols = tf.reshape(tf.range(tf.shape(mask_p)[1], dtype=x1.dtype), (1, -1, 1))

    cols = tf.broadcast_to(cols, (tf.shape(mask_p)[0], tf.shape(mask_p)[1], tf.shape(mask_p)[2]))
    rows = tf.broadcast_to(rows, (tf.shape(mask_p)[0], tf.shape(mask_p)[1], tf.shape(mask_p)[2]))

    mask_left = tf.cast(cols, tf.float32) >= tf.reshape(x1, (1, 1, -1))
    mask_right = tf.cast(cols, tf.float32) < tf.reshape(x2, (1, 1, -1))
    mask_bottom = tf.cast(rows, tf.float32) >= tf.reshape(y1, (1, 1, -1))
    mask_top = tf.cast(rows, tf.float32) < tf.reshape(y2, (1, 1, -1))

    crop_mask = tf.math.logical_and(tf.math.logical_and(mask_left, mask_right), tf.math.logical_and(mask_bottom, mask_top))
    crop_mask = tf.cast(crop_mask, tf.float32)

    return mask_p * crop_mask

# decode the offset back to center form bounding box when evaluation and prediction
def map_to_bbox(x):
    pass

def _batch_decode(box_p, priors, include_variances=True):
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

def _decode(box_p, priors, include_variances=True):
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

def _encode(map_loc, anchors, include_variances=True):
    # For variance in priorbox layer:
    # https://github.com/weiliu89/caffe/issues/155

    # center_gt = tf.map_fn(lambda x: map_to_center_form(x), map_loc)
    # center_anchors in [ymin, xmin, ymax, xmax ]
    # map_loc in [ymin, xmin, ymax, xmax ]
    gh = map_loc[:, 2] - map_loc[:, 0]
    gw = map_loc[:, 3] - map_loc[:, 1]
    center_gt = tf.cast(tf.stack(
        [map_loc[:, 1] + (gw / 2), 
        map_loc[:, 0] + (gh / 2), gw, gh], 
        axis=-1), tf.float32)

    ph = anchors[:, 2] - anchors[:, 0]
    pw = anchors[:, 3] - anchors[:, 1]
    center_anchors = tf.cast(tf.stack(
        [anchors[:, 1] + (pw / 2), 
        anchors[:, 0] + (ph / 2), pw, ph], 
        axis=-1), tf.float32)
    variances = [0.1, 0.2]

    # calculate offset
    if include_variances:
        g_hat_cx = (center_gt[:, 0] - center_anchors[:, 0]
            ) / center_anchors[:, 2] / variances[0]
        g_hat_cy = (center_gt[:, 1] - center_anchors[:, 1]
            ) / center_anchors[:, 3] / variances[0]
    else:
        g_hat_cx = (center_gt[:, 0] - center_anchors[:, 0]
            ) / center_anchors[:, 2]
        g_hat_cy = (center_gt[:, 1] - center_anchors[:, 1]
            ) / center_anchors[:, 3]
    tf.debugging.assert_non_negative(center_anchors[:, 2] / center_gt[:, 2])
    tf.debugging.assert_non_negative(center_anchors[:, 3] / center_gt[:, 3])
    if include_variances:
        g_hat_w = tf.math.log(center_gt[:, 2] / center_anchors[:, 2]
            ) / variances[1]
        g_hat_h = tf.math.log(center_gt[:, 3] / center_anchors[:, 3]
            ) / variances[1]
    else:
        g_hat_w = tf.math.log(center_gt[:, 2] / center_anchors[:, 2])
        g_hat_h = tf.math.log(center_gt[:, 3] / center_anchors[:, 3])
    tf.debugging.assert_all_finite(g_hat_cx, 
        "Ground truth box x encoding NaN/Inf")
    tf.debugging.assert_all_finite(g_hat_cy, 
        "Ground truth box y encoding NaN/Inf")
    tf.debugging.assert_all_finite(g_hat_w, 
        "Ground truth box width encoding NaN/Inf")
    tf.debugging.assert_all_finite(g_hat_h, 
        "Ground truth box height encoding NaN/Inf")
    offsets = tf.stack([g_hat_cx, g_hat_cy, g_hat_w, g_hat_h], axis=-1)
    
    return offsets
