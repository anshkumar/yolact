"""
Arthor: Vedanshu
"""

import tensorflow as tf

def bboxes_intersection(bbox_ref, bboxes):
    """Compute relative intersection between a reference box and a
    collection of bounding boxes. Namely, compute the quotient between
    intersection area and box area.
    Args:
      bbox_ref: (N, 4) or (4,) Tensor with reference bounding box(es).
      bboxes: (N, 4) Tensor, collection of bounding boxes.
    Return:
      (N,) Tensor with relative intersection.
    """

    # Should be more efficient to first transpose.
    bboxes = tf.transpose(bboxes)
    bbox_ref = tf.transpose(bbox_ref)
    # Intersection bbox and volume.
    int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
    int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
    int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
    int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
    h = tf.maximum(int_ymax - int_ymin, 0.)
    w = tf.maximum(int_xmax - int_xmin, 0.)
    # Volumes.
    inter_vol = h * w
    bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])

    return tf.where(
        tf.equal(bboxes_vol, 0.0),
        tf.zeros_like(inter_vol), inter_vol / bboxes_vol)


def normalize_image(image,
                    offset=(0.485, 0.456, 0.406),
                    scale=(0.229, 0.224, 0.225)):
    """Normalizes the image to zero mean and unit variance.
     ref: https://github.com/tensorflow/models/blob/3462436c91897f885e3593f0955d24cbe805333d/official/vision/detection/utils/input_utils.py
  """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    offset = tf.constant(offset)
    offset = tf.expand_dims(offset, axis=0)
    offset = tf.expand_dims(offset, axis=0)
    image -= offset

    scale = tf.constant(scale)
    scale = tf.expand_dims(scale, axis=0)
    scale = tf.expand_dims(scale, axis=0)
    image /= scale
    image *= 255
    return image


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

def sanitize_coordinates(_x1, _x2, img_size, padding = 0):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size

    x1 = tf.math.minimum(_x1, _x2)
    x2 = tf.math.maximum(_x1, _x2)
    x1 = tf.clip_by_value(x1 - padding, clip_value_min=0, clip_value_max=img_size)
    x2 = tf.clip_by_value(x2 + padding, clip_value_min=0, clip_value_max=img_size)

    return x1, x2


# crop the prediction of mask so as to calculate the linear combination mask loss
def crop(mask_p, boxes, padding = 1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Args:
        - mask_p should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    x1, x2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], tf.cast(tf.shape(mask_p)[1], tf.float32), padding)
    y1, y2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], tf.cast(tf.shape(mask_p)[0], tf.float32), padding)

    rows = tf.reshape(tf.range(tf.shape(mask_p)[0], dtype=x1.dtype), (-1, 1, 1))
    cols = tf.reshape(tf.range(tf.shape(mask_p)[1], dtype=x1.dtype), (1, -1, 1))

    cols = tf.broadcast_to(cols, (tf.shape(mask_p)[0], tf.shape(mask_p)[1], tf.shape(mask_p)[2]))
    rows = tf.broadcast_to(rows, (tf.shape(mask_p)[0], tf.shape(mask_p)[1], tf.shape(mask_p)[2]))

    mask_left = tf.cast(cols, tf.float32) >= tf.reshape(x1, (1, 1, -1))
    mask_right = tf.cast(cols, tf.float32) <= tf.reshape(x2, (1, 1, -1))
    mask_bottom = tf.cast(rows, tf.float32) >= tf.reshape(y1, (1, 1, -1))
    mask_top = tf.cast(rows, tf.float32) <= tf.reshape(y2, (1, 1, -1))

    crop_mask = tf.math.logical_and(tf.math.logical_and(mask_left, mask_right), tf.math.logical_and(mask_bottom, mask_top))
    crop_mask = tf.cast(crop_mask, tf.float32)

    return mask_p * crop_mask

# decode the offset back to center form bounding box when evaluation and prediction
def map_to_bbox(x):
    pass
