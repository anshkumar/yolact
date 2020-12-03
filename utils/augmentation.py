import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from utils import utils
import math

"""
Ref: https://github.com/balancap/SSD-Tensorflow/blob/master/preprocessing/ssd_vgg_preprocessing.py
"""


def geometric_distortion(img, bboxes, masks, output_size, classes):
    # Geometric Distortions (img, bbox, mask)
    # Each bounding box has shape [batch, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].
    bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
        tf.shape(img),
        bounding_boxes=tf.expand_dims(bboxes, 0),
        min_object_covered=1,
        aspect_ratio_range=(0.5, 2),
        area_range=(0.1, 1.0),
        max_attempts=100)
    # the distort box is the area of the cropped image, original image will be [0, 0, 1, 1]
    distort_bbox = distort_bbox[0, 0]
    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(img, bbox_begin, bbox_size)
    # Restore the shape since the dynamic slice loses 3rd dimension.
    cropped_image.set_shape([None, None, 3])

    # cropped the mask
    bbox_begin = tf.concat([[0], bbox_begin], axis=0)
    bbox_size = tf.concat([[-1], bbox_size], axis=0)
    cropped_masks = tf.slice(masks, bbox_begin, bbox_size)
    cropped_masks.set_shape([None, None, None, 1])

    # resize the scale of bboxes for cropped image
    v = tf.stack([distort_bbox[0], distort_bbox[1], distort_bbox[0], distort_bbox[1]])
    bboxes = bboxes - v
    s = tf.stack([distort_bbox[2] - distort_bbox[0],
                  distort_bbox[3] - distort_bbox[1],
                  distort_bbox[2] - distort_bbox[0],
                  distort_bbox[3] - distort_bbox[1]])
    bboxes = bboxes / s

    # filter out
    scores = utils.bboxes_intersection(tf.constant([0, 0, 1, 1], bboxes.dtype), bboxes)
    bool_mask = scores > 0.5
    classes = tf.boolean_mask(classes, bool_mask)
    bboxes = tf.boolean_mask(bboxes, bool_mask)

    # deal with negative value of bbox
    bboxes = tf.clip_by_value(bboxes, clip_value_min=0, clip_value_max=1)

    cropped_masks = tf.boolean_mask(cropped_masks, bool_mask)
    # resize cropped to output size
    cropped_image = tf.image.resize(cropped_image, [output_size[0], output_size[1]], method=tf.image.ResizeMethod.BILINEAR)

    return cropped_image, bboxes, cropped_masks, classes


def photometric_distortion(image):
    color_ordering = tf.random.uniform([1], minval=0, maxval=4)[0]

    if color_ordering < 1 and color_ordering > 0:
        image = tf.image.random_brightness(image, max_delta=32.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    elif color_ordering < 2 and color_ordering > 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    elif color_ordering < 3 and color_ordering > 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    else:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32.)

    return tf.clip_by_value(image, 0.0, 255.0)

def _clip_bbox(min_y, min_x, max_y, max_x):
  # https://github.com/tensorflow/models/blob/e3f8ea2227ef5ce67df04bd175e6c20711079d8f/research/object_detection/utils/autoaugment_utils.py#L445
  """Clip bounding box coordinates between 0 and 1.

  Args:
    min_y: Normalized bbox coordinate of type float between 0 and 1.
    min_x: Normalized bbox coordinate of type float between 0 and 1.
    max_y: Normalized bbox coordinate of type float between 0 and 1.
    max_x: Normalized bbox coordinate of type float between 0 and 1.

  Returns:
    Clipped coordinate values between 0 and 1.
  """
  min_y = tf.clip_by_value(min_y, 0.0, 1.0)
  min_x = tf.clip_by_value(min_x, 0.0, 1.0)
  max_y = tf.clip_by_value(max_y, 0.0, 1.0)
  max_x = tf.clip_by_value(max_x, 0.0, 1.0)
  return min_y, min_x, max_y, max_x

def _check_bbox_area(min_y, min_x, max_y, max_x, delta=0.05):
  # https://github.com/tensorflow/models/blob/e3f8ea2227ef5ce67df04bd175e6c20711079d8f/research/object_detection/utils/autoaugment_utils.py#L464
  """Adjusts bbox coordinates to make sure the area is > 0.

  Args:
    min_y: Normalized bbox coordinate of type float between 0 and 1.
    min_x: Normalized bbox coordinate of type float between 0 and 1.
    max_y: Normalized bbox coordinate of type float between 0 and 1.
    max_x: Normalized bbox coordinate of type float between 0 and 1.
    delta: Float, this is used to create a gap of size 2 * delta between
      bbox min/max coordinates that are the same on the boundary.
      This prevents the bbox from having an area of zero.

  Returns:
    Tuple of new bbox coordinates between 0 and 1 that will now have a
    guaranteed area > 0.
  """
  height = max_y - min_y
  width = max_x - min_x
  def _adjust_bbox_boundaries(min_coord, max_coord):
    # Make sure max is never 0 and min is never 1.
    max_coord = tf.maximum(max_coord, 0.0 + delta)
    min_coord = tf.minimum(min_coord, 1.0 - delta)
    return min_coord, max_coord
  min_y, max_y = tf.cond(tf.equal(height, 0.0),
                         lambda: _adjust_bbox_boundaries(min_y, max_y),
                         lambda: (min_y, max_y))
  min_x, max_x = tf.cond(tf.equal(width, 0.0),
                         lambda: _adjust_bbox_boundaries(min_x, max_x),
                         lambda: (min_x, max_x))
  return min_y, min_x, max_y, max_x


def rotate(image, degrees):
  # https://github.com/tensorflow/models/blob/e3f8ea2227ef5ce67df04bd175e6c20711079d8f/research/object_detection/utils/autoaugment_utils.py#L305
  """Rotates the image by degrees either clockwise or counterclockwise.

  Args:
    image: An image Tensor of type uint8.
    degrees: Float, a scalar angle in degrees to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.

  Returns:
    The rotated version of image.
  """
  # Convert from degrees to radians.
  degrees_to_radians = math.pi / 180.0
  radians = degrees * degrees_to_radians

  # In practice, we should randomize the rotation degrees by flipping
  # it negatively half the time, but that's done on 'degrees' outside
  # of the function.

  image = tfa.image.rotate(image, radians)
  return image

def _rotate_bbox(bbox, image_height, image_width, degrees):
  # https://github.com/tensorflow/models/blob/e3f8ea2227ef5ce67df04bd175e6c20711079d8f/research/object_detection/utils/autoaugment_utils.py#L795
  """Rotates the bbox coordinated by degrees.

  Args:
    bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
      of type float that represents the normalized coordinates between 0 and 1.
    image_height: Int, height of the image.
    image_width: Int, height of the image.
    degrees: Float, a scalar angle in degrees to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.

  Returns:
    A tensor of the same shape as bbox, but now with the rotated coordinates.
  """
  image_height, image_width = (
      tf.cast(image_height, tf.float32), tf.cast(image_width, tf.float32))

  # Convert from degrees to radians.
  degrees_to_radians = math.pi / 180.0
  radians = degrees * degrees_to_radians

  # Translate the bbox to the center of the image and turn the normalized 0-1
  # coordinates to absolute pixel locations.
  # Y coordinates are made negative as the y axis of images goes down with
  # increasing pixel values, so we negate to make sure x axis and y axis points
  # are in the traditionally positive direction.
  min_y = -tf.cast(image_height * (bbox[0] - 0.5), tf.int32)
  min_x = tf.cast(image_width * (bbox[1] - 0.5), tf.int32)
  max_y = -tf.cast(image_height * (bbox[2] - 0.5), tf.int32)
  max_x = tf.cast(image_width * (bbox[3] - 0.5), tf.int32)
  coordinates = tf.stack(
      [[min_y, min_x], [min_y, max_x], [max_y, min_x], [max_y, max_x]])
  coordinates = tf.cast(coordinates, tf.float32)
  # Rotate the coordinates according to the rotation matrix clockwise if
  # radians is positive, else negative
  rotation_matrix = tf.stack(
      [[tf.cos(radians), tf.sin(radians)],
       [-tf.sin(radians), tf.cos(radians)]])
  new_coords = tf.cast(
      tf.matmul(rotation_matrix, tf.transpose(coordinates)), tf.int32)
  # Find min/max values and convert them back to normalized 0-1 floats.
  min_y = -(tf.cast(tf.reduce_max(new_coords[0, :]), tf.float32) / image_height - 0.5)
  min_x = tf.cast(tf.reduce_min(new_coords[1, :]), tf.float32) / image_width + 0.5
  max_y = -(tf.cast(tf.reduce_min(new_coords[0, :]), tf.float32) / image_height - 0.5)
  max_x = tf.cast(tf.reduce_max(new_coords[1, :]), tf.float32) / image_width + 0.5

  # Clip the bboxes to be sure the fall between [0, 1].
  min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
  min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
  return tf.stack([min_y, min_x, max_y, max_x])


def rotate_with_bboxes(image, mask, bboxes, degrees):
  # https://github.com/tensorflow/models/blob/e3f8ea2227ef5ce67df04bd175e6c20711079d8f/research/object_detection/utils/autoaugment_utils.py#L848
  """Equivalent of PIL Rotate that rotates the image and bbox.

  Args:
    image: 3D uint8 Tensor.
    bboxes: 2D Tensor that is a list of the bboxes in the image. Each bbox
      has 4 elements (min_y, min_x, max_y, max_x) of type float.
    degrees: Float, a scalar angle in degrees to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.

  Returns:
    A tuple containing a 3D uint8 Tensor that will be the result of rotating
    image by degrees. The second element of the tuple is bboxes, where now
    the coordinates will be shifted to reflect the rotated image.
  """
  # Rotate the image.
  image = rotate(image, degrees)
  mask = rotate(mask, degrees)

  # Convert bbox coordinates to pixel values.
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  # pylint:disable=g-long-lambda
  wrapped_rotate_bbox = lambda bbox: _rotate_bbox(
      bbox, image_height, image_width, degrees)
  # pylint:enable=g-long-lambda
  bboxes = tf.map_fn(wrapped_rotate_bbox, bboxes)
  return image, bboxes, mask


def random_augmentation(img, bboxes, masks, output_size, proto_output_size, classes):
    # generate random
    FLAGS = tf.random.uniform([3], minval=0, maxval=1)
    FLAG_GEO_DISTORTION = FLAGS[0]
    FLAG_PHOTO_DISTORTION = FLAGS[1]
    FLAG_HOR_FLIP = FLAGS[2]

    # Random Geometric Distortion (img, bboxes, masks)
    # if FLAG_GEO_DISTORTION > 0.5:
    #     img, bboxes, masks, classes = geometric_distortion(img, bboxes, masks, output_size, classes)

    # Random Photometric Distortions (img)
    # if FLAG_PHOTO_DISTORTION > 0.5:
    #     img = photometric_distortion(img)

    # if FLAG_HOR_FLIP > 0.5:
    rand_angle = tf.random.uniform([1], minval=0, maxval=360)[0]
    # tf.print("Augmention angle: ", rand_angle)
    img, bboxes, masks = rotate_with_bboxes(img, masks, bboxes, rand_angle)

    return img, bboxes, masks, classes
