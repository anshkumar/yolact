import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from utils import utils
from utils import box_list
import math
import functools
import sys

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

def _flip_boxes_left_right(boxes):
  # https://github.com/tensorflow/models/blob/2986bcafb9eaa8fed4d78f17a04c4c5afc8f6691/official/vision/detection/utils/object_detection/preprocessor.py#L49
  """Left-right flip the boxes.

  Args:
    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].

  Returns:
    Flipped boxes.
  """
  ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
  flipped_xmin = tf.subtract(1.0, xmax)
  flipped_xmax = tf.subtract(1.0, xmin)
  flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], 1)
  return flipped_boxes

def _flip_masks_left_right(masks):
  # https://github.com/tensorflow/models/blob/2986bcafb9eaa8fed4d78f17a04c4c5afc8f6691/official/vision/detection/utils/object_detection/preprocessor.py#L67
  """Left-right flip masks.

  Args:
    masks: rank 3 float32 tensor with shape
      [num_instances, height, width] representing instance masks.

  Returns:
    flipped masks: rank 3 float32 tensor with shape
      [num_instances, height, width] representing instance masks.
  """
  return masks[:, :, ::-1]

def random_horizontal_flip(image,
                           boxes=None,
                           masks=None,
                           seed=None):
  # https://github.com/tensorflow/models/blob/2986bcafb9eaa8fed4d78f17a04c4c5afc8f6691/official/vision/detection/utils/object_detection/preprocessor.py#L182
  """Randomly flips the image and detections horizontally.

  The probability of flipping the image is 50%.

  Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    boxes: (optional) rank 2 float32 tensor with shape [N, 4]
           containing the bounding boxes.
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width] containing instance masks. The masks
           are of the same height, width as the input `image`.
    seed: random seed

  Returns:
    image: image which is the same shape as input image.

    If boxes, masks, keypoints, and keypoint_flip_permutation are not None,
    the function also returns the following tensors.

    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
    masks: rank 3 float32 tensor with shape [num_instances, height, width]
           containing instance masks.
  """

  def _flip_image(image):
    # flip image
    image_flipped = tf.image.flip_left_right(image)
    return image_flipped

  result = []
  # random variable defining whether to do flip or not
  do_a_flip_random = tf.greater(tf.random.uniform([], seed=seed), 0.5)

  # flip image
  image = tf.cond(
      pred=do_a_flip_random,
      true_fn=lambda: _flip_image(image),
      false_fn=lambda: image)
  result.append(image)

  # flip boxes
  if boxes is not None:
    boxes = tf.cond(
        pred=do_a_flip_random,
        true_fn=lambda: _flip_boxes_left_right(boxes),
        false_fn=lambda: boxes)
    result.append(boxes)

  # flip masks
  if masks is not None:
    masks = tf.cond(
        pred=do_a_flip_random,
        true_fn=lambda: _flip_masks_left_right(masks),
        false_fn=lambda: masks)
    result.append(masks)

  return tuple(result)

def _get_or_create_preprocess_rand_vars(generator_func,
                                        function_id,
                                        preprocess_vars_cache,
                                        key=''):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/preprocessor.py#L172
  """Returns a tensor stored in preprocess_vars_cache or using generator_func.
  If the tensor was previously generated and appears in the PreprocessorCache,
  the previously generated tensor will be returned. Otherwise, a new tensor
  is generated using generator_func and stored in the cache.
  Args:
    generator_func: A 0-argument function that generates a tensor.
    function_id: identifier for the preprocessing function used.
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.
    key: identifier for the variable stored.
  Returns:
    The generated tensor.
  """
  if preprocess_vars_cache is not None:
    var = preprocess_vars_cache.get(function_id, key)
    if var is None:
      var = generator_func()
      preprocess_vars_cache.update(function_id, key, var)
  else:
    var = generator_func()
  return var

def _random_integer(minval, maxval, seed):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/preprocessor.py#L203
  """Returns a random 0-D tensor between minval and maxval.

  Args:
    minval: minimum value of the random tensor.
    maxval: maximum value of the random tensor.
    seed: random seed.

  Returns:
    A random 0-D tensor between minval and maxval.
  """
  return tf.random.uniform(
      [], minval=minval, maxval=maxval, dtype=tf.int32, seed=seed)

def _get_crop_border(border, size):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/preprocessor.py#L3982
  border = tf.cast(border, tf.float32)
  size = tf.cast(size, tf.float32)

  i = tf.math.ceil(tf.math.log(2.0 * border / size) / tf.math.log(2.0))
  divisor = tf.pow(2.0, i)
  divisor = tf.clip_by_value(divisor, 1, border)
  divisor = tf.cast(divisor, tf.int32)

  return tf.cast(border, tf.int32) // divisor

def _copy_extra_fields(boxlist_to_copy_to, boxlist_to_copy_from):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/box_list_ops.py#L829
  """Copies the extra fields of boxlist_to_copy_from to boxlist_to_copy_to.

  Args:
    boxlist_to_copy_to: BoxList to which extra fields are copied.
    boxlist_to_copy_from: BoxList from which fields are copied.

  Returns:
    boxlist_to_copy_to with extra fields.
  """
  for field in boxlist_to_copy_from.get_extra_fields():
    boxlist_to_copy_to.add_field(field, boxlist_to_copy_from.get_field(field))
  return boxlist_to_copy_to

def scale(boxlist, y_scale, x_scale, scope=None):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/box_list_ops.py#L82
  """scale box coordinates in x and y dimensions.

  Args:
    boxlist: BoxList holding N boxes
    y_scale: (float) scalar tensor
    x_scale: (float) scalar tensor
    scope: name scope.

  Returns:
    boxlist: BoxList holding N boxes
  """
  y_scale = tf.cast(y_scale, tf.float32)
  x_scale = tf.cast(x_scale, tf.float32)
  y_min, x_min, y_max, x_max = tf.split(
      value=boxlist.get(), num_or_size_splits=4, axis=1)
  y_min = y_scale * y_min
  y_max = y_scale * y_max
  x_min = x_scale * x_min
  x_max = x_scale * x_max
  scaled_boxlist = box_list.BoxList(
      tf.concat([y_min, x_min, y_max, x_max], 1))
  return _copy_extra_fields(scaled_boxlist, boxlist)

def change_coordinate_frame(boxlist, window, scope=None):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/box_list_ops.py#L442
  """Change coordinate frame of the boxlist to be relative to window's frame.

  Given a window of the form [ymin, xmin, ymax, xmax],
  changes bounding box coordinates from boxlist to be relative to this window
  (e.g., the min corner maps to (0,0) and the max corner maps to (1,1)).

  An example use case is data augmentation: where we are given groundtruth
  boxes (boxlist) and would like to randomly crop the image to some
  window (window). In this case we need to change the coordinate frame of
  each groundtruth box to be relative to this new window.

  Args:
    boxlist: A BoxList object holding N boxes.
    window: A rank 1 tensor [4].
    scope: name scope.

  Returns:
    Returns a BoxList object with N boxes.
  """
  win_height = window[2] - window[0]
  win_width = window[3] - window[1]
  boxlist_new = scale(box_list.BoxList(
      boxlist.get() - [window[0], window[1], window[0], window[1]]),
                      1.0 / win_height, 1.0 / win_width)
  boxlist_new = _copy_extra_fields(boxlist_new, boxlist)
  return boxlist_new

def combined_static_and_dynamic_shape(tensor):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/utils/shape_utils.py#L163
  """Returns a list containing static and dynamic values for the dimensions.
  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.
  Args:
    tensor: A tensor of any type.
  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape


def matmul_gather_on_zeroth_axis(params, indices, scope=None):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/utils/ops.py#L990
  """Matrix multiplication based implementation of tf.gather on zeroth axis.
  TODO(rathodv, jonathanhuang): enable sparse matmul option.
  Args:
    params: A float32 Tensor. The tensor from which to gather values.
      Must be at least rank 1.
    indices: A Tensor. Must be one of the following types: int32, int64.
      Must be in range [0, params.shape[0])
    scope: A name for the operation (optional).
  Returns:
    A Tensor. Has the same type as params. Values from params gathered
    from indices given by indices, with shape indices.shape + params.shape[1:].
  """
  params_shape = combined_static_and_dynamic_shape(params)
  indices_shape = combined_static_and_dynamic_shape(indices)
  params2d = tf.reshape(params, [params_shape[0], -1])
  indicator_matrix = tf.one_hot(indices, params_shape[0])
  gathered_result_flattened = tf.matmul(indicator_matrix, params2d)
  return tf.reshape(gathered_result_flattened,
                    tf.stack(indices_shape + params_shape[1:]))

def gather(boxlist, indices, fields=None, scope=None, use_static_shapes=False):
  """Gather boxes from BoxList according to indices and return new BoxList.

  By default, `gather` returns boxes corresponding to the input index list, as
  well as all additional fields stored in the boxlist (indexing into the
  first dimension).  However one can optionally only gather from a
  subset of fields.

  Args:
    boxlist: BoxList holding N boxes
    indices: a rank-1 tensor of type int32 / int64
    fields: (optional) list of fields to also gather from.  If None (default),
      all fields are gathered from.  Pass an empty fields list to only gather
      the box coordinates.
    scope: name scope.
    use_static_shapes: Whether to use an implementation with static shape
      gurantees.

  Returns:
    subboxlist: a BoxList corresponding to the subset of the input BoxList
    specified by indices
  Raises:
    ValueError: if specified field is not contained in boxlist or if the
      indices are not of type int32
  """
  if len(indices.shape.as_list()) != 1:
    raise ValueError('indices should have rank 1')
  if indices.dtype != tf.int32 and indices.dtype != tf.int64:
    raise ValueError('indices should be an int32 / int64 tensor')
  gather_op = tf.gather
  if use_static_shapes:
    gather_op = matmul_gather_on_zeroth_axis
  subboxlist = box_list.BoxList(gather_op(boxlist.get(), indices))
  if fields is None:
    fields = boxlist.get_extra_fields()
  fields += ['boxes']
  for field in fields:
    if not boxlist.has_field(field):
      raise ValueError('boxlist must contain all specified fields')
    subfieldlist = gather_op(boxlist.get_field(field), indices)
    subboxlist.add_field(field, subfieldlist)
  return subboxlist

def prune_completely_outside_window(boxlist, window, scope=None):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/box_list_ops.py#L206
  """Prunes bounding boxes that fall completely outside of the given window.

  The function clip_to_window prunes bounding boxes that fall
  completely outside the window, but also clips any bounding boxes that
  partially overflow. This function does not clip partially overflowing boxes.

  Args:
    boxlist: a BoxList holding M_in boxes.
    window: a float tensor of shape [4] representing [ymin, xmin, ymax, xmax]
      of the window
    scope: name scope.

  Returns:
    pruned_boxlist: a new BoxList with all bounding boxes partially or fully in
      the window.
    valid_indices: a tensor with shape [M_out] indexing the valid bounding boxes
     in the input tensor.
  """
  y_min, x_min, y_max, x_max = tf.split(
      value=boxlist.get(), num_or_size_splits=4, axis=1)
  win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
  coordinate_violations = tf.concat([
      tf.greater_equal(y_min, win_y_max), tf.greater_equal(x_min, win_x_max),
      tf.less_equal(y_max, win_y_min), tf.less_equal(x_max, win_x_min)
  ], 1)
  valid_indices = tf.reshape(
      tf.where(tf.math.logical_not(tf.reduce_any(coordinate_violations, 1))), [-1])
  return gather(boxlist, valid_indices), valid_indices

def intersection(boxlist1, boxlist2, scope=None):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/box_list_ops.py#L238
  """Compute pairwise intersection areas between boxes.
  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.
  Returns:
    a tensor with shape [N, M] representing pairwise intersections
  """
  y_min1, x_min1, y_max1, x_max1 = tf.split(
      value=boxlist1.get(), num_or_size_splits=4, axis=1)
  y_min2, x_min2, y_max2, x_max2 = tf.split(
      value=boxlist2.get(), num_or_size_splits=4, axis=1)
  all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
  all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
  intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
  all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
  all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
  intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
  return intersect_heights * intersect_widths

def ioa(boxlist1, boxlist2, scope=None):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/box_list_ops.py#L375
  """Computes pairwise intersection-over-area between box collections.
  intersection-over-area (IOA) between two boxes box1 and box2 is defined as
  their intersection area over box2's area. Note that ioa is not symmetric,
  that is, ioa(box1, box2) != ioa(box2, box1).
  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.
  Returns:
    a tensor with shape [N, M] representing pairwise ioa scores.
  """
  intersections = intersection(boxlist1, boxlist2)
  areas = tf.expand_dims(area(boxlist2), 0)
  return tf.truediv(intersections, areas)

def prune_non_overlapping_boxes(
    boxlist1, boxlist2, min_overlap=0.0, scope=None):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/box_list_ops.py#L396
  """Prunes the boxes in boxlist1 that overlap less than thresh with boxlist2.
  For each box in boxlist1, we want its IOA to be more than minoverlap with
  at least one of the boxes in boxlist2. If it does not, we remove it.
  Args:
    boxlist1: BoxList holding N boxes.
    boxlist2: BoxList holding M boxes.
    min_overlap: Minimum required overlap between boxes, to count them as
                overlapping.
    scope: name scope.
  Returns:
    new_boxlist1: A pruned boxlist with size [N', 4].
    keep_inds: A tensor with shape [N'] indexing kept bounding boxes in the
      first input BoxList `boxlist1`.
  """
  ioa_ = ioa(boxlist2, boxlist1)  # [M, N] tensor
  ioa_ = tf.reduce_max(ioa_, axis=[0])  # [N] tensor
  keep_bool = tf.greater_equal(ioa_, tf.constant(min_overlap))
  keep_inds = tf.squeeze(tf.where(keep_bool), axis=[1])
  new_boxlist1 = gather(boxlist1, keep_inds)
  return new_boxlist1, keep_inds

def area(boxlist, scope=None):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/box_list_ops.py#L49
  """Computes area of boxes.

  Args:
    boxlist: BoxList holding N boxes
    scope: name scope.

  Returns:
    a tensor with shape [N] representing box areas.
  """
  y_min, x_min, y_max, x_max = tf.split(
      value=boxlist.get(), num_or_size_splits=4, axis=1)
  return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

def clip_to_window(boxlist, window, filter_nonoverlapping=True, scope=None):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/box_list_ops.py#L133
  """Clip bounding boxes to a window.

  This op clips any input bounding boxes (represented by bounding box
  corners) to a window, optionally filtering out boxes that do not
  overlap at all with the window.

  Args:
    boxlist: BoxList holding M_in boxes
    window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]
      window to which the op should clip boxes.
    filter_nonoverlapping: whether to filter out boxes that do not overlap at
      all with the window.
    scope: name scope.

  Returns:
    a BoxList holding M_out boxes where M_out <= M_in
  """
  y_min, x_min, y_max, x_max = tf.split(
      value=boxlist.get(), num_or_size_splits=4, axis=1)
  win_y_min = window[0]
  win_x_min = window[1]
  win_y_max = window[2]
  win_x_max = window[3]
  y_min_clipped = tf.maximum(tf.minimum(y_min, win_y_max), win_y_min)
  y_max_clipped = tf.maximum(tf.minimum(y_max, win_y_max), win_y_min)
  x_min_clipped = tf.maximum(tf.minimum(x_min, win_x_max), win_x_min)
  x_max_clipped = tf.maximum(tf.minimum(x_max, win_x_max), win_x_min)
  clipped = box_list.BoxList(
      tf.concat([y_min_clipped, x_min_clipped, y_max_clipped, x_max_clipped],
                1))
  clipped = _copy_extra_fields(clipped, boxlist)
  if filter_nonoverlapping:
    areas = area(clipped)
    nonzero_area_indices = tf.cast(
        tf.reshape(tf.where(tf.greater(areas, 0.0)), [-1]), tf.int32)
    clipped = gather(clipped, nonzero_area_indices)
  return clipped

def random_square_crop_by_scale(image, boxes, labels, label_weights,
                                label_confidences=None, masks=None,
                                keypoints=None, max_border=128, scale_min=0.6,
                                scale_max=1.3, num_scales=8, seed=None,
                                preprocess_vars_cache=None):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/preprocessor.py#L3994
  """Randomly crop a square in proportion to scale and image size.
   Extract a square sized crop from an image whose side length is sampled by
   randomly scaling the maximum spatial dimension of the image. If part of
   the crop falls outside the image, it is filled with zeros.
   The augmentation is borrowed from [1]
   [1]: https://arxiv.org/abs/1904.07850
  Args:
    image: rank 3 float32 tensor containing 1 image ->
           [height, width, channels].
    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1]. Each row is in the form of [ymin, xmin, ymax, xmax].
           Boxes on the crop boundary are clipped to the boundary and boxes
           falling outside the crop are ignored.
    labels: rank 1 int32 tensor containing the object classes.
    label_weights: float32 tensor of shape [num_instances] representing the
      weight for each box.
    label_confidences: (optional) float32 tensor of shape [num_instances]
      representing the confidence for each box.
    masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width, 1] containing instance masks. The masks
           are of the same height, width as the input `image`.
    keypoints: (optional) rank 3 float32 tensor with shape
      [num_instances, num_keypoints, 2]. The keypoints are in y-x normalized
      coordinates.
    max_border: The maximum size of the border. The border defines distance in
      pixels to the image boundaries that will not be considered as a center of
      a crop. To make sure that the border does not go over the center of the
      image, we chose the border value by computing the minimum k, such that
      (max_border / (2**k)) < image_dimension/2.
    scale_min: float, the minimum value for scale.
    scale_max: float, the maximum value for scale.
    num_scales: int, the number of discrete scale values to sample between
      [scale_min, scale_max]
    seed: random seed.
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.
  Returns:
    image: image which is the same rank as input image.
    boxes: boxes which is the same rank as input boxes.
           Boxes are in normalized form.
    labels: new labels.
    label_weights: rank 1 float32 tensor with shape [num_instances].
    label_confidences: (optional) float32 tensor of shape [num_instances]
      representing the confidence for each box.
    masks: rank 3 float32 tensor with shape [num_instances, height, width]
           containing instance masks.
  """

  img_shape = tf.shape(image)
  height, width = img_shape[0], img_shape[1]
  scales = tf.linspace(scale_min, scale_max, num_scales)

  scale = _get_or_create_preprocess_rand_vars(
      lambda: scales[_random_integer(0, num_scales, seed)],
      'square_crop_scale',
      preprocess_vars_cache, 'scale')

  image_size = scale * tf.cast(tf.maximum(height, width), tf.float32)
  image_size = tf.cast(image_size, tf.int32)
  h_border = _get_crop_border(max_border, height)
  w_border = _get_crop_border(max_border, width)

  def y_function():
    y = _random_integer(h_border,
                        tf.cast(height, tf.int32) - h_border + 1,
                        seed)
    return y

  def x_function():
    x = _random_integer(w_border,
                        tf.cast(width, tf.int32) - w_border + 1,
                        seed)
    return x

  y_center = _get_or_create_preprocess_rand_vars(
      y_function,
      'square_crop_scale',
      preprocess_vars_cache, 'y_center')

  x_center = _get_or_create_preprocess_rand_vars(
      x_function,
      'square_crop_scale',
      preprocess_vars_cache, 'x_center')

  half_size = tf.cast(image_size / 2, tf.int32)
  crop_ymin, crop_ymax = y_center - half_size, y_center + half_size
  crop_xmin, crop_xmax = x_center - half_size, x_center + half_size

  ymin = tf.maximum(crop_ymin, 0)
  xmin = tf.maximum(crop_xmin, 0)
  ymax = tf.minimum(crop_ymax, height - 1)
  xmax = tf.minimum(crop_xmax, width - 1)

  cropped_image = image[ymin:ymax, xmin:xmax]
  offset_y = tf.maximum(0, ymin - crop_ymin)
  offset_x = tf.maximum(0, xmin - crop_xmin)

  oy_i = offset_y
  ox_i = offset_x

  output_image = tf.image.pad_to_bounding_box(
      cropped_image, offset_height=oy_i, offset_width=ox_i,
      target_height=image_size, target_width=image_size)

  if ymin == 0:
    # We might be padding the image.
    box_ymin = -offset_y
  else:
    box_ymin = crop_ymin

  if xmin == 0:
    # We might be padding the image.
    box_xmin = -offset_x
  else:
    box_xmin = crop_xmin

  box_ymax = box_ymin + image_size
  box_xmax = box_xmin + image_size

  image_box = [box_ymin / height, box_xmin / width,
               box_ymax / height, box_xmax / width]
  boxlist = box_list.BoxList(boxes)
  boxlist = change_coordinate_frame(boxlist, image_box)
  boxlist, indices = prune_completely_outside_window(
      boxlist, [0.0, 0.0, 1.0, 1.0])
  boxlist = clip_to_window(boxlist, [0.0, 0.0, 1.0, 1.0],
                                        filter_nonoverlapping=False)

  return_values = [output_image, boxlist.get(),
                   tf.gather(labels, indices),
                   tf.gather(label_weights, indices)]

  if label_confidences is not None:
    return_values.append(tf.gather(label_confidences, indices))

  if masks is not None:
    new_masks = tf.expand_dims(masks, -1)
    new_masks = new_masks[:, ymin:ymax, xmin:xmax]
    new_masks = tf.image.pad_to_bounding_box(
        new_masks, oy_i, ox_i, image_size, image_size)
    new_masks = tf.squeeze(new_masks, [-1])
    return_values.append(tf.gather(new_masks, indices))

  return return_values

def _strict_random_crop_image(image,
                              boxes,
                              labels,
                              label_weights,
                              label_confidences=None,
                              multiclass_scores=None,
                              masks=None,
                              keypoints=None,
                              keypoint_visibilities=None,
                              densepose_num_points=None,
                              densepose_part_ids=None,
                              densepose_surface_coords=None,
                              min_object_covered=1.0,
                              aspect_ratio_range=(0.75, 1.33),
                              area_range=(0.1, 1.0),
                              overlap_thresh=0.3,
                              clip_boxes=True,
                              preprocess_vars_cache=None):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/preprocessor.py#L1334
  """Performs random crop.
  Note: Keypoint coordinates that are outside the crop will be set to NaN, which
  is consistent with the original keypoint encoding for non-existing keypoints.
  This function always crops the image and is supposed to be used by
  `random_crop_image` function which sometimes returns the image unchanged.
  Args:
    image: rank 3 float32 tensor containing 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    boxes: rank 2 float32 tensor containing the bounding boxes with shape
           [num_instances, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    labels: rank 1 int32 tensor containing the object classes.
    label_weights: float32 tensor of shape [num_instances] representing the
      weight for each box.
    label_confidences: (optional) float32 tensor of shape [num_instances]
      representing the confidence for each box.
    multiclass_scores: (optional) float32 tensor of shape
      [num_instances, num_classes] representing the score for each box for each
      class.
    masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width] containing instance masks. The masks
           are of the same height, width as the input `image`.
    keypoints: (optional) rank 3 float32 tensor with shape
               [num_instances, num_keypoints, 2]. The keypoints are in y-x
               normalized coordinates.
    keypoint_visibilities: (optional) rank 2 bool tensor with shape
               [num_instances, num_keypoints].
    densepose_num_points: (optional) rank 1 int32 tensor with shape
                          [num_instances] with the number of sampled points per
                          instance.
    densepose_part_ids: (optional) rank 2 int32 tensor with shape
                        [num_instances, num_points] holding the part id for each
                        sampled point. These part_ids are 0-indexed, where the
                        first non-background part has index 0.
    densepose_surface_coords: (optional) rank 3 float32 tensor with shape
                              [num_instances, num_points, 4]. The DensePose
                              coordinates are of the form (y, x, v, u) where
                              (y, x) are the normalized image coordinates for a
                              sampled point, and (v, u) is the surface
                              coordinate for the part.
    min_object_covered: the cropped image must cover at least this fraction of
                        at least one of the input bounding boxes.
    aspect_ratio_range: allowed range for aspect ratio of cropped image.
    area_range: allowed range for area ratio between cropped image and the
                original image.
    overlap_thresh: minimum overlap thresh with new cropped
                    image to keep the box.
    clip_boxes: whether to clip the boxes to the cropped image.
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.
  Returns:
    image: image which is the same rank as input image.
    boxes: boxes which is the same rank as input boxes.
           Boxes are in normalized form.
    labels: new labels.
    If label_weights, multiclass_scores, masks, keypoints,
    keypoint_visibilities, densepose_num_points, densepose_part_ids, or
    densepose_surface_coords is not None, the function also returns:
    label_weights: rank 1 float32 tensor with shape [num_instances].
    multiclass_scores: rank 2 float32 tensor with shape
                       [num_instances, num_classes]
    masks: rank 3 float32 tensor with shape [num_instances, height, width]
           containing instance masks.
    keypoints: rank 3 float32 tensor with shape
               [num_instances, num_keypoints, 2]
    keypoint_visibilities: rank 2 bool tensor with shape
                           [num_instances, num_keypoints]
    densepose_num_points: rank 1 int32 tensor with shape [num_instances].
    densepose_part_ids: rank 2 int32 tensor with shape
                        [num_instances, num_points].
    densepose_surface_coords: rank 3 float32 tensor with shape
                              [num_instances, num_points, 4].
  Raises:
    ValueError: If some but not all of the DensePose tensors are provided.
  """
  densepose_tensors = [densepose_num_points, densepose_part_ids,
                       densepose_surface_coords]
  if (any(t is not None for t in densepose_tensors) and
      not all(t is not None for t in densepose_tensors)):
    raise ValueError('If cropping DensePose labels, must provide '
                     '`densepose_num_points`, `densepose_part_ids`, and '
                     '`densepose_surface_coords`')
  image_shape = tf.shape(image)

  # boxes are [N, 4]. Lets first make them [N, 1, 4].
  boxes_expanded = tf.expand_dims(
      tf.clip_by_value(
          boxes, clip_value_min=0.0, clip_value_max=1.0), 1)

  generator_func = functools.partial(
      tf.image.sample_distorted_bounding_box,
      image_shape,
      bounding_boxes=boxes_expanded,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=100,
      use_image_if_no_bounding_boxes=True)

  # for ssd cropping, each value of min_object_covered has its own
  # cached random variable
  sample_distorted_bounding_box = _get_or_create_preprocess_rand_vars(
      generator_func,
      'strict_crop_image',
      preprocess_vars_cache, key=min_object_covered)

  im_box_begin, im_box_size, im_box = sample_distorted_bounding_box
  im_box_end = im_box_begin + im_box_size
  new_image = image[im_box_begin[0]:im_box_end[0],
                    im_box_begin[1]:im_box_end[1], :]
  new_image.set_shape([None, None, image.get_shape()[2]])

  # [1, 4]
  im_box_rank2 = tf.squeeze(im_box, axis=[0])
  # [4]
  im_box_rank1 = tf.squeeze(im_box)

  boxlist = box_list.BoxList(boxes)
  boxlist.add_field('labels', labels)

  if label_weights is not None:
    boxlist.add_field('label_weights', label_weights)

  if label_confidences is not None:
    boxlist.add_field('label_confidences', label_confidences)

  if multiclass_scores is not None:
    boxlist.add_field('multiclass_scores', multiclass_scores)

  im_boxlist = box_list.BoxList(im_box_rank2)

  # remove boxes that are outside cropped image
  boxlist, inside_window_ids = prune_completely_outside_window(
      boxlist, im_box_rank1)

  # remove boxes that are outside image
  overlapping_boxlist, keep_ids = prune_non_overlapping_boxes(
      boxlist, im_boxlist, overlap_thresh)

  # change the coordinate of the remaining boxes
  new_labels = overlapping_boxlist.get_field('labels')
  new_boxlist = change_coordinate_frame(overlapping_boxlist,
                                                     im_box_rank1)
  new_boxes = new_boxlist.get()
  if clip_boxes:
    new_boxes = tf.clip_by_value(
        new_boxes, clip_value_min=0.0, clip_value_max=1.0)

  result = [new_image, new_boxes, new_labels]

  if label_weights is not None:
    new_label_weights = overlapping_boxlist.get_field('label_weights')
    result.append(new_label_weights)

  if label_confidences is not None:
    new_label_confidences = overlapping_boxlist.get_field('label_confidences')
    result.append(new_label_confidences)

  if multiclass_scores is not None:
    new_multiclass_scores = overlapping_boxlist.get_field('multiclass_scores')
    result.append(new_multiclass_scores)

  if masks is not None:
    masks_of_boxes_inside_window = tf.gather(masks, inside_window_ids)
    masks_of_boxes_completely_inside_window = tf.gather(
        masks_of_boxes_inside_window, keep_ids)
    new_masks = masks_of_boxes_completely_inside_window[:, im_box_begin[
        0]:im_box_end[0], im_box_begin[1]:im_box_end[1]]
    result.append(new_masks)

  return tuple(result)

def random_pad_image(image,
                     boxes,
                     masks=None,
                     keypoints=None,
                     densepose_surface_coords=None,
                     min_image_size=None,
                     max_image_size=None,
                     pad_color=None,
                     center_pad=False,
                     seed=None,
                     preprocess_vars_cache=None):
  """Randomly pads the image.
  This function randomly pads the image with zeros. The final size of the
  padded image will be between min_image_size and max_image_size.
  if min_image_size is smaller than the input image size, min_image_size will
  be set to the input image size. The same for max_image_size. The input image
  will be located at a uniformly random location inside the padded image.
  The relative location of the boxes to the original image will remain the same.
  Args:
    image: rank 3 float32 tensor containing 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    masks: (optional) rank 3 float32 tensor with shape
           [N, height, width] containing instance masks. The masks
           are of the same height, width as the input `image`.
    keypoints: (optional) rank 3 float32 tensor with shape
               [N, num_keypoints, 2]. The keypoints are in y-x normalized
               coordinates.
    densepose_surface_coords: (optional) rank 3 float32 tensor with shape
                              [N, num_points, 4]. The DensePose coordinates are
                              of the form (y, x, v, u) where (y, x) are the
                              normalized image coordinates for a sampled point,
                              and (v, u) is the surface coordinate for the part.
    min_image_size: a tensor of size [min_height, min_width], type tf.int32.
                    If passed as None, will be set to image size
                    [height, width].
    max_image_size: a tensor of size [max_height, max_width], type tf.int32.
                    If passed as None, will be set to twice the
                    image [height * 2, width * 2].
    pad_color: padding color. A rank 1 tensor of [channels] with dtype=
               tf.float32. if set as None, it will be set to average color of
               the input image.
    center_pad: whether the original image will be padded to the center, or
                randomly padded (which is default).
    seed: random seed.
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.
  Returns:
    image: Image shape will be [new_height, new_width, channels].
    boxes: boxes which is the same rank as input boxes. Boxes are in normalized
           form.
    if masks is not None, the function also returns:
    masks: rank 3 float32 tensor with shape [N, new_height, new_width]
    if keypoints is not None, the function also returns:
    keypoints: rank 3 float32 tensor with shape [N, num_keypoints, 2]
    if densepose_surface_coords is not None, the function also returns:
    densepose_surface_coords: rank 3 float32 tensor with shape
      [num_instances, num_points, 4]
  """
  if pad_color is None:
    pad_color = tf.reduce_mean(image, axis=[0, 1])

  image_shape = tf.shape(image)
  image_height = image_shape[0]
  image_width = image_shape[1]

  if max_image_size is None:
    max_image_size = tf.stack([image_height * 2, image_width * 2])
  max_image_size = tf.maximum(max_image_size,
                              tf.stack([image_height, image_width]))

  if min_image_size is None:
    min_image_size = tf.stack([image_height, image_width])
  min_image_size = tf.maximum(min_image_size,
                              tf.stack([image_height, image_width]))
  target_height = tf.cond(
      max_image_size[0] > min_image_size[0],
      lambda: _random_integer(min_image_size[0], max_image_size[0], seed),
      lambda: max_image_size[0])

  target_width = tf.cond(
      max_image_size[1] > min_image_size[1],
      lambda: _random_integer(min_image_size[1], max_image_size[1], seed),
      lambda: max_image_size[1])

  offset_height = tf.cond(
      target_height > image_height,
      lambda: _random_integer(0, target_height - image_height, seed),
      lambda: tf.constant(0, dtype=tf.int32))

  offset_width = tf.cond(
      target_width > image_width,
      lambda: _random_integer(0, target_width - image_width, seed),
      lambda: tf.constant(0, dtype=tf.int32))

  if center_pad:
    offset_height = tf.cast(tf.floor((target_height - image_height) / 2),
                            tf.int32)
    offset_width = tf.cast(tf.floor((target_width - image_width) / 2),
                           tf.int32)

  gen_func = lambda: (target_height, target_width, offset_height, offset_width)
  params = _get_or_create_preprocess_rand_vars(
      gen_func, 'pad_image',
      preprocess_vars_cache)
  target_height, target_width, offset_height, offset_width = params

  new_image = tf.image.pad_to_bounding_box(
      image,
      offset_height=offset_height,
      offset_width=offset_width,
      target_height=target_height,
      target_width=target_width)

  # Setting color of the padded pixels
  image_ones = tf.ones_like(image)
  image_ones_padded = tf.image.pad_to_bounding_box(
      image_ones,
      offset_height=offset_height,
      offset_width=offset_width,
      target_height=target_height,
      target_width=target_width)
  image_color_padded = (1- image_ones_padded) * pad_color
  new_image += image_color_padded

  # setting boxes
  new_window = tf.cast(
      tf.stack([
          -offset_height, -offset_width, target_height - offset_height,
          target_width - offset_width
      ]),
      dtype=tf.float32)
  new_window /= tf.cast(
      tf.stack([image_height, image_width, image_height, image_width]),
      dtype=tf.float32)
  boxlist = box_list.BoxList(boxes)
  new_boxlist = change_coordinate_frame(boxlist, new_window)
  new_boxes = new_boxlist.get()

  result = [new_image, new_boxes]

  if masks is not None:
    new_masks = tf.image.pad_to_bounding_box(
        masks[:, :, :, tf.newaxis],
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=target_height,
        target_width=target_width)[:, :, :, 0]
    result.append(new_masks)

  return tuple(result)


def random_absolute_pad_image(image,
                              boxes,
                              masks=None,
                              keypoints=None,
                              densepose_surface_coords=None,
                              max_height_padding=None,
                              max_width_padding=None,
                              pad_color=None,
                              seed=None,
                              preprocess_vars_cache=None):
  """Randomly pads the image by small absolute amounts.
  As random_pad_image above, but the padding is of size [0, max_height_padding]
  or [0, max_width_padding] instead of padding to a fixed size of
  max_height_padding for all images.
  Args:
    image: rank 3 float32 tensor containing 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    masks: (optional) rank 3 float32 tensor with shape
           [N, height, width] containing instance masks. The masks
           are of the same height, width as the input `image`.
    keypoints: (optional) rank 3 float32 tensor with shape
               [N, num_keypoints, 2]. The keypoints are in y-x normalized
               coordinates.
    densepose_surface_coords: (optional) rank 3 float32 tensor with shape
                              [N, num_points, 4]. The DensePose coordinates are
                              of the form (y, x, v, u) where (y, x) are the
                              normalized image coordinates for a sampled point,
                              and (v, u) is the surface coordinate for the part.
    max_height_padding: a scalar tf.int32 tensor denoting the maximum amount of
                        height padding. The padding will be chosen uniformly at
                        random from [0, max_height_padding).
    max_width_padding: a scalar tf.int32 tensor denoting the maximum amount of
                       width padding. The padding will be chosen uniformly at
                       random from [0, max_width_padding).
    pad_color: padding color. A rank 1 tensor of [3] with dtype=tf.float32.
               if set as None, it will be set to average color of the input
               image.
    seed: random seed.
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.
  Returns:
    image: Image shape will be [new_height, new_width, channels].
    boxes: boxes which is the same rank as input boxes. Boxes are in normalized
           form.
    if masks is not None, the function also returns:
    masks: rank 3 float32 tensor with shape [N, new_height, new_width]
    if keypoints is not None, the function also returns:
    keypoints: rank 3 float32 tensor with shape [N, num_keypoints, 2]
  """
  min_image_size = tf.shape(image)[:2]
  max_image_size = [min_image_size[0] + tf.cast(max_height_padding, dtype=tf.int32), 
                    min_image_size[1] + tf.cast(max_width_padding, dtype=tf.int32)]
  max_image_size = tf.squeeze(max_image_size)
  return random_pad_image(
      image,
      boxes,
      masks=masks,
      keypoints=keypoints,
      densepose_surface_coords=densepose_surface_coords,
      min_image_size=min_image_size,
      max_image_size=max_image_size,
      pad_color=pad_color,
      seed=seed,
      preprocess_vars_cache=preprocess_vars_cache)

def random_crop_image(image,
                      boxes,
                      labels,
                      label_weights,
                      label_confidences=None,
                      multiclass_scores=None,
                      masks=None,
                      keypoints=None,
                      keypoint_visibilities=None,
                      densepose_num_points=None,
                      densepose_part_ids=None,
                      densepose_surface_coords=None,
                      min_object_covered=1.0,
                      aspect_ratio_range=(0.75, 1.33),
                      area_range=(0.1, 1.0),
                      overlap_thresh=0.3,
                      clip_boxes=True,
                      random_coef=0.0,
                      seed=None,
                      preprocess_vars_cache=None):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/preprocessor.py#L1574
  """Randomly crops the image.
  Given the input image and its bounding boxes, this op randomly
  crops a subimage.  Given a user-provided set of input constraints,
  the crop window is resampled until it satisfies these constraints.
  If within 100 trials it is unable to find a valid crop, the original
  image is returned. See the Args section for a description of the input
  constraints. Both input boxes and returned Boxes are in normalized
  form (e.g., lie in the unit square [0, 1]).
  This function will return the original image with probability random_coef.
  Note: Keypoint coordinates that are outside the crop will be set to NaN, which
  is consistent with the original keypoint encoding for non-existing keypoints.
  Also, the keypoint visibility will be set to False.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    boxes: rank 2 float32 tensor containing the bounding boxes with shape
           [num_instances, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    labels: rank 1 int32 tensor containing the object classes.
    label_weights: float32 tensor of shape [num_instances] representing the
      weight for each box.
    label_confidences: (optional) float32 tensor of shape [num_instances].
      representing the confidence for each box.
    multiclass_scores: (optional) float32 tensor of shape
      [num_instances, num_classes] representing the score for each box for each
      class.
    masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width] containing instance masks. The masks
           are of the same height, width as the input `image`.
    keypoints: (optional) rank 3 float32 tensor with shape
               [num_instances, num_keypoints, 2]. The keypoints are in y-x
               normalized coordinates.
    keypoint_visibilities: (optional) rank 2 bool tensor with shape
                           [num_instances, num_keypoints].
    densepose_num_points: (optional) rank 1 int32 tensor with shape
                          [num_instances] with the number of sampled points per
                          instance.
    densepose_part_ids: (optional) rank 2 int32 tensor with shape
                        [num_instances, num_points] holding the part id for each
                        sampled point. These part_ids are 0-indexed, where the
                        first non-background part has index 0.
    densepose_surface_coords: (optional) rank 3 float32 tensor with shape
                              [num_instances, num_points, 4]. The DensePose
                              coordinates are of the form (y, x, v, u) where
                              (y, x) are the normalized image coordinates for a
                              sampled point, and (v, u) is the surface
                              coordinate for the part.
    min_object_covered: the cropped image must cover at least this fraction of
                        at least one of the input bounding boxes.
    aspect_ratio_range: allowed range for aspect ratio of cropped image.
    area_range: allowed range for area ratio between cropped image and the
                original image.
    overlap_thresh: minimum overlap thresh with new cropped
                    image to keep the box.
    clip_boxes: whether to clip the boxes to the cropped image.
    random_coef: a random coefficient that defines the chance of getting the
                 original image. If random_coef is 0, we will always get the
                 cropped image, and if it is 1.0, we will always get the
                 original image.
    seed: random seed.
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.
  Returns:
    image: Image shape will be [new_height, new_width, channels].
    boxes: boxes which is the same rank as input boxes. Boxes are in normalized
           form.
    labels: new labels.
    If label_weights, multiclass_scores, masks, keypoints,
    keypoint_visibilities, densepose_num_points, densepose_part_ids,
    densepose_surface_coords is not None, the function also returns:
    label_weights: rank 1 float32 tensor with shape [num_instances].
    multiclass_scores: rank 2 float32 tensor with shape
                       [num_instances, num_classes]
    masks: rank 3 float32 tensor with shape [num_instances, height, width]
           containing instance masks.
    keypoints: rank 3 float32 tensor with shape
               [num_instances, num_keypoints, 2]
    keypoint_visibilities: rank 2 bool tensor with shape
                           [num_instances, num_keypoints]
    densepose_num_points: rank 1 int32 tensor with shape [num_instances].
    densepose_part_ids: rank 2 int32 tensor with shape
                        [num_instances, num_points].
    densepose_surface_coords: rank 3 float32 tensor with shape
                              [num_instances, num_points, 4].
  """

  def strict_random_crop_image_fn():
    return _strict_random_crop_image(
        image,
        boxes,
        labels,
        label_weights,
        label_confidences=label_confidences,
        multiclass_scores=multiclass_scores,
        masks=masks,
        keypoints=keypoints,
        keypoint_visibilities=keypoint_visibilities,
        densepose_num_points=densepose_num_points,
        densepose_part_ids=densepose_part_ids,
        densepose_surface_coords=densepose_surface_coords,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        overlap_thresh=overlap_thresh,
        clip_boxes=clip_boxes,
        preprocess_vars_cache=preprocess_vars_cache)

  # avoids tf.cond to make faster RCNN training on borg. See b/140057645.
  if random_coef < sys.float_info.min:
    result = strict_random_crop_image_fn()
  else:
    generator_func = functools.partial(tf.random_uniform, [], seed=seed)
    do_a_crop_random = _get_or_create_preprocess_rand_vars(
        generator_func, 'crop_image',
        preprocess_vars_cache)
    do_a_crop_random = tf.greater(do_a_crop_random, random_coef)

    outputs = [image, boxes, labels]

    if label_weights is not None:
      outputs.append(label_weights)
    if label_confidences is not None:
      outputs.append(label_confidences)
    if multiclass_scores is not None:
      outputs.append(multiclass_scores)
    if masks is not None:
      outputs.append(masks)
    if keypoints is not None:
      outputs.append(keypoints)
    if keypoint_visibilities is not None:
      outputs.append(keypoint_visibilities)
    if densepose_num_points is not None:
      outputs.extend([densepose_num_points, densepose_part_ids,
                      densepose_surface_coords])

    result = tf.cond(do_a_crop_random, strict_random_crop_image_fn,
                     lambda: tuple(outputs))
  return result

def random_vertical_flip(image,
                         boxes=None,
                         masks=None,
                         keypoints=None,
                         keypoint_flip_permutation=None,
                         probability=0.5,
                         seed=None,
                         preprocess_vars_cache=None):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/preprocessor.py#L714
  """Randomly flips the image and detections vertically.
  The probability of flipping the image is 50%.
  Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    boxes: (optional) rank 2 float32 tensor with shape [N, 4]
           containing the bounding boxes.
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width] containing instance masks. The masks
           are of the same height, width as the input `image`.
    keypoints: (optional) rank 3 float32 tensor with shape
               [num_instances, num_keypoints, 2]. The keypoints are in y-x
               normalized coordinates.
    keypoint_flip_permutation: rank 1 int32 tensor containing the keypoint flip
                               permutation.
    probability: the probability of performing this augmentation.
    seed: random seed
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.
  Returns:
    image: image which is the same shape as input image.
    If boxes, masks, keypoints, and keypoint_flip_permutation are not None,
    the function also returns the following tensors.
    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
    masks: rank 3 float32 tensor with shape [num_instances, height, width]
           containing instance masks.
    keypoints: rank 3 float32 tensor with shape
               [num_instances, num_keypoints, 2]
  Raises:
    ValueError: if keypoints are provided but keypoint_flip_permutation is not.
  """

  def _flip_image(image):
    # flip image
    image_flipped = tf.image.flip_up_down(image)
    return image_flipped

  def _flip_boxes_up_down(boxes):
    """Up-down flip the boxes.
    Args:
      boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
             Boxes are in normalized form meaning their coordinates vary
             between [0, 1].
             Each row is in the form of [ymin, xmin, ymax, xmax].
    Returns:
      Flipped boxes.
    """
    ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
    flipped_ymin = tf.subtract(1.0, ymax)
    flipped_ymax = tf.subtract(1.0, ymin)
    flipped_boxes = tf.concat([flipped_ymin, xmin, flipped_ymax, xmax], 1)
    return flipped_boxes

  def _flip_masks_up_down(masks):
    """Up-down flip masks.
    Args:
      masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.
    Returns:
      flipped masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.
    """
    return masks[:, ::-1, :]

  result = []
  # random variable defining whether to do flip or not
  do_a_flip_random = tf.greater(tf.random.uniform([], seed=seed), probability)

  # flip image
  image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)
  result.append(image)

  # flip boxes
  if boxes is not None:
    boxes = tf.cond(do_a_flip_random, lambda: _flip_boxes_up_down(boxes),
                    lambda: boxes)
    result.append(boxes)

  # flip masks
  if masks is not None:
    masks = tf.cond(do_a_flip_random, lambda: _flip_masks_up_down(masks),
                    lambda: masks)
    result.append(masks)

    return tuple(result)

def random_rotation90(image,
                      boxes=None,
                      masks=None,
                      keypoints=None,
                      keypoint_rot_permutation=None,
                      probability=0.5,
                      seed=None,
                      preprocess_vars_cache=None):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/preprocessor.py#L812
  """Randomly rotates the image and detections 90 degrees counter-clockwise.
  The probability of rotating the image is 50%. This can be combined with
  random_horizontal_flip and random_vertical_flip to produce an output with a
  uniform distribution of the eight possible 90 degree rotation / reflection
  combinations.
  Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    boxes: (optional) rank 2 float32 tensor with shape [N, 4]
           containing the bounding boxes.
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width] containing instance masks. The masks
           are of the same height, width as the input `image`.
    keypoints: (optional) rank 3 float32 tensor with shape
               [num_instances, num_keypoints, 2]. The keypoints are in y-x
               normalized coordinates.
    keypoint_rot_permutation: rank 1 int32 tensor containing the keypoint flip
                              permutation.
    probability: the probability of performing this augmentation.
    seed: random seed
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.
  Returns:
    image: image which is the same shape as input image.
    If boxes, masks, and keypoints, are not None,
    the function also returns the following tensors.
    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
    masks: rank 3 float32 tensor with shape [num_instances, height, width]
           containing instance masks.
    keypoints: rank 3 float32 tensor with shape
               [num_instances, num_keypoints, 2]
  """

  def _rot90_image(image):
    # flip image
    image_rotated = tf.image.rot90(image)
    return image_rotated

  def _rot90_boxes(boxes):
    """Rotate boxes counter-clockwise by 90 degrees.
    Args:
      boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
             Boxes are in normalized form meaning their coordinates vary
             between [0, 1].
             Each row is in the form of [ymin, xmin, ymax, xmax].
    Returns:
      Rotated boxes.
    """
    ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
    rotated_ymin = tf.subtract(1.0, xmax)
    rotated_ymax = tf.subtract(1.0, xmin)
    rotated_xmin = ymin
    rotated_xmax = ymax
    rotated_boxes = tf.concat(
        [rotated_ymin, rotated_xmin, rotated_ymax, rotated_xmax], 1)
    return rotated_boxes

  def _rot90_masks(masks):
    """Rotate masks counter-clockwise by 90 degrees.
    Args:
      masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.
    Returns:
      rotated masks: rank 3 float32 tensor with shape
        [num_instances, height, width] representing instance masks.
    """
    masks = tf.transpose(masks, [0, 2, 1])
    return masks[:, ::-1, :]

  result = []

  # random variable defining whether to do flip or not
  do_a_rot90_random = tf.greater(tf.random.uniform([], seed=seed), probability)

  # flip image
  image = tf.cond(do_a_rot90_random, lambda: _rot90_image(image),
                  lambda: image)
  result.append(image)

  # flip boxes
  if boxes is not None:
    boxes = tf.cond(do_a_rot90_random, lambda: _rot90_boxes(boxes),
                    lambda: boxes)
    result.append(boxes)

  # flip masks
  if masks is not None:
    masks = tf.cond(do_a_rot90_random, lambda: _rot90_masks(masks),
                    lambda: masks)
    result.append(masks)

  return tuple(result)

def _augment_only_rgb_channels(image, augment_function):
  """Augments only the RGB slice of an image with additional channels."""
  rgb_slice = image[:, :, :3]
  augmented_rgb_slice = augment_function(rgb_slice)
  image = tf.concat([augmented_rgb_slice, image[:, :, 3:]], -1)
  return image
  
def random_adjust_brightness(image,
                             max_delta=0.2,
                             seed=None,
                             preprocess_vars_cache=None):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/preprocessor.py#L1074
  """Randomly adjusts brightness.
  Makes sure the output image is still between 0 and 255.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    max_delta: how much to change the brightness. A value between [0, 1).
    seed: random seed.
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.
  Returns:
    image: image which is the same shape as input image.
  """
  delta = tf.random.uniform([], -max_delta, max_delta, seed=seed)

  def _adjust_brightness(image):
    image = tf.image.adjust_brightness(image / 255, delta) * 255
    image = tf.cast(image, tf.uint8)
    image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=255)
    return image

  image = _augment_only_rgb_channels(image, _adjust_brightness)
  return image

def random_adjust_contrast(image,
                           min_delta=0.8,
                           max_delta=1.25,
                           seed=None,
                           preprocess_vars_cache=None):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/preprocessor.py#L1112
  """Randomly adjusts contrast.
  Makes sure the output image is still between 0 and 255.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    min_delta: see max_delta.
    max_delta: how much to change the contrast. Contrast will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current contrast of the image.
    seed: random seed.
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.
  Returns:
    image: image which is the same shape as input image.
  """
  contrast_factor = tf.random.uniform([], min_delta, max_delta, seed=seed)

  def _adjust_contrast(image):
    image = tf.image.adjust_contrast(image / 255, contrast_factor) * 255
    image = tf.cast(image, tf.uint8)
    image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=255)
    return image
  image = _augment_only_rgb_channels(image, _adjust_contrast)
  return image

def random_adjust_hue(image,
                      max_delta=0.02,
                      seed=None,
                      preprocess_vars_cache=None):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/preprocessor.py#L1153
  """Randomly adjusts hue.
  Makes sure the output image is still between 0 and 255.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    max_delta: change hue randomly with a value between 0 and max_delta.
    seed: random seed.
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.
  Returns:
    image: image which is the same shape as input image.
  """
  delta = tf.random.uniform([], -max_delta, max_delta, seed=seed)
  def _adjust_hue(image):
    image = tf.image.adjust_hue(image / 255, delta) * 255
    image = tf.cast(image, tf.uint8)
    image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=255)
    return image
  image = _augment_only_rgb_channels(image, _adjust_hue)
  return image

def random_adjust_saturation(image,
                             min_delta=0.8,
                             max_delta=1.25,
                             seed=None,
                             preprocess_vars_cache=None):
  # https://github.com/tensorflow/models/blob/859f94a23c31f385fc3fb6f73f9d4fc276a4bd6a/research/object_detection/core/preprocessor.py#L1188
  """Randomly adjusts saturation.
  Makes sure the output image is still between 0 and 255.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    min_delta: see max_delta.
    max_delta: how much to change the saturation. Saturation will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current saturation of the image.
    seed: random seed.
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.
  Returns:
    image: image which is the same shape as input image.
  """
  saturation_factor = tf.random.uniform([], min_delta, max_delta, seed=seed)
  def _adjust_saturation(image):
    image = tf.image.adjust_saturation(image / 255, saturation_factor) * 255
    image = tf.cast(image, tf.uint8)
    image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=255)
    return image
  image = _augment_only_rgb_channels(image, _adjust_saturation)
  return image

def convertColor(image, current='RGB', transform='HSV'):
    image = tf.cast(image, tf.float32)
    if current == 'RGB' and transform == 'HSV':
        image = tf.image.rgb_to_hsv(image)
    elif current == 'HSV' and transform == 'RGB':
        image =  tf.image.hsv_to_rgb (image)
    else:
        raise NotImplementedError
    image = tf.cast(image, tf.uint8)
    return image

def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.
  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.
  Args:
    tensor: A tensor of any type.
  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape

def resize_image(image,
                 masks=None,
                 new_height=600,
                 new_width=1024,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
  """Resizes images to the given height and width.
  Args:
    image: A 3D tensor of shape [height, width, channels]
    masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width] containing instance masks.
    new_height: (optional) (scalar) desired height of the image.
    new_width: (optional) (scalar) desired width of the image.
    method: (optional) interpolation method used in resizing. Defaults to
            BILINEAR.
    align_corners: bool. If true, exactly align all 4 corners of the input
                   and output. Defaults to False.
  Returns:
    Note that the position of the resized_image_shape changes based on whether
    masks are present.
    resized_image: A tensor of size [new_height, new_width, channels].
    resized_masks: If masks is not None, also outputs masks. A 3D tensor of
      shape [num_instances, new_height, new_width]
    resized_image_shape: A 1D tensor of shape [3] containing the shape of the
      resized image.
  """
  new_image = tf.image.resize(
      image, tf.stack([new_height, new_width]),
      method=method)
  image_shape = combined_static_and_dynamic_shape(image)
  result = [new_image]
  if masks is not None:
    num_instances = tf.shape(masks)[0]
    new_size = tf.stack([new_height, new_width])
    def resize_masks_branch():
      new_masks = tf.expand_dims(masks, 3)
      new_masks =  tf.compat.v1.image.resize_nearest_neighbor (
          new_masks, new_size, align_corners=align_corners)
      new_masks = tf.squeeze(new_masks, axis=3)
      return new_masks

    def reshape_masks_branch():
      # The shape function will be computed for both branches of the
      # condition, regardless of which branch is actually taken. Make sure
      # that we don't trigger an assertion in the shape function when trying
      # to reshape a non empty tensor into an empty one.
      new_masks = tf.reshape(masks, [-1, new_size[0], new_size[1]])
      return new_masks

    masks = tf.cond(num_instances > 0, resize_masks_branch,
                    reshape_masks_branch)
    result.append(masks)

  result.append(tf.stack([new_height, new_width, image_shape[2]]))
  return result

def random_augmentation(images, bboxes, masks, output_size, proto_output_size, classes):
    a_image = images
    a_bboxes = bboxes
    a_classes = classes
    a_masks = masks
    
    #   rand_angle = tf.random.uniform([1], minval=0, maxval=360)[0]
    #   img, bboxes, masks = rotate_with_bboxes(img, masks, bboxes, rand_angle)
    # a_image, a_bboxes, a_masks = random_rotation90(img, bboxes, masks)
    # a_image, a_bboxes, a_masks = random_vertical_flip(img, bboxes, masks)

    if tf.random.uniform([1], minval=0, maxval=2, dtype=tf.int32) == 1:
      a_image = random_adjust_brightness(a_image, max_delta=0.12)

    if tf.random.uniform([1], minval=0, maxval=2, dtype=tf.int32) == 1:
      if tf.random.uniform([1], minval=0, maxval=2, dtype=tf.int32) == 1:
        a_image = random_adjust_contrast(a_image, min_delta=0.5, max_delta=1.5)

      # a_image = convertColor(a_image, current='RGB', transform='HSV')
      if tf.random.uniform([1], minval=0, maxval=2, dtype=tf.int32) == 1:
        a_image = random_adjust_saturation(a_image, min_delta=0.5, max_delta=1.5)

      if tf.random.uniform([1], minval=0, maxval=2, dtype=tf.int32) == 1:
        a_image = random_adjust_hue(a_image, max_delta=0.07)

      # a_image = convertColor(a_image, current='HSV', transform='RGB')
    else:
      # a_image = convertColor(a_image, current='RGB', transform='HSV')
      if tf.random.uniform([1], minval=0, maxval=2, dtype=tf.int32) == 1:
        a_image = random_adjust_saturation(a_image, min_delta=0.5, max_delta=1.5)

      if tf.random.uniform([1], minval=0, maxval=2, dtype=tf.int32) == 1:
        a_image = random_adjust_hue(a_image, max_delta=0.07)
      # a_image = convertColor(a_image, current='HSV', transform='RGB')

      if tf.random.uniform([1], minval=0, maxval=2, dtype=tf.int32) == 1:
        a_image = random_adjust_contrast(a_image, min_delta=0.5, max_delta=1.5)
        
    if tf.random.uniform([1], minval=0, maxval=2, dtype=tf.int32) == 1:
      ratio = tf.random.uniform([1], minval=1, maxval=4)
      max_height_padding, max_width_padding = output_size
      max_height_padding = tf.cast(max_height_padding, tf.float32)*ratio
      max_width_padding = tf.cast(max_width_padding, tf.float32)*ratio
      a_image, a_bboxes, a_masks = random_absolute_pad_image(a_image, a_bboxes, a_masks, max_height_padding=max_height_padding, max_width_padding=max_width_padding)

    if tf.random.uniform([1], minval=0, maxval=2, dtype=tf.int32) == 1:
      t_flag = tf.random.uniform([5], minval=0, maxval=1)
      min_ious = (0.1, 0.3, 0.5, 0.7, 0.9)
      if t_flag[0] > 0.5:
          (a_image, a_bboxes, a_classes, _, a_masks) = random_crop_image(
                 a_image,
                 a_bboxes,
                 a_classes,
                 a_classes*0+1, # equal weights to all
                 masks=a_masks,
                 aspect_ratio_range=(0.5, 1.5),
                 overlap_thresh=min_ious[0]
                 )
      elif t_flag[1] > 0.5:
            (a_image, a_bboxes, a_classes, _, a_masks) = random_crop_image(
                   a_image,
                   a_bboxes,
                   a_classes,
                   a_classes*0+1, # equal weights to all
                   masks=a_masks,
                   aspect_ratio_range=(0.5, 1.5),
                   overlap_thresh=min_ious[1]
                   )
      elif t_flag[2] > 0.5:
            (a_image, a_bboxes, a_classes, _, a_masks) = random_crop_image(
                   a_image,
                   a_bboxes,
                   a_classes,
                   a_classes*0+1, # equal weights to all
                   masks=a_masks,
                   aspect_ratio_range=(0.5, 1.5),
                   overlap_thresh=min_ious[2]
                   )
      elif t_flag[3] > 0.5:
            (a_image, a_bboxes, a_classes, _, a_masks) = random_crop_image(
                   a_image,
                   a_bboxes,
                   a_classes,
                   a_classes*0+1, # equal weights to all
                   masks=a_masks,
                   aspect_ratio_range=(0.5, 1.5),
                   overlap_thresh=min_ious[3]
                   )
      elif t_flag[4] > 0.5:
            (a_image, a_bboxes, a_classes, _, a_masks) = random_crop_image(
                   a_image,
                   a_bboxes,
                   a_classes,
                   a_classes*0+1, # equal weights to all
                   masks=a_masks,
                   aspect_ratio_range=(0.5, 1.5),
                   overlap_thresh=min_ious[4]
                   )

    a_image, a_bboxes, a_masks = random_horizontal_flip(a_image, a_bboxes, a_masks, 123)

    a_image, a_masks, _ = resize_image(a_image, a_masks, new_height=output_size[0], new_width=output_size[1])
    img_h = tf.cast(tf.shape(a_image)[0], tf.float32)
    img_w = tf.cast(tf.shape(a_image)[1], tf.float32)
    a_bboxes = tf.stack([a_bboxes[:, 0]*output_size[0]/img_h, a_bboxes[:, 1]*output_size[1]/img_w, a_bboxes[:,2]*output_size[0]/img_h, a_bboxes[:,3]*output_size[1]/img_w], axis=1)
    _h = a_bboxes[:, 2] - a_bboxes[:, 0]
    _w = a_bboxes[:, 3] - a_bboxes[:, 1]
    keep = tf.where( tf.math.logical_and (_h > 8/output_size[0], _w > 8/output_size[1]))
    a_bboxes = tf.gather_nd(a_bboxes, keep)
    a_masks = tf.gather_nd(a_masks, keep)
    a_classes = tf.gather_nd(a_classes, keep)

    # if FLAGS[5] > 0.5:
    #   (a_image, a_bboxes, a_classes, _, a_masks) = random_square_crop_by_scale(
    #          a_image,
    #          a_bboxes,
    #          a_classes,
    #          a_classes*0+1, # equal weights to all
    #          masks=a_masks,
    #          max_border=256,
    #          scale_min=0.6,
    #          scale_max=1.3)

    # return img, bboxes, masks, classes
    return a_image, a_bboxes, a_masks, a_classes
