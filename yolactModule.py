import tensorflow as tf

class YOLACTModule(tf.Module):
  """Inference Module for TFLite-friendly models."""

  def __init__(self, detection_model, use_regular_nms):
    """Initialization.
    Args:
      detection_model: The detection model to use for inference.
      use_regular_nms: If True, TFLite model uses the (slower) multi-class NMS.
    """
    self._model = detection_model
    self._use_regular_nms = use_regular_nms

  def _get_postprocess_fn(self):
    # There is no TF equivalent for TFLite's custom post-processing op.
    # So we add an 'empty' composite function here, that is legalized to the
    # custom op with MLIR.
    @tf.function()
    # pylint: disable=g-unused-argument,unused-argument
    def dummy_post_processing(box_encodings, class_predictions, anchors):
      boxes = tf.constant(0.0, dtype=tf.float32, name='boxes')
      scores = tf.constant(0.0, dtype=tf.float32, name='scores')
      classes = tf.constant(0.0, dtype=tf.float32, name='classes')
      masks = tf.constant(0.0, dtype=tf.float32, name='masks')
      num_detections = tf.constant(0.0, dtype=tf.float32, name='num_detections')
      return boxes, classes, scores, masks, num_detections

    return dummy_post_processing

  @tf.function
  def inference_fn(self, image):
    """Encapsulates model inference for TFLite conversion.
    NOTE: The Args & Returns sections below indicate the TFLite model signature,
    and not what the TF graph does (since the latter does not include the custom
    NMS op used by TFLite)
    Args:
      image: a float32 tensor of shape [1, image_height, image_width, channel]
        denoting the image pixel values.
    Returns:
      num_detections: a float32 scalar denoting number of total detections.
      classes: a float32 tensor denoting class ID for each detection.
      scores: a float32 tensor denoting score for each detection.
      boxes: a float32 tensor denoting coordinates of each detected box.
    """
    predicted_tensors = self._model(image)
    # The score conversion occurs before the post-processing custom op

    with tf.name_scope('raw_outputs'):
      # 'raw_outputs/box_encodings': a float32 tensor of shape
      #   [1, num_anchors, 4] containing the encoded box predictions. Note that
      #   these are raw predictions and no Non-Max suppression is applied on
      #   them and no decode center size boxes is applied to them.
      box_encodings = tf.identity(
          predicted_tensors['pred_offset'], name='box_encodings')
      # 'raw_outputs/class_predictions': a float32 tensor of shape
      #   [1, num_anchors, num_classes] containing the class scores for each
      #   anchor after applying score conversion.
      class_predictions = tf.identity(
          predicted_tensors['pred_cls'], name='class_predictions')
      mask_coef_predictions = tf.identity(
          predicted_tensors['pred_mask_coef'], name='mask_coef_predictions')
      mask_proto_predictions = tf.identity(
          predicted_tensors['proto_out'], name='mask_proto_predictions')

    # 'anchors': a float32 tensor of shape
    #   [num_anchors, 4] containing the anchors as a constant node.
    anchors = tf.identity(
          predicted_tensors['priors'], name='anchors')

    # tf.function@ seems to reverse order of inputs, so reverse them here.
    return self._get_postprocess_fn()(box_encodings,
        class_predictions,
        anchors)[::-1]
