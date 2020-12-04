from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import zip
import tensorflow.compat.v1 as tf

from utils import standard_fields
from utils import coco_tools
from utils import json_utils

def convert_masks_to_binary(masks):
  """Converts masks to 0 or 1 and uint8 type."""
  return (masks > 0).astype(np.uint8)
  
class CocoMaskEvaluator():
  """Class to evaluate COCO detection metrics."""

  def __init__(self, categories,
               include_metrics_per_category=False,
               all_metrics_per_category=False,
               super_categories=None):
    """Constructor.

    Args:
      categories: A list of dicts, each of which has the following keys -
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name e.g., 'cat', 'dog'.
      include_metrics_per_category: If True, include metrics for each category.
      all_metrics_per_category: Whether to include all the summary metrics for
        each category in per_category_ap. Be careful with setting it to true if
        you have more than handful of categories, because it will pollute
        your mldash.
      super_categories: None or a python dict mapping super-category names
        (strings) to lists of categories (corresponding to category names
        in the label_map).  Metrics are aggregated along these super-categories
        and added to the `per_category_ap` and are associated with the name
          `PerformanceBySuperCategory/<super-category-name>`.
    """
    self._categories = categories
    self._image_id_to_mask_shape_map = {}
    self._image_ids_with_detections = set([])
    self._groundtruth_list = []
    self._detection_masks_list = []
    self._category_id_set = set([cat['id'] for cat in self._categories])
    self._annotation_id = 1
    self._include_metrics_per_category = include_metrics_per_category
    self._super_categories = super_categories
    self._all_metrics_per_category = all_metrics_per_category

  def clear(self):
    """Clears the state to prepare for a fresh evaluation."""
    self._image_id_to_mask_shape_map.clear()
    self._image_ids_with_detections.clear()
    self._groundtruth_list = []
    self._detection_masks_list = []

  def add_single_ground_truth_image_info(self,
                                         image_id,
                                         groundtruth_dict):
    """Adds groundtruth for a single image to be used for evaluation.

    If the image has already been added, a warning is logged, and groundtruth is
    ignored.

    Args:
      image_id: A unique string/integer identifier for the image.
      groundtruth_dict: A dictionary containing -
        InputDataFields.groundtruth_boxes: float32 numpy array of shape
          [num_boxes, 4] containing `num_boxes` groundtruth boxes of the format
          [ymin, xmin, ymax, xmax] in absolute image coordinates.
        InputDataFields.groundtruth_classes: integer numpy array of shape
          [num_boxes] containing 1-indexed groundtruth classes for the boxes.
        InputDataFields.groundtruth_instance_masks: uint8 numpy array of shape
          [num_boxes, image_height, image_width] containing groundtruth masks
          corresponding to the boxes. The elements of the array must be in
          {0, 1}.
        InputDataFields.groundtruth_is_crowd (optional): integer numpy array of
          shape [num_boxes] containing iscrowd flag for groundtruth boxes.
        InputDataFields.groundtruth_area (optional): float numpy array of
          shape [num_boxes] containing the area (in the original absolute
          coordinates) of the annotated object.
    """
    if image_id in self._image_id_to_mask_shape_map:
      tf.logging.warning('Ignoring ground truth with image id %s since it was '
                         'previously added', image_id)
      return

    # Drop optional fields if empty tensor.
    groundtruth_is_crowd = groundtruth_dict.get(
        standard_fields.InputDataFields.groundtruth_is_crowd)
    groundtruth_area = groundtruth_dict.get(
        standard_fields.InputDataFields.groundtruth_area)
    if groundtruth_is_crowd is not None and not groundtruth_is_crowd.shape[0]:
      groundtruth_is_crowd = None
    if groundtruth_area is not None and not groundtruth_area.shape[0]:
      groundtruth_area = None

    groundtruth_instance_masks = groundtruth_dict[
        standard_fields.InputDataFields.groundtruth_instance_masks]
    groundtruth_instance_masks = convert_masks_to_binary(
        groundtruth_instance_masks)
    self._groundtruth_list.extend(
        coco_tools.
        ExportSingleImageGroundtruthToCoco(
            image_id=image_id,
            next_annotation_id=self._annotation_id,
            category_id_set=self._category_id_set,
            groundtruth_boxes=groundtruth_dict[standard_fields.InputDataFields.
                                               groundtruth_boxes],
            groundtruth_classes=groundtruth_dict[standard_fields.
                                                 InputDataFields.
                                                 groundtruth_classes],
            groundtruth_masks=groundtruth_instance_masks,
            groundtruth_is_crowd=groundtruth_is_crowd,
            groundtruth_area=groundtruth_area))
    self._annotation_id += groundtruth_dict[standard_fields.InputDataFields.
                                            groundtruth_boxes].shape[0]
    self._image_id_to_mask_shape_map[image_id] = groundtruth_dict[
        standard_fields.InputDataFields.groundtruth_instance_masks].shape

  def add_single_detected_image_info(self,
                                     image_id,
                                     detections_dict):
    """Adds detections for a single image to be used for evaluation.

    If a detection has already been added for this image id, a warning is
    logged, and the detection is skipped.

    Args:
      image_id: A unique string/integer identifier for the image.
      detections_dict: A dictionary containing -
        DetectionResultFields.detection_scores: float32 numpy array of shape
          [num_boxes] containing detection scores for the boxes.
        DetectionResultFields.detection_classes: integer numpy array of shape
          [num_boxes] containing 1-indexed detection classes for the boxes.
        DetectionResultFields.detection_masks: optional uint8 numpy array of
          shape [num_boxes, image_height, image_width] containing instance
          masks corresponding to the boxes. The elements of the array must be
          in {0, 1}.

    Raises:
      ValueError: If groundtruth for the image_id is not available or if
        spatial shapes of groundtruth_instance_masks and detection_masks are
        incompatible.
    """
    if image_id not in self._image_id_to_mask_shape_map:
      raise ValueError('Missing groundtruth for image id: {}'.format(image_id))

    if image_id in self._image_ids_with_detections:
      tf.logging.warning('Ignoring detection with image id %s since it was '
                         'previously added', image_id)
      return

    groundtruth_masks_shape = self._image_id_to_mask_shape_map[image_id]
    detection_masks = detections_dict[standard_fields.DetectionResultFields.
                                      detection_masks]
    if groundtruth_masks_shape[1:] != detection_masks.shape[1:]:
      raise ValueError('Spatial shape of groundtruth masks and detection masks '
                       'are incompatible: {} vs {}'.format(
                           groundtruth_masks_shape,
                           detection_masks.shape))
    detection_masks = convert_masks_to_binary(detection_masks)
    self._detection_masks_list.extend(
        coco_tools.ExportSingleImageDetectionMasksToCoco(
            image_id=image_id,
            category_id_set=self._category_id_set,
            detection_masks=detection_masks,
            detection_scores=detections_dict[standard_fields.
                                             DetectionResultFields.
                                             detection_scores],
            detection_classes=detections_dict[standard_fields.
                                              DetectionResultFields.
                                              detection_classes]))
    self._image_ids_with_detections.update([image_id])

  def dump_detections_to_json_file(self, json_output_path):
    """Saves the detections into json_output_path in the format used by MS COCO.

    Args:
      json_output_path: String containing the output file's path. It can be also
        None. In that case nothing will be written to the output file.
    """
    if json_output_path and json_output_path is not None:
      tf.logging.info('Dumping detections to output json file.')
      with tf.gfile.GFile(json_output_path, 'w') as fid:
        json_utils.Dump(
            obj=self._detection_masks_list, fid=fid, float_digits=4, indent=2)

  def evaluate(self):
    """Evaluates the detection masks and returns a dictionary of coco metrics.

    Returns:
      A dictionary holding -

      1. summary_metrics:
      'DetectionMasks_Precision/mAP': mean average precision over classes
        averaged over IOU thresholds ranging from .5 to .95 with .05 increments.
      'DetectionMasks_Precision/mAP@.50IOU': mean average precision at 50% IOU.
      'DetectionMasks_Precision/mAP@.75IOU': mean average precision at 75% IOU.
      'DetectionMasks_Precision/mAP (small)': mean average precision for small
        objects (area < 32^2 pixels).
      'DetectionMasks_Precision/mAP (medium)': mean average precision for medium
        sized objects (32^2 pixels < area < 96^2 pixels).
      'DetectionMasks_Precision/mAP (large)': mean average precision for large
        objects (96^2 pixels < area < 10000^2 pixels).
      'DetectionMasks_Recall/AR@1': average recall with 1 detection.
      'DetectionMasks_Recall/AR@10': average recall with 10 detections.
      'DetectionMasks_Recall/AR@100': average recall with 100 detections.
      'DetectionMasks_Recall/AR@100 (small)': average recall for small objects
        with 100 detections.
      'DetectionMasks_Recall/AR@100 (medium)': average recall for medium objects
        with 100 detections.
      'DetectionMasks_Recall/AR@100 (large)': average recall for large objects
        with 100 detections.

      2. per_category_ap: if include_metrics_per_category is True, category
      specific results with keys of the form:
      'Precision mAP ByCategory/category' (without the supercategory part if
      no supercategories exist). For backward compatibility
      'PerformanceByCategory' is included in the output regardless of
      all_metrics_per_category.
        If super_categories are provided, then this will additionally include
      metrics aggregated along the super_categories with keys of the form:
      `PerformanceBySuperCategory/<super-category-name>`
    """
    groundtruth_dict = {
        'annotations': self._groundtruth_list,
        'images': [{'id': image_id, 'height': shape[1], 'width': shape[2]}
                   for image_id, shape in self._image_id_to_mask_shape_map.
                   items()],
        'categories': self._categories
    }
    coco_wrapped_groundtruth = coco_tools.COCOWrapper(
        groundtruth_dict, detection_type='segmentation')
    coco_wrapped_detection_masks = coco_wrapped_groundtruth.LoadAnnotations(
        self._detection_masks_list)
    mask_evaluator = coco_tools.COCOEvalWrapper(
        coco_wrapped_groundtruth, coco_wrapped_detection_masks,
        agnostic_mode=False, iou_type='segm')
    mask_metrics, mask_per_category_ap = mask_evaluator.ComputeMetrics(
        include_metrics_per_category=self._include_metrics_per_category,
        super_categories=self._super_categories,
        all_metrics_per_category=self._all_metrics_per_category)
    mask_metrics.update(mask_per_category_ap)
    mask_metrics = {'DetectionMasks_'+ key: value
                    for key, value in mask_metrics.items()}
    return mask_metrics

  def add_eval_dict(self, eval_dict):
    """Observes an evaluation result dict for a single example.

    When executing eagerly, once all observations have been observed by this
    method you can use `.evaluate()` to get the final metrics.

    When using `tf.estimator.Estimator` for evaluation this function is used by
    `get_estimator_eval_metric_ops()` to construct the metric update op.

    Args:
      eval_dict: A dictionary that holds tensors for evaluating an object
        detection model, returned from
        eval_util.result_dict_for_single_example().

    Returns:
      None when executing eagerly, or an update_op that can be used to update
      the eval metrics in `tf.estimator.EstimatorSpec`.
    """
    def update_op(image_id_batched, groundtruth_boxes_batched,
                  groundtruth_classes_batched,
                  groundtruth_instance_masks_batched,
                  groundtruth_is_crowd_batched, num_gt_boxes_per_image,
                  detection_scores_batched, detection_classes_batched,
                  detection_masks_batched, num_det_boxes_per_image,
                  original_image_spatial_shape):
      """Update op for metrics."""

      for (image_id, groundtruth_boxes, groundtruth_classes,
           groundtruth_instance_masks, groundtruth_is_crowd, num_gt_box,
           detection_scores, detection_classes,
           detection_masks, num_det_box, original_image_shape) in zip(
               image_id_batched, groundtruth_boxes_batched,
               groundtruth_classes_batched, groundtruth_instance_masks_batched,
               groundtruth_is_crowd_batched, num_gt_boxes_per_image,
               detection_scores_batched, detection_classes_batched,
               detection_masks_batched, num_det_boxes_per_image,
               original_image_spatial_shape):
        self.add_single_ground_truth_image_info(
            image_id, {
                'groundtruth_boxes':
                    groundtruth_boxes[:num_gt_box],
                'groundtruth_classes':
                    groundtruth_classes[:num_gt_box],
                'groundtruth_instance_masks':
                    groundtruth_instance_masks[
                        :num_gt_box,
                        :original_image_shape[0],
                        :original_image_shape[1]],
                'groundtruth_is_crowd':
                    groundtruth_is_crowd[:num_gt_box]
            })
        self.add_single_detected_image_info(
            image_id, {
                'detection_scores': detection_scores[:num_det_box],
                'detection_classes': detection_classes[:num_det_box],
                'detection_masks': detection_masks[
                    :num_det_box,
                    :original_image_shape[0],
                    :original_image_shape[1]]
            })

    # Unpack items from the evaluation dictionary.
    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    image_id = eval_dict[input_data_fields.key]
    original_image_spatial_shape = eval_dict[
        input_data_fields.original_image_spatial_shape]
    groundtruth_boxes = eval_dict[input_data_fields.groundtruth_boxes]
    groundtruth_classes = eval_dict[input_data_fields.groundtruth_classes]
    groundtruth_instance_masks = eval_dict[
        input_data_fields.groundtruth_instance_masks]
    groundtruth_is_crowd = eval_dict.get(
        input_data_fields.groundtruth_is_crowd, None)
    num_gt_boxes_per_image = eval_dict.get(
        input_data_fields.num_groundtruth_boxes, None)
    detection_scores = eval_dict[detection_fields.detection_scores]
    detection_classes = eval_dict[detection_fields.detection_classes]
    detection_masks = eval_dict[detection_fields.detection_masks]
    num_det_boxes_per_image = eval_dict.get(detection_fields.num_detections,
                                            None)

    if groundtruth_is_crowd is None:
      groundtruth_is_crowd = tf.zeros_like(groundtruth_classes, dtype=tf.bool)

    if not image_id.shape.as_list():
      # Apply a batch dimension to all tensors.
      image_id = tf.expand_dims(image_id, 0)
      groundtruth_boxes = tf.expand_dims(groundtruth_boxes, 0)
      groundtruth_classes = tf.expand_dims(groundtruth_classes, 0)
      groundtruth_instance_masks = tf.expand_dims(groundtruth_instance_masks, 0)
      groundtruth_is_crowd = tf.expand_dims(groundtruth_is_crowd, 0)
      detection_scores = tf.expand_dims(detection_scores, 0)
      detection_classes = tf.expand_dims(detection_classes, 0)
      detection_masks = tf.expand_dims(detection_masks, 0)

      if num_gt_boxes_per_image is None:
        num_gt_boxes_per_image = tf.shape(groundtruth_boxes)[1:2]
      else:
        num_gt_boxes_per_image = tf.expand_dims(num_gt_boxes_per_image, 0)

      if num_det_boxes_per_image is None:
        num_det_boxes_per_image = tf.shape(detection_scores)[1:2]
      else:
        num_det_boxes_per_image = tf.expand_dims(num_det_boxes_per_image, 0)
    else:
      if num_gt_boxes_per_image is None:
        num_gt_boxes_per_image = tf.tile(
            tf.shape(groundtruth_boxes)[1:2],
            multiples=tf.shape(groundtruth_boxes)[0:1])
      if num_det_boxes_per_image is None:
        num_det_boxes_per_image = tf.tile(
            tf.shape(detection_scores)[1:2],
            multiples=tf.shape(detection_scores)[0:1])

    return tf.py_func(update_op, [
        image_id, groundtruth_boxes, groundtruth_classes,
        groundtruth_instance_masks, groundtruth_is_crowd,
        num_gt_boxes_per_image, detection_scores, detection_classes,
        detection_masks, num_det_boxes_per_image, original_image_spatial_shape
    ], [])

  def get_estimator_eval_metric_ops(self, eval_dict):
    """Returns a dictionary of eval metric ops.

    Note that once value_op is called, the detections and groundtruth added via
    update_op are cleared.

    Args:
      eval_dict: A dictionary that holds tensors for evaluating object detection
        performance. For single-image evaluation, this dictionary may be
        produced from eval_util.result_dict_for_single_example(). If multi-image
        evaluation, `eval_dict` should contain the fields
        'num_groundtruth_boxes_per_image' and 'num_det_boxes_per_image' to
        properly unpad the tensors from the batch.

    Returns:
      a dictionary of metric names to tuple of value_op and update_op that can
      be used as eval metric ops in tf.estimator.EstimatorSpec. Note that all
      update ops  must be run together and similarly all value ops must be run
      together to guarantee correct behaviour.
    """
    update_op = self.add_eval_dict(eval_dict)
    metric_names = ['DetectionMasks_Precision/mAP',
                    'DetectionMasks_Precision/mAP@.50IOU',
                    'DetectionMasks_Precision/mAP@.75IOU',
                    'DetectionMasks_Precision/mAP (small)',
                    'DetectionMasks_Precision/mAP (medium)',
                    'DetectionMasks_Precision/mAP (large)',
                    'DetectionMasks_Recall/AR@1',
                    'DetectionMasks_Recall/AR@10',
                    'DetectionMasks_Recall/AR@100',
                    'DetectionMasks_Recall/AR@100 (small)',
                    'DetectionMasks_Recall/AR@100 (medium)',
                    'DetectionMasks_Recall/AR@100 (large)']
    if self._include_metrics_per_category:
      for category_dict in self._categories:
        metric_names.append('DetectionMasks_PerformanceByCategory/mAP/' +
                            category_dict['name'])

    def first_value_func():
      self._metrics = self.evaluate()
      self.clear()
      return np.float32(self._metrics[metric_names[0]])

    def value_func_factory(metric_name):
      def value_func():
        return np.float32(self._metrics[metric_name])
      return value_func

    # Ensure that the metrics are only evaluated once.
    first_value_op = tf.py_func(first_value_func, [], tf.float32)
    eval_metric_ops = {metric_names[0]: (first_value_op, update_op)}
    with tf.control_dependencies([first_value_op]):
      for metric_name in metric_names[1:]:
        eval_metric_ops[metric_name] = (tf.py_func(
            value_func_factory(metric_name), [], np.float32), update_op)
    return eval_metric_ops
