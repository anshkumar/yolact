import tensorflow as tf
import yolact
from pycocotools import mask as m
import numpy as np
import cv2
import os
from google.protobuf import text_format
from protos import string_int_label_map_pb2
import json
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('eval_dir', './coco/test',
                    'directory of testing images')
flags.DEFINE_string('img_info', './coco/annotations/image_info_test-dev2017.json',
                    'Image info file.')
flags.DEFINE_string('out_dir', './out',
                    'Output image dir.')
flags.DEFINE_string('label_map', './label_map.pbtxt',
                    'path to label_map.pbtxt')
flags.DEFINE_string('output_json', './detections_test-dev2017_yolact_results.json',
                    'json_output_path to save in the format used by MS COCO')
flags.DEFINE_string('saved_model_dir', None,
                    'saved_model directory containg inference model')

def _validate_label_map(label_map):
  # https://github.com/tensorflow/models/blob/
  # 67fd2bef6500c14b95e0b0de846960ed13802405/research/object_detection/utils/
  # label_map_util.py#L34
  """Checks if a label map is valid.
  Args:
    label_map: StringIntLabelMap to validate.
  Raises:
    ValueError: if label map is invalid.
  """
  for item in label_map.item:
    if item.id < 0:
      raise ValueError('Label map ids should be >= 0.')
    if (item.id == 0 and item.name != 'background' and
        item.display_name != 'background'):
      raise ValueError('Label map id 0 is reserved for the background label')

def load_labelmap(path):
  # https://github.com/tensorflow/models/blob/
  # 67fd2bef6500c14b95e0b0de846960ed13802405/research/object_detection/utils/
  # label_map_util.py#L159
  """Loads label map proto.
  Args:
    path: path to StringIntLabelMap proto text file.
  Returns:
    a StringIntLabelMapProto
  """
  with tf.io.gfile.GFile(path, 'r') as fid:
    label_map_string = fid.read()
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    try:
      text_format.Merge(label_map_string, label_map)
    except text_format.ParseError:
      label_map.ParseFromString(label_map_string)
  _validate_label_map(label_map)
  return label_map

def _get_categories_list(label_map_path):
    # https://github.com/tensorflow/models/blob/\
    # 67fd2bef6500c14b95e0b0de846960ed13802405/research/cognitive_planning/
    # label_map_util.py#L73
    '''
    return [{
          'id': 1,
          'name': 'person'
      }, {
          'id': 2,
          'name': 'dog'
      }, {
          'id': 3,
          'name': 'cat'
      }]
    '''
    label_map = load_labelmap(label_map_path)
    categories = []
    list_of_ids_already_added = []
    for item in label_map.item:
        name = item.name
        if item.id not in list_of_ids_already_added:
          list_of_ids_already_added.append(item.id)
          categories.append({'id': item.id, 'name': name})
    return categories

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def main(argv):
    model = tf.saved_model.load(FLAGS.saved_model_dir)
    infer = model.signatures["serving_default"]

    # COCO evalator for showing MAP
    annotations_lst = []
    categories = _get_categories_list(FLAGS.label_map)
    count_id = 0

    with open(FLAGS.img_info) as f:
      info = json.load(f)

    for _info in info["images"]:
            img = _info["file_name"]
            image_org = cv2.imread(os.path.join(FLAGS.eval_dir, img))
            image = cv2.resize(image_org, (550, 550))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
            output = infer(tf.constant(image[None, ...]))

            _h = image_org.shape[0]
            _w = image_org.shape[1]

            det_num = np.count_nonzero(output['detection_scores'][0].numpy()> 0.15)
            det_boxes = output['detection_boxes'][0][:det_num]
            det_boxes = det_boxes.numpy()*np.array([_h,_w,_h,_w])
            det_masks = output['detection_masks'][0][:det_num].numpy()

            det_scores = output['detection_scores'][0][:det_num].numpy()
            det_classes = output['detection_classes'][0][:det_num].numpy()

            for i in range(det_num):
                count_id += 1
                _mask = det_masks[i]
                _mask = cv2.resize(_mask, (_w, _h))
                mask = np.array(_mask > 0.5, dtype=np.bool, order='F')
                rle_mask = m.encode(mask)
                rle_mask['counts'] = rle_mask['counts'].decode('ascii')

                _y1, _x1, _y2, _x2  = det_boxes[i].astype(int)
                _y1, _x1, _y2, _x2 = int(_y1), int(_x1), int(_y2), int(_x2)
                _class = int(det_classes[i])

                cv2.rectangle(image_org, (_x1, _y1), (_x2, _y2), (0, 255, 0), 2)
                cv2.putText(image_org, str(_class)+'; '+str(round(det_scores[i],2)), (_x1, _y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), lineType=cv2.LINE_AA)
                mask = (_mask > 0.5)
                roi = image_org[mask]
                blended = roi.astype("uint8")
                image_org[mask] = blended*[0,0,1]

                annotations_lst.append({
                    "image_id": _info["id"],
                    "category_id": _class,
                    "segmentation": rle_mask,
                    "score": float(det_scores[i])
                    })
            cv2.imwrite(os.path.join(FLAGS.out_dir, _info["file_name"]), image_org)

    with open(FLAGS.output_json, 'w') as f:
        json.dump(annotations_lst, f)

if __name__ == '__main__':
    app.run(main)
