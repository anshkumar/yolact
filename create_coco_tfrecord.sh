python3 create_coco_tfrecord.py --logtostderr \
      --include_masks="True" \
      --image_dir="/home/deploy/ved/coco/val2017" \
      --object_annotations_file="/home/deploy/ved/coco/annotations/instances_val2017.json" \
      --output_file_prefix="/home/deploy/ved/coco/records/val/records" \
      --num_shards=100

