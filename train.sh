python3 train.py -tfrecord_train_dir '/home/deploy/ved/data/train' -tfrecord_val_dir '/home/deploy/ved/data/val' -label_map 'label_map.pbtxt' -train_iter '102000' -valid_iter '20' -batch_size '2' -img_h '720' -img_w '2410' -num_class '1' -aspect_ratio '1,0.5, 2' -scale '24, 48, 96, 130, 192' -use_dcn True
