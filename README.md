# YOLACT Real-time Instance Segmentation
This is a Tensorflow 2.3 implementation of the paper [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689). The paper presents a fully-convolutional model for real- time instance segmentation that achieves 29.8 mAP on MS COCO at 33.5 fps evaluated on a single Titan Xp, which is significantly faster than any previous competitive approach. In this Repo implements "ResNet50-FPN". Unlike original implemetation of YOLACT in which image is resized to 550x550, this repo can handle image of size MxN. The part for training this model is ready, and the part for inference and mAP evaluation will be updated soon. <br/>

## Create TFRecord for training 
Refer to the tensorflow object detection api for tfrecord creation. ([link](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md))


## Train
### (1) Usage
Training procedure can be conducted directly by following command:
```
python train.py -tfrecord_dir 'path of TFRecord files'
                -weights 'path to store weights' 
                -train_iter 'number of  iteration for training'
                -batch_size 'batch_size'
                -lr 'learning rate'
                -momentum 'momentum for SGD'
                -weight_decay 'weight_decay rate for SGD'
                -print_interval 'interval for printing training result'
                -save_interval 'interval for conducting validation'
                -valid_iter 'number of iteration for validation'
```
The default hyperparameters in train.py follows the original setting from the paper:
* Batch size = 8, which is recommanded by paper
* SGD optimizer with learning rate 1e-3 and divided by 10 at iterations 280K, 600K, 700K and 750K, using a momentum 0.9, a weight decay 5* 1e-4. In the original implementation of paper, a warm up learning rate 1e-4 and warm up iterations 500 are used, I put all those setting in a learning schedule object in *utils/learning_rate_schedule.py*.
* Random photometrics distortion, horizontal flip(mirroring) and crop are used here for data augmentation.

## Inference (To Be Updated)

## Reference
* https://github.com/feiyuhuahuo/Yolact_minimal
* https://github.com/dbolya/yolact
* https://github.com/leohsuofnthu/Tensorflow-YOLACT/
