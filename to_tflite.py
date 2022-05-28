import tensorflow as tf
import os
import cv2

IMAGE_PATH = '/home/sort/ved/test/'

#train_images = tf.keras.preprocessing.image_dataset_from_directory(
#        directory=IMAGE_PATH, labels='inferred', label_mode='int', class_names=None,
#        color_mode='rgb', batch_size=1)

#def representative_dataset():
#  for i in range(25):
#    image = train_images[i]
#    image = tf.io.read_file(image)
#    image = tf.io.decode_jpeg(image, channels=3)
#    image = tf.image.resize(image, [2410, 720])
#    image = tf.cast(image / 255., tf.float64)
#    image = tf.expand_dims(image, 0)
#    yield [image]


images = []
for img in os.listdir(IMAGE_PATH):
    if img.endswith("jpg"):
        image = os.path.join(IMAGE_PATH, img)
        print('found image ', image)
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tf.image.resize(image, [550, 550])
        images.append(image)

def representative_dataset():
  for data in tf.data.Dataset.from_tensor_slices((images)).batch(1).take(50):
    yield [tf.dtypes.cast(data, tf.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model('saved_models_cc_fast_nms/saved_model_1.1352494')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.experimental_new_quantizer = True
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
            ]
converter.inference_input_type = tf.uint8  # or tf.uint8
# converter.inference_output_type = tf.uint8  # or tf.uint8
tflite_quant_model = converter.convert()

open("yolact.tflite", "wb").write(tflite_quant_model)
