import tensorflow as tf
import yolact
import cv2

model = yolact.Yolact(img_h=360, 
                      img_w=1205,
                      fpn_channels=256,
                      num_class=2,
                      num_mask=32,
                      aspect_ratio=[1.81, 0.86, 0.78],
                      scales=[24, 48, 96, 130, 192])

model.load_weights('./weights/weights_3.4661324.h5')

img = cv2.imread('test.jpg')
out = model.predict(img)

from IPython import embed
embed()
