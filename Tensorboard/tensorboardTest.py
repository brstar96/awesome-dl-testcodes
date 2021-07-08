# Read YUV image stored in *.pt and visualization with tensorboard

import cv2, pickle
import tensorboard as tf
import tensorflow as tf
import numpy as np

img_path = '/home/datasets/DIV2K/DIV2K_train_HR/0801.png'

with open('/home/datasets/DIV2K/DIV2K_train_HR_pt/0801.pt', 'rb') as f:
    img_YUV_frompt = pickle.load(f)

w = tf.summary.create_file_writer('test/logs')
with w.as_default():
    img_RGB_changefrompt = cv2.cvtColor(img_YUV_frompt, cv2.COLOR_YUV2BGR)
    img_BGR = cv2.imread(img_path)
    img_RGB = tf.convert_to_tensor(img_BGR)
    img_RGB_changefrompt = tf.convert_to_tensor(img_RGB_changefrompt)
    img_YUV = tf.convert_to_tensor(img_YUV_frompt)
    img_Y = tf.convert_to_tensor(np.expand_dims(img_YUV_frompt[:, :, 0], axis=-1))

    print('img_shape:', img_Y.shape, img_YUV.shape)

    tf.summary.image("original RGB", [img_RGB], step=0)
    tf.summary.image("RGB from pt", [img_RGB_changefrompt], step=0)
    tf.summary.image("YUV from pt", [img_YUV], step=0)
    tf.summary.image("Y from pt", [img_Y], step=0)