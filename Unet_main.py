import numpy as np
import tensorflow as tf

from Unet_model import UnetModel, UnetDiceLoss

BATCH_SIZE = 4
NUM_ITERS = 10000


x = tf.placeholder(tf.float32,shape=(None,500,500,3))
y = tf.placeholder(tf.float32, shape=(None,500,500,1))


pred  = UnetModel(x)
loss = UnetDiceLoss(pred,y)

grad_step = tf.train.AdamOptimizer().minimize(loss,var_list=tf.GraphKeys.TRAINABLE_VARIABLES)

with tf.Session() as sess:
    for i in range(NUM_ITERS):
        images = '''get x'''
        labels = '''get y'''
        l = sess.run([grad_step],feed_dict={x:images,y:labels})