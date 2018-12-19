import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops

def UnetModel(image):
    with tf.scope('down_block1'):
        x = tf.layers.conv2d(image,filters = 64,kernel_size = 3,padding = 'same',activation = tf.nn.leaky_relu)
        short1 = tf.layers.conv2d(x,filters = 64,kernel_size = 3,padding = 'same',activation = tf.nn.leaky_relu)
        
        x = tf.layers.pooling2d(short1,pool_size = 2,strides = 1, padding = 'same')
    
    with tf.scope('down_block2'):
        x = tf.layers.conv2d(x,filters = 128,kernel_size = 3,padding = 'same',activation = tf.nn.leaky_relu)
        short2 = tf.layers.conv2d(x,filters = 128,kernel_size = 3,padding = 'same',activation = tf.nn.leaky_relu)
        
        x = tf.layers.pooling2d(short2,pool_size = 2,strides = 1, padding = 'same')    
    
    with tf.scope('down_block3'):
        x = tf.layers.conv2d(x,filters = 256,kernel_size = 3,padding = 'same',activation = tf.nn.leaky_relu)
        short3 = tf.layers.conv2d(x,filters = 256,kernel_size = 3,padding = 'same',activation = tf.nn.leaky_relu)
        
        x = tf.layers.pooling2d(short3,pool_size = 2,strides = 1, padding = 'same')
    
    with tf.scope('down_block4'):
        x = tf.layers.conv2d(x,filters = 512,kernel_size = 3,padding = 'same',activation = tf.nn.leaky_relu)
        short4 = tf.layers.conv2d(x,filters = 512,kernel_size = 3,padding = 'same',activation = tf.nn.leaky_relu)
        
        x = tf.layers.pooling2d(short4,pool_size = 2,strides = 1, padding = 'same')    
    
    with tf.scope('horizontal_block'):
        x = tf.layers.conv2d(x,filters = 1024,kernel_size = 3,padding = 'same',activation = tf.nn.leaky_relu)
        short2 = tf.layers.conv2d(x,filters = 1024,kernel_size = 3,padding = 'same',activation = tf.nn.leaky_relu)
        
        x = tf.layers.pooling2d(short1,pool_size = 2,strides = 1, padding = 'same')

    with tf.scope('up_block1'):
        x = tf.layers.conv2d_transpose(x,filters = 512,kernel_size = 3,padding = 'same')
        x = tf.concat([short4,x],axis = -1)
        
        x = tf.layers.conv2d(x,filters = 512,kernel_size = 3,padding = 'same',activation = tf.nn.leaky_relu)
        x = tf.layers.conv2d(x,filters = 512,kernel_size = 3,padding = 'same',activation = tf.nn.leaky_relu)
        
    with tf.scope('up_block2'):
        x = tf.layers.conv2d_transpose(x,filters = 256,kernel_size = 3,padding = 'same')
        x = tf.concat([short3,x],axis = -1)
        
        x = tf.layers.conv2d(x,filters = 256,kernel_size = 3,padding = 'same',activation = tf.nn.leaky_relu)
        x = tf.layers.conv2d(x,filters = 256,kernel_size = 3,padding = 'same',activation = tf.nn.leaky_relu)
        
    with tf.scope('up_block3'):
        x = tf.layers.conv2d_transpose(x,filters = 128,kernel_size = 3,padding = 'same')
        x = tf.concat([short2,x],axis = -1)
        
        x = tf.layers.conv2d(x,filters = 128,kernel_size = 3,padding = 'same',activation = tf.nn.leaky_relu)
        x = tf.layers.conv2d(x,filters = 128,kernel_size = 3,padding = 'same',activation = tf.nn.leaky_relu)
    
    with tf.scope('up_block4'):
        x = tf.layers.conv2d_transpose(x,filters = 64,kernel_size = 3,padding = 'same')
        x = tf.concat([short1,x],axis = -1)
        
        x = tf.layers.conv2d(x,filters = 64,kernel_size = 3,padding = 'same',activation = tf.nn.leaky_relu)
        x = tf.layers.conv2d(x,filters = 64,kernel_size = 3,padding = 'same',activation = tf.nn.leaky_relu)
        
    x  = tf.layers.conv2d(x,filters = 1,kernel_size = 1, padding = 'same',activation = tf.nn.sigmoid) 
    return x
    
def UnetDiceLoss(out,target,loss_type='jaccard', axis=(1, 2, 3),smooth=1e-5):
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        #axis=(1,2,3), therefore reduce to tensor of shape (batch_size,1)
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)

    dice = tf.reduce_mean(dice, name='dice_coe')
    return dice
    
    
def UnetFocalLoss(out, target, alpha=0.25, gamma=2):
    """Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     out: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0))
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    
    return tf.reduce_sum(per_entry_cross_ent)
    