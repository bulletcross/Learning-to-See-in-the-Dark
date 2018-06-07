#uniform content loss + adaptive threshold + per_class_input + recursive G
#improvement upon cqf37
from __future__ import division
import os,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
import pdb
import rawpy
import glob


input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
checkpoint_dir = './result_Sony/'
result_dir = './result_Sony/'

#get train and test IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = []
for i in range(len(train_fns)):
    _, train_fn = os.path.split(train_fns[i])
    train_ids.append(int(train_fn[0:5]))

test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = []
for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    test_ids.append(int(test_fn[0:5]))



ps = 512 #patch size for training
save_freq = 500


# For debugging on example images
DEBUG = 1
if DEBUG == 1:
  save_freq = 2
  train_ids = train_ids[0:5]
  test_ids = test_ids[0:5]


"""
Leaky-Relu : 0 activation is replaced with linear function
A helpful function for transposition
Input: Tensor of any size
Output: Tensor of Input size
"""
def lrelu(x):
    return tf.maximum(x*0.2,x)

"""
Upsampling step in decoder network
Inputs: Tensor1, Tensor2, nr_out_ch(Tensor2), nr_in_ch(Tensor1)
Output: Tensor of same shape as Tensor2, with 2*nr_out_ch
"""
def upsample_and_concat(x1, x2, output_channels, in_channels, name):
    pool_size = 2
    with tf.name_scope(name) as scope:
        deconv_filter = tf.Variable(tf.truncated_normal( [pool_size, pool_size, output_channels, in_channels], stddev=0.02))
        # tf.shape(x2) is the output shape of deconv operation
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )
        # Number of output channels are now doubled
        deconv_output =  tf.concat([deconv, x2],3)
        deconv_output.set_shape([None, None, None, output_channels*2])
        return deconv_output

"""
Self descriptive
"""
def conv(input, filter_size, nr_filters, stride, name, padding = 'SAME', dilation = 1):
    input_channels = int(input.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name = name + '_weights', shape = [filter_size, filter_size, input_channels, nr_filters])
        biases = tf.get_variable(name = name + '_biases', shape = [nr_filters])
        conv = tf.nn.conv2d(input, weights, strides = [1, stride, stride, 1], padding = padding, name = name)
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        relu = tf.nn.relu(bias, name = scope.name)
        return relu


"""
Self descriptive
"""
def depthconv(input, filter_size, stride, name, padding = 'SAME', dilation = 1, multiplier = 1):
    input_channels = int(input.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name = name + '_weights', shape = [filter_size, filter_size, input_channels, multiplier])
        biases = tf.get_variable(name = name + '_biases', shape = [input_channels*multiplier])
        depthconv = tf.nn.depthwise_conv2d(input, weights, strides = [1, stride, stride, 1],
                                      padding = padding, rate = [dilation, dilation], name = name)
        bias = tf.reshape(tf.nn.bias_add(depthconv, biases), depthconv.get_shape().as_list())
        relu = tf.nn.relu(bias, name = scope.name)
        return relu

"""
Self descriptive
"""
def pointconv(input, nr_filters, stride, name):
    input_channels = int(input.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name = name + '_weights', shape = [1, 1, input_channels, nr_filters])
        biases = tf.get_variable(name = name + '_biases', shape = [nr_filters])
        pointconv = tf.nn.conv2d(input, weights, strides = [1, stride, stride, 1], padding = 'SAME', name = name)
        bias = tf.reshape(tf.nn.bias_add(pointconv, biases), pointconv.get_shape().as_list())
        relu = tf.nn.relu(bias, name = scope.name)
        return relu

"""
Inputs: Tensor1, Tensor2 (same shape)
Output: Tensor of same shape with same number of channels as any input
"""
def channel_weighted_pooling(weights, channel):
    nr_channel = channel.get_shape().as_list()[-1]
        with tf.name_scope('cwp') as scope:
        pool_weights = tf.split(weights, num_or_size_splits = nr_channel, axis = 3)
        channel_outputs = tf.split(channel, num_or_size_splits = nr_channel, axis = 3)
        prod = []
        for i in range(nr_channel):
            prod.append(tf.multiply(pool_weights[i], channel_outputs[i]))
        output = tf.concat(prod, axis = 3)
        return output

"""
Inputs: Tensor1, Tensor2
Outputs: Tensor1, Tensor2 (H/4, W/4, C*4)
"""
def intermediate_residual(depth_in, point_in, name):
    nr_channels = int(depth_in.get_shape()[-1])
    with tf.name_scope('ir') as scope:
        depthconv_inter = depthconv(depth_in, filter_size = 3, stride = 2,
                                        padding = 'SAME', name = name+'_depth1',
                                        dilation = 1, multiplier = 2)
        pointconv_inter = pointconv(point_in, nr_channels*2, stride = 2,
                                        name = name+'_point1')
        tensor_inter = tf.concat([depth_in, point_in], axis = 3)
        # tensor_inter is of half the spatial dimention and four times the input tensor
        conv_inter = conv(tensor_inter, filter_size = 3, nr_filters = 2*nr_channels,
                            stride = 2, name = name+'_conv', padding = 'SAME')
        # By this time, channels are halved, so double on original input
        # Use shuffling and fire module here, instead of conv
        depth_out_tensor = tf.concat([depthconv_inter, conv_inter], axis = 3)
        # Four times the input channel now
        depthconv_out = depthconv(depth_out_tensor, filter_size = 3, stride = 2,
                                        padding = 'SAME', name = name+'_depth2',
                                        dilation = 1, multiplier = 1)
        # Again spatial dims halved, so one-fourth of original
        point_out_tensor = tf.concat([pointconv_inter, conv_inter], axis = 3)
        pointconv_out = pointconv(point_out_tensor, nr_channels*2*2, stride = 2,
                                        name = name+'_point2')
        return depthconv_out, pointconv_out

def network(input):
    conv1=slim.conv2d(input,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
    conv1=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )

    conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
    conv2=slim.conv2d(conv2,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )

    conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_1')
    conv3=slim.conv2d(conv3,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )

    conv4=slim.conv2d(pool3,256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_1')
    conv4=slim.conv2d(conv4,256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )

    conv5=slim.conv2d(pool4,512,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_1')
    conv5=slim.conv2d(conv5,512,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_2')

    up6 =  upsample_and_concat( conv5, conv4, 256, 512 , 'up_conv1' )
    conv6=slim.conv2d(up6,  256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_1')
    conv6=slim.conv2d(conv6,256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_2')

    up7 =  upsample_and_concat( conv6, conv3, 128, 256 , 'up_conv2' )
    conv7=slim.conv2d(up7,  128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
    conv7=slim.conv2d(conv7,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')

    up8 =  upsample_and_concat( conv7, conv2, 64, 128 , 'up_conv3')
    conv8=slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
    conv8=slim.conv2d(conv8,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_2')

    up9 =  upsample_and_concat( conv8, conv1, 32, 64 , 'up_conv4')
    conv9=slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
    conv9=slim.conv2d(conv9,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_2')

    conv10=slim.conv2d(conv9,12,[1,1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10,2)
    return out

def network_modified(input):
    """[input -> 32_channel -> 64_channel] (lrelu, lrelu)

    one residual, with spatial conv1 and conv2

    [64_channel -> 128_channel -> 256_channel] (lrelu, lrelu)

    one residual, with spatial conv3 and conv4

    [256_channel -> 512_channel] (no activation)

    conv5 = weighted_pooled after residual(should halve channels now)
    """

    up6 =  upsample_and_concat( conv5, conv4, 256, 512 , 'up_conv1' )
    conv6=slim.conv2d(up6,  256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_1')
    conv6=slim.conv2d(conv6,256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_2')

    up7 =  upsample_and_concat( conv6, conv3, 128, 256 , 'up_conv2' )
    conv7=slim.conv2d(up7,  128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
    conv7=slim.conv2d(conv7,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')

    up8 =  upsample_and_concat( conv7, conv2, 64, 128 , 'up_conv3')
    conv8=slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
    conv8=slim.conv2d(conv8,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_2')

    up9 =  upsample_and_concat( conv8, conv1, 32, 64 , 'up_conv4')
    conv9=slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
    conv9=slim.conv2d(conv9,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_2')

    conv10=slim.conv2d(conv9,12,[1,1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10,2)
    return out


def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512,0)/ (16383 - 512) #subtract the black level

    im = np.expand_dims(im,axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:],
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out



sess=tf.Session()
in_image=tf.placeholder(tf.float32,[None,None,None,4])
gt_image=tf.placeholder(tf.float32,[None,None,None,3])
out_image=network(in_image)

G_loss=tf.reduce_mean(tf.abs(out_image - gt_image))

t_vars=tf.trainable_variables()
lr=tf.placeholder(tf.float32)
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss,var_list=[var for var in t_vars if var.name.startswith('g_')])

saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

#Raw data takes long time to load. Keep them in memory after loaded.
gt_images=[None]*6000
input_images = {}
input_images['300'] = [None]*len(train_ids)
input_images['250'] = [None]*len(train_ids)
input_images['100'] = [None]*len(train_ids)

g_loss = np.zeros((5000,1))



allfolders = glob.glob('./result/*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

learning_rate = 1e-4
for epoch in range(lastepoch,4001):
    if os.path.isdir("result/%04d"%epoch):
        continue
    cnt=0
    if epoch > 2000:
        learning_rate = 1e-5

    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW'%train_id)
        in_path = in_files[np.random.random_integers(0,len(in_files)-1)]
        _, in_fn = os.path.split(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW'%train_id)
        gt_path = gt_files[0]
        _, gt_fn = os.path.split(gt_path)
        in_exposure =  float(in_fn[9:-5])
        gt_exposure =  float(gt_fn[9:-5])
        ratio = min(gt_exposure/in_exposure,300)

        st=time.time()
        cnt+=1

        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw),axis=0) *ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im/65535.0),axis = 0)


        #crop
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0,W-ps)
        yy = np.random.randint(0,H-ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:,yy:yy+ps,xx:xx+ps,:]
        gt_patch = gt_images[ind][:,yy*2:yy*2+ps*2,xx*2:xx*2+ps*2,:]

        if np.random.randint(2,size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2,size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=0)
            gt_patch = np.flip(gt_patch, axis=0)
        if np.random.randint(2,size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0,2,1,3))
            gt_patch = np.transpose(gt_patch, (0,2,1,3))

        input_patch = np.minimum(input_patch,1.0)

        _,G_current,output=sess.run([G_opt,G_loss,out_image],feed_dict={in_image:input_patch,gt_image:gt_patch,lr:learning_rate})
        output = np.minimum(np.maximum(output,0),1)
        g_loss[ind]=G_current

        print("%d %d Loss=%.3f Time=%.3f"%(epoch,cnt,np.mean(g_loss[np.where(g_loss)]),time.time()-st))

        if epoch%save_freq==0:
          if not os.path.isdir(result_dir + '%04d'%epoch):
              os.makedirs(result_dir + '%04d'%epoch)

          temp = np.concatenate((gt_patch[0,:,:,:],output[0,:,:,:]),axis=1)
          scipy.misc.toimage(temp*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%04d/%05d_00_train_%d.jpg'%(epoch,train_id,ratio))

    saver.save(sess, checkpoint_dir + 'model.ckpt')
