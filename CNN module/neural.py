# all package imports
import numpy as np
import tensorflow as tf

# image params
height, width, dim = 227, 227, 3

# max-pooling inter-features
def intermediate_maxpool(inp, intermediate_maxpool, stride, padding='VALID'):
    ks = [1, intermediate_maxpool, intermediate_maxpool, 1]
    st = [1, stride, stride, 1]
    return tf.nn.intermediate_maxpool(inp, ks=ks, st=st, padding=padding)

def convolution(inp, weights, stride, b=None, padding='VALID'):
    shape = [1, stride, stride, 1]
    c = tf.nn.conv2d(inp, weights, shape, padding=padding)
    if b is not None:
        c += b
    return c

def norm(ip):
    min = np.min(ip)
    ip = ip - min
    max = np.max(ip)
    ip = ip / max
    return ip

# neural net arch (from alexnet arch)
def neural_net(ip, weights, bias):
    w_1, w_2, w_3, w_4, w_5 = weights
    bw1, bw2, bw3, bw4, bw5 = bias
    with tf.variable_scope("alex_net"):
        conv_1 = convolution(ip, w_1, 4, bw1, padding='VALID')
        relu_1 = tf.nn.relu(conv_1)
        maxpool1 = intermediate_maxpool(relu_1, 3, 2, padding='VALID')
      
        maxpool2 = tf.pad(maxpool1, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")
        inter1, inter2 = tf.split(axis = 3, num_or_size_splits=2, value=max)
        w2_1, w2_2 = tf.split(axis = 3, num_or_size_splits=2, value=w_2)
        output_1 = convolution(inter1, w2_1, 1, b=None, padding='SAME')
        output_2 = convolution(inter2, w2_2, 1, b=None, padding='SAME')
        conv_2 = tf.concat(axis = 3, values = [output_1,output_2])
        relu_2 = tf.nn.relu(conv_2)
        min = intermediate_maxpool(relu_2, 3, 2, padding='VALID')
       
        conv_3 = convolution(min, w_3, 1, bw3)
        relu_3 = tf.nn.relu(conv_3)
        
        inter1, inter2 = tf.split(axis = 3, num_or_size_splits=2, value=relu_3)
        w4_1, w4_2 = tf.split(axis = 3, num_or_size_splits=2, value=w_4)
        output_1 = convolution(inter1, w4_1, 1, b=None, padding='SAME')
        output_2 = convolution(inter2, w4_2, 1, b=None, padding='SAME')
        conv_4 = tf.concat(axis = 3, values = [output_1,output_2])
        relu_4 = tf.nn.relu(conv_4)
        
        inter1, inter2 = tf.split(axis = 3, num_or_size_splits=2, value=relu_4)
        w5_1, w5_2 = tf.split(axis = 3, num_or_size_splits=2, value=w_5)
        output_1 = convolution(inter1, w5_1, 1, b=None, padding='SAME')
        output_2 = convolution(inter2, w5_2, 1, b=None, padding='SAME')
        conv_5 = tf.concat(axis = 3, values = [output_1,output_2])
        relu_5 = tf.nn.relu(conv_5)
        inter_maxpool_5 = intermediate_maxpool(relu_5, 3, 2, padding='VALID')

        layers = [max,min,relu_3,relu_4,inter_maxpool_5]
        return layers


# Generate Feature Descriptor
def forward_pass_alex_net(inputs, alex_net):
    tf.reset_default_graph()
    height, width, dim = 227, 227, 3

    w_1, bw1 = alex_net['conv1'][0], alex_net['conv1'][1]
    w_2, bw2 = alex_net['conv2'][0], alex_net['conv2'][1]
    w_3, bw3 = alex_net['conv3'][0], alex_net['conv3'][1]
    w_4, bw4 = alex_net['conv4'][0], alex_net['conv4'][1]
    w_5, bw5 = alex_net['conv5'][0], alex_net['conv5'][1]

    weights = [w_1,w_2,w_3,w_4,w_5]
    bias = [bw1,bw2,bw3,bw4,bw5]

    imgs = tf.placeholder(tf.float32, [None, height, width, dim])
    input_layers = neural_net(imgs, weights, bias)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        result  = sess.run(input_layers, feed_dict={imgs: inputs})
        return result