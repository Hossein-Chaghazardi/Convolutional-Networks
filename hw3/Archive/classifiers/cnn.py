import numpy as np

from hw3.layers import *
from hw3.conv_layers import *
from hw3.fast_layers import *
from hw3.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels. In this convnet, the convolutional layer doesn't change the
  image size.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, batch = False):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.batch = batch
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    # Extracting params
    C, H, W = input_dim
    # Since the size doens't change we should set the pad = (filter -1) / 2 and stride = 1

    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))  # Square filters
    self.params['b1'] = np.zeros(num_filters)

    self.params['W2'] = np.random.normal(0, weight_scale, (num_filters*H/2*W/2, hidden_dim))  # affine layer. Note /2 for pooling layer
    self.params['b2'] = np.zeros(hidden_dim)

    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))  # affine
    self.params['b3'] = np.zeros(num_classes)


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    out2, cache2 = affine_relu_forward(out1, W2, b2)
    out3, cache3 = affine_forward(out2, W3, b3)
    scores = out3

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
    #Back prop
    dx3, dw3, db3 = affine_backward(dout, cache3)
    dx2, dw2, db2 = affine_relu_backward(dx3, cache2)
    dx1, dw1, db1 = conv_relu_pool_backward(dx2, cache1)

    grads['W3'] = dw3 + self.reg * W3; grads['b3'] = db3;
    grads['W2'] = dw2 + self.reg * W2; grads['b2'] = db2;
    grads['W1'] = dw1 + self.reg * W1; grads['b1'] = db1;
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class HosseinNet():
  """

  Architecture:

    conv - batch - relu - 2x2 max pool - affine - batch - relu - affine - softmax


  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):

    """

    :param input_dim: of size (channels, height, width)
    :param num_filters: number of filters in the conv layer
    :param filter_size: the size of filter in conv layer (filter_size by filter_size)
    :param hidden_dim: hidden dimensions of FC layer
    :param num_classes: the number of categories
    :param weight_scale: initialization of Weight matrices out of a normal distribution
    :param reg: regularization parameter
    :param dtype: nype of the data
    :param use_batchnorm: the flag for using batch normalization (Spatial for conv and normal for FC)technique
    """

    self.params = {}
    self.reg = reg
    self.bn_params = {}  # Initialize the batch normalization mode, gamma, beta
    self.hidden_dim = hidden_dim
    self.num_filters = num_filters
    self.filter_size = filter_size

    """

    We should initialize the weights, affine values(offsets), gamma and beta for batch normalization.
    For each layer (conv -> pool > affine > affine). We should resize the dimensions. We apply conv layers with stride
    of one and pad = (Filter_size - 1)/2

    """

    """ For first Conv: """
    C, input_height, input_width = input_dim
    stride = 1
    pad = (self.filter_size - 1) / 2
    conv_height = (input_height + 2*pad - self.filter_size) / stride  + 1
    conv_width = (input_width + 2*pad - self.filter_size) / stride  + 1
    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, self.filter_size, self.filter_size))
    self.params['b1'] = np.zeros(num_filters)

    # Batch would not change the dimmension

    """ Pooling layer: """
    pool_size = 2
    pool_stride = 2
    pool_height = conv_height / 2
    pool_width = conv_width / 2

    """ Affine layer 1 """
    self.params['W2'] = np.random.normal(0, weight_scale, (pool_height*pool_width*num_filters, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)

    # batch and relu wouldn't touch the hidden dimensions


    """ Affine layer 2 """
    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)


    # first batch layer:
    self.params['gamma1'] = np.ones(num_filters)
    self.params['beta1'] = np.zeros(num_filters)
    self.bn_params['bn_param1'] = {'mode':'train', 'running_mean': np.zeros(num_filters), 'running_var': np.zeros(num_filters)}

    # Second bath layer
    self.params['gamma2'] = np.ones(hidden_dim)
    self.params['beta2'] = np.zeros(hidden_dim)
    self.bn_params['bn_param2'] = {'mode':'train', 'running_mean': np.zeros(hidden_dim), 'running_var': np.zeros(hidden_dim)}


    # Changing the data type
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """

      Calculate the forward and backward pass of the architecture. If y is none, the scores will be reported

      X: data with shape: (N,C,H,W)
      y: labels shape (N,)

    """

    # Extracting params
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    bn_param1, gamma1, beta1 = self.bn_params['bn_param1'], self.params['gamma1'], self.params['beta1']
    bn_param2, gamma2, beta2 = self.bn_params['bn_param2'], self.params['gamma2'], self.params['beta2']

    conv_param = {'stride':1, 'pad':(self.filter_size - 1)/2}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}


    """

    Forward:

    """

    # First convinience conv-batch-relu-pool

    conv_out, conv_cache = conv_batch_relu_pool_forward(x=X, w=W1, b=b1, conv_param=conv_param, pool_param=pool_param,
                                                        gamma=gamma1, beta=beta1, bn_param=bn_param1)

    # Second: affine-batch-relu

    affine_out1, affine_cache1 = affine_norm_relu_forward(x=conv_out, w=W2, b=b2, gamma=gamma2, beta=beta2, bn_param=bn_param2)

    # Third: affine

    affine_out2, affine_cache2 = affine_forward(affine_out1, w=W3, b=b3)

    # Scores = affine_out2
    scores = affine_out2

    """

    Report the the scores if y is None

    """
    if y is None:
      return scores


    """

    Back propagation:
    We use caches arrived from last section to go back. We use softmax. we add regularization at the last step.

    """
    grads, loss= {}, None
    # Using softmax to derive loss and dout

    loss, dout = softmax_loss(x=scores, y=y)

    dx3, dw3, db3 = affine_backward(dout, affine_cache2)
    dx2, dw2, db2, dgamma2, dbeta2 = affine_norm_relu_backward(dx3, affine_cache1)
    dx, dw1, db1, dgamma1, dbeta1 = conv_batch_relu_pool_backward(dx2, conv_cache)

    # Adding regularization
    loss += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
    dw1 += self.reg * W1
    dw2 += self.reg * W2
    dw3 += self.reg * W3

    # Updating grad dictionary
    grads.update({'W1': dw1,
                  'b1': db1,
                  'W2': dw2,
                  'b2': db2,
                  'W3': dw3,
                  'b3': db3,
                  'beta1': dbeta1,
                  'beta2': dbeta2,
                  'gamma1': dgamma1,
                 'gamma2': dgamma2})

    return loss, grads

