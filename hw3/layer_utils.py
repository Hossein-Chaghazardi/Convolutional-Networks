from hw3.layers import *
from hw3.fast_layers import *
from hw3.conv_layers import *

def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_batch_relu_pool_forward(x, w, b, conv_param, pool_param, bn_param, gamma, beta):
  conv_out, conv_cache = conv_forward_fast(x=x, w=w, b=b, conv_param=conv_param)
  batch_out, batch_cache = spatial_batchnorm_forward(conv_out, gamma=gamma, beta=beta, bn_param=bn_param)
  relu_out, relu_cache = relu_forward(batch_out)
  pool_out, pool_cache = max_pool_forward_fast(x=relu_out, pool_param=pool_param)

  cache = (conv_cache, batch_cache, relu_cache, pool_cache)

  return pool_out, cache


def conv_batch_relu_pool_backward(dout, cache):

  # Extracting cache
  conv_cache, batch_cache, relu_cache, pool_cache = cache

  # back prop
  pool_der = max_pool_backward_fast(dout, pool_cache)
  relu_der = relu_backward(pool_der, relu_cache)
  batch_der, dgamma, dbeta = spatial_batchnorm_backward(relu_der, batch_cache)
  dx, dw, db = conv_backward_fast(batch_der, conv_cache)

  return dx, dw, db, dgamma, dbeta


def affine_norm_relu_forward(x, w, b, gamma, beta, bn_param):
  """ Convinience layer  """

  affine_out, affine_cache = affine_forward(x, w, b)
  batch_out, batch_cache = batchnorm_forward(affine_out, gamma, beta, bn_param)
  relu_out, relu_cache = relu_forward(batch_out)

  cache = (affine_cache, batch_cache, relu_cache)

  return relu_out, cache


def affine_norm_relu_backward(dout, cache):
  """  We perform the backward path for this convinience layer  """

  # Cache Extraction
  affine_cache, batch_cache, relu_cache = cache

  # back prop
  relu_der = relu_backward(dout, relu_cache)
  batch_der, dgamma, dbeta = batchnorm_backward(relu_der, batch_cache)
  dx, dw, db = affine_backward(batch_der, affine_cache)

  return dx, dw, db, dgamma, dbeta











