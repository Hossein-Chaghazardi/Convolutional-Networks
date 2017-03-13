# As usual, a bit of setup

import numpy as np
import matplotlib.pyplot as plt
from hw3.data_utils import get_CIFAR10_data
from hw3.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from hw3.conv_layers import *
from hw3.solver import Solver
from hw3.classifiers.cnn import *

data = get_CIFAR10_data()
for k, v in data.iteritems():
  print '%s: ' % k, v.shape

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

######################


from hw3.layer_utils import conv_batch_relu_pool_forward, conv_batch_relu_pool_backward


C=3
""" For FORWARD
x = np.random.randn(2, 3, 16, 16)
w = np.random.randn(3, 3, 3, 3)
b = np.random.randn(3,)
gamma = np.ones(3)
beta = np.ones(3)
bn_param = {'mode': 'train',
            'running_mean': np.zeros(C),
            'running_var': np.zeros(C)}

dout = np.random.randn(2, 3, 8, 8)
conv_param = {'stride': 1, 'pad': 1}
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

out, cache = conv_batch_relu_pool_forward(x=x, w=w, b=b, conv_param=conv_param, pool_param=pool_param, gamma=gamma, beta=beta, bn_param=bn_param)
dx, dw, db, dgamma, dbeta = conv_batch_relu_pool_backward(dout, cache)

dx_num = eval_numerical_gradient_array(lambda x: conv_batch_relu_pool_forward(x, w, b, conv_param=conv_param, pool_param=pool_param, gamma=gamma, beta=beta, bn_param=bn_param)[0], x=x, df=dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_batch_relu_pool_forward(x, w, b, conv_param=conv_param, pool_param=pool_param, gamma=gamma, beta=beta, bn_param=bn_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_batch_relu_pool_forward(x, w, b, conv_param=conv_param, pool_param=pool_param, gamma=gamma, beta=beta, bn_param=bn_param)[0], b, dout)
dgamma_num = eval_numerical_gradient_array(lambda gamma:conv_batch_relu_pool_forward(x, w, b, conv_param=conv_param, pool_param=pool_param, gamma=gamma, beta=beta, bn_param=bn_param)[0], gamma, dout)
dbeta_num = eval_numerical_gradient_array(lambda beta:conv_batch_relu_pool_forward(x, w, b, conv_param=conv_param, pool_param=pool_param, gamma=gamma, beta=beta, bn_param=bn_param)[0], beta, dout)


print 'Testing conv_relu_pool'
print 'dx error: ', rel_error(dx_num, dx)
print 'dw error: ', rel_error(dw_num, dw)
print 'db error: ', rel_error(db_num, db)
print 'dbeta error: ', rel_error(dbeta_num, dbeta)
print 'dgamma error: ', rel_error(dgamma_num, dgamma)"""

num_train = 100
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

model = HosseinNet(weight_scale=1e-2)

solver = Solver(model, small_data,
                num_epochs=20, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-4,
                },
                verbose=True, print_every=1)
solver.train()