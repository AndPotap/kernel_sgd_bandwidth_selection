# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================
# Imports
# ===========================================================================
import numpy as np
import tensorflow as tf
from Models.kernel_funcs import compute_loss
# ===========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================
# Generate the data
# ===========================================================================
x_train = np.array([[0.4, 2.3, -0.6, 0.3],
                    [0.3, 1.0, -1.0, -1.5],
                    [1.2, -1.4, 0.7, 0.7],
                    [-0.5, -0.8, 0.7, 0.6],
                    [1.0, 0.9, -0.3, -0.4],
                    [-0.3, -0.8, -3.4, -0.5],
                    [-1.5, 0.1, -0.4, -0.8],
                    [-1.0, -0.8, 1.1, -1.0],
                    [0.2, 1.0, 0.1, -2.0],
                    [-1.9, -0.9, -1.9, 0.6]])
x_train = tf.constant(value=x_train, dtype=tf.float32, shape=x_train.shape)

y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = tf.constant(value=y_train, dtype=tf.float32, shape=y_train.shape)

x_test = np.array([[-2.1, -1.1, 0.0, -0.2],
                   [-1.6, 2.0, -1.1, 0.2],
                   [-0.5, 0.7, 0.9, 0.0]])
x_test = tf.constant(value=x_test, dtype=tf.float32, shape=x_test.shape)

y_test = np.array([-1, -2, -3])
y_test = tf.constant(value=y_test, dtype=tf.float32, shape=y_test.shape)
# ===========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================
# Get Gradient
# ===========================================================================
chi = tf.constant(value=np.array([0, 0, 0, 0]), dtype=tf.float32)
# chi = tf.constant(value=0.0, dtype=tf.float32)  # works with only one bandwidth as well
with tf.GradientTape(watch_accessed_variables=False) as g:
    g.watch(chi)
    bandwidth = tf.math.exp(chi)
    loss = compute_loss(x_test=x_test,
                        y_test=y_test,
                        x_train=x_train,
                        y_train=y_train,
                        bandwidth=bandwidth,
                        folds_n=1)

gradient = g.gradient(target=loss, sources=chi)
print(gradient.numpy())
# ===========================================================================
