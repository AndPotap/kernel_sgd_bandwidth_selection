# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================
# Imports
# ===========================================================================
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from Models.kernel_sgd import perform_sgd
from Models.kernel_funcs import get_all_kernel_predictions
from Utils.train_test import get_train_test_ind
# ===========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================
# Generate some data
# ===========================================================================
np.random.seed(seed=21)
obs_n = 100
folds_n = 8
x = np.random.uniform(low=-5, high=5, size=(obs_n, 1))
noise = np.random.normal(loc=0, scale=0.5, size=obs_n)
y = np.sin(np.reshape(a=x, newshape=obs_n)) + noise
# ===========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================
# Estimate kernel regression
# ===========================================================================
train_ind, test_ind = get_train_test_ind(obs_n=obs_n, folds_n=folds_n, train_split=0.8)
chi, loss, gradient = perform_sgd(x=x, y=y, train_ind=train_ind, test_ind=test_ind, print_every=1,
                                  folds_n=folds_n, multiple_regularizers=False, distributed=True,
                                  learning_rate=0.1, max_iters=7, tolerance=1.e-4,
                                  batch_pct=0.7)
bandwidth = tf.math.exp(chi)
# ===========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================
# Plot the data
# ===========================================================================
x_grid = np.linspace(start=-5, stop=5, num=1_000)
y_hat = get_all_kernel_predictions(x_test=tf.constant(value=np.reshape(x_grid, newshape=(1_000, 1)),
                                                      dtype=tf.float32),
                                   y_train=tf.constant(value=y, dtype=tf.float32),
                                   x_train=tf.constant(value=x, dtype=tf.float32),
                                   bandwidth=bandwidth)
y_bad = get_all_kernel_predictions(x_test=tf.constant(value=np.reshape(x_grid, newshape=(1_000, 1)),
                                                      dtype=tf.float32),
                                   y_train=tf.constant(value=y, dtype=tf.float32),
                                   x_train=tf.constant(value=x, dtype=tf.float32),
                                   bandwidth=tf.constant(value=1, dtype=tf.float32))
plt.figure()
plt.title('Bandwidth smoothing via AD')
plt.scatter(x=x, y=y, c='orange', label='data')
plt.plot(x_grid, y_hat.numpy(), c='blue', label='prediction')
plt.plot(x_grid, y_bad.numpy(), c='gray', label='neutral')
plt.plot(x_grid, np.sin(x_grid), c='red', label='actual')
plt.legend()
plt.savefig(fname='./Paper/pics/example01.png')
plt.show()
# ===========================================================================
