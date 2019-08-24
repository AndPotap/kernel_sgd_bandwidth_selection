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
np.set_printoptions(precision=2, formatter={'all': lambda options: '%4.2f' % options})
# ===========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================
# Generate some data
# ===========================================================================
np.random.seed(seed=21)
obs_n = 250
folds_n = 8
x = np.random.uniform(low=0, high=10, size=(obs_n, 2))
noise = np.random.normal(loc=0, scale=0.5, size=obs_n)
y = x[:, 0] - x[:, 1] ** 2 + 0.3 * np.log(1 + x[:, 0] * x[:, 1]) + noise
# ===========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================
# Estimate kernel regression
# ===========================================================================
train_ind, test_ind = get_train_test_ind(obs_n=obs_n, folds_n=folds_n, train_split=0.8)
chi_mult, loss, gradient = perform_sgd(x=x, y=y, train_ind=train_ind, test_ind=test_ind,
                                       print_every=1, distributed=True,
                                       folds_n=folds_n, multiple_regularizers=True,
                                       learning_rate=0.0001, max_iters=20, tolerance=1.e-4,
                                       batch_pct=1)
bandwidth_mult = tf.math.exp(chi_mult)
print(bandwidth_mult.numpy())

train_ind, test_ind = get_train_test_ind(obs_n=obs_n, folds_n=folds_n, train_split=0.8)
chi, *_ = perform_sgd(x=x, y=y, train_ind=train_ind, test_ind=test_ind,
                      print_every=1, distributed=False,
                      folds_n=folds_n, multiple_regularizers=False,
                      learning_rate=0.0001, max_iters=20, tolerance=1.e-4,
                      batch_pct=1)
bandwidth = tf.math.exp(chi)
print(bandwidth.numpy())

# ===========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================
# Plot the data
# ===========================================================================
x_grid = np.linspace(start=0, stop=10, num=1_000)
y_hat = get_all_kernel_predictions(x_test=tf.constant(value=np.reshape(x_grid, newshape=(1_000, 1)),
                                                      dtype=tf.float32),
                                   y_train=tf.constant(value=y, dtype=tf.float32),
                                   x_train=tf.constant(value=x, dtype=tf.float32),
                                   bandwidth=bandwidth_mult)
y_bad = get_all_kernel_predictions(x_test=tf.constant(value=np.reshape(x_grid, newshape=(1_000, 1)),
                                                      dtype=tf.float32),
                                   y_train=tf.constant(value=y, dtype=tf.float32),
                                   x_train=tf.constant(value=x, dtype=tf.float32),
                                   bandwidth=tf.constant(value=bandwidth, dtype=tf.float32))

plt.figure()
plt.title('Kernel - Single vs Multiple Bandwidths')
plt.scatter(x=x[:, 1], y=y, c='orange', label='data')
plt.plot(x_grid, y_hat.numpy(), c='blue', label='multiple bandwidths')
plt.plot(x_grid, y_bad.numpy(), c='gray', label='single bandwidth')
plt.legend()
plt.savefig(fname='./Paper/pics/example02.png')
plt.show()
# ===========================================================================
