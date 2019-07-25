import numpy as np
import tensorflow as tf


def compute_loss(x_test: tf.Tensor,
                 y_test: tf.Tensor,
                 x_train: tf.Tensor,
                 y_train: tf.Tensor,
                 bandwidth: tf.Tensor,
                 folds_n: int) -> tf.Tensor:
    test_obs_n = x_test.shape[0]
    loss = tf.constant(value=0, dtype=tf.float32)
    for j in range(test_obs_n):
        y_hat = compute_kernel_prediction(x_test_j=x_test[j, :], y_train=y_train,
                                          x_train=x_train, bandwidth=bandwidth)
        loss += (1 / folds_n) * (y_test[j] - y_hat) ** 2
    return loss


def get_all_kernel_predictions(x_test: tf.Tensor,
                               y_train: tf.Tensor,
                               x_train: tf.Tensor,
                               bandwidth: tf.Tensor) -> tf.Tensor:
    test_obs_n = x_test.shape[0]
    y_hat = np.zeros(shape=test_obs_n)
    for j in range(test_obs_n):
        y_pred = compute_kernel_prediction(x_test_j=x_test[j, :],
                                           y_train=y_train,
                                           x_train=x_train,
                                           bandwidth=bandwidth)
        y_hat[j] = y_pred.numpy()
    return tf.constant(value=y_hat, dtype=tf.float32, shape=y_hat.shape)


def compute_kernel_prediction(x_test_j: tf.Tensor,
                              y_train: tf.Tensor,
                              x_train: tf.Tensor,
                              bandwidth: tf.Tensor) -> tf.Tensor:
    diff = tf.reduce_sum(((x_test_j - x_train) / bandwidth) ** 2, axis=1)
    kernel = tf.math.exp(-0.5 * diff)
    y_hat = tf.reduce_sum(kernel * y_train) / tf.reduce_sum(kernel)
    return y_hat
