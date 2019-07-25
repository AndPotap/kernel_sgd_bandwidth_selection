# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================
# Imports
# ===========================================================================
import numpy as np
import tensorflow as tf
import time
from typing import Tuple
import multiprocessing
from functools import partial
from Utils.train_test import get_train_test_partition
from Models.kernel_funcs import compute_loss
# ===========================================================================


def perform_sgd(x: np.ndarray,
                y: np.ndarray,
                train_ind: dict,
                test_ind: dict,
                folds_n: int,
                batch_pct: float = 1,
                multiple_regularizers: bool = False,
                learning_rate: float = 0.01,
                max_iters: int = 100,
                tolerance: float = 1.e-3,
                verbose: bool = True,
                distributed: bool = False,
                print_every: int = 10) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    iteration, norm = 0, 1
    chi, loss_t0, gradient, learning_rate = initialize_variables(
                                                        learning_rate=learning_rate,
                                                        multiple_regularizers=multiple_regularizers,
                                                        variable_num=x.shape[1])
    pool = multiprocessing.Pool()
    tic = time.time()
    while (iteration < max_iters) and (norm > tolerance):
        t0 = time.time()
        loss_t1, gradient = calculate_loss_and_gradient(chi=chi, x=x, y=y,
                                                        train_ind=train_ind, test_ind=test_ind,
                                                        folds_n=folds_n, batch_pct=batch_pct,
                                                        pool=pool, distributed=distributed)
        chi = chi - learning_rate * gradient
        t1 = time.time()

        iteration += 1
        norm = np.abs((loss_t1.numpy() - loss_t0.numpy()) / (loss_t1.numpy()))
        loss_t0 = loss_t1

        if verbose and (iteration % print_every == 0):
            print(f'ITER {iteration:4d} || LOSS {loss_t0.numpy():2.4e} || '
                  f'NORM {norm: 2.2e} || '
                  f'GRAD {tf.linalg.norm(gradient):2.1e} || TIME {t1-t0:2.1e} sec')

    toc = time.time()
    print(f'TOTAL TIME: {toc - tic:2.1e} sec')
    return chi, loss_t0, gradient


def initialize_variables(learning_rate: float,
                         multiple_regularizers: bool,
                         variable_num: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    if multiple_regularizers:
        chi = tf.constant(value=0, dtype=tf.float32, shape=variable_num)
    else:
        chi = tf.constant(value=0, dtype=tf.float32)
    loss = tf.constant(value=0, dtype=tf.float32)
    gradient = tf.constant(value=0, dtype=tf.float32)
    learning_rate = tf.constant(value=learning_rate, dtype=tf.float32)
    return chi, loss, gradient, learning_rate


def calculate_loss_and_gradient(chi: tf.Tensor,
                                x: np.ndarray,
                                y: np.ndarray,
                                train_ind: dict,
                                test_ind: dict,
                                folds_n: int,
                                distributed: bool,
                                batch_pct: float,
                                pool):
    kwargs = {'chi': chi, 'x': x, 'y': y, 'batch_pct': batch_pct,
              'train_ind': train_ind, 'test_ind': test_ind, 'folds_n': folds_n}
    if distributed:
        loss, gradient = calculate_loss_and_gradient_distributed(**kwargs, pool=pool)
    else:
        loss, gradient = calculate_loss_and_gradient_single(**kwargs)
    return loss, gradient


def calculate_loss_and_gradient_single(chi: tf.Tensor,
                                       x: np.ndarray,
                                       y: np.ndarray,
                                       train_ind: dict,
                                       test_ind: dict,
                                       folds_n: int,
                                       batch_pct: float):
    loss = tf.constant(value=0, dtype=tf.float32)
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(tensor=chi)
        bandwidth = tf.math.exp(chi)
        for current_fold in range(folds_n):
            x_train, y_train, x_test, y_test = get_train_test_partition(x=x, y=y,
                                                                        train_ind=train_ind,
                                                                        test_ind=test_ind,
                                                                        current_fold=current_fold,
                                                                        batch_pct=batch_pct)

            loss += compute_loss(x_test=x_test, y_test=y_test, x_train=x_train, y_train=y_train,
                                 bandwidth=bandwidth, folds_n=folds_n)

    gradient = g.gradient(target=loss, sources=chi)
    return loss, gradient


def calculate_loss_and_gradient_distributed(chi: tf.Tensor,
                                            x: np.ndarray,
                                            y: np.ndarray,
                                            train_ind: dict,
                                            test_ind: dict,
                                            folds_n: int,
                                            batch_pct: float,
                                            pool):
    loss = tf.constant(value=0, dtype=tf.float32)
    gradient = tf.constant(value=0, dtype=tf.float32)

    iterable = [i for i in range(folds_n)]
    f = partial(compute_loss_and_gradient_of_fold,
                chi=chi, x=x,  y=y, train_ind=train_ind, test_ind=test_ind,
                folds_n=folds_n, batch_pct=batch_pct)
    losses_gradients = pool.map(func=f, iterable=iterable)

    loss_location, gradient_location = 0, 1
    for k in range(folds_n):
        loss += losses_gradients[k][loss_location]
        gradient += losses_gradients[k][gradient_location]

    return loss, gradient


def compute_loss_and_gradient_of_fold(current_fold: int,
                                      chi: tf.Tensor,
                                      x: np.ndarray,
                                      y: np.ndarray,
                                      train_ind: dict,
                                      test_ind: dict,
                                      folds_n: int,
                                      batch_pct: float):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(tensor=chi)
        bandwidth = tf.math.exp(chi)
        x_train, y_train, x_test, y_test = get_train_test_partition(x=x, y=y,
                                                                    train_ind=train_ind,
                                                                    test_ind=test_ind,
                                                                    current_fold=current_fold,
                                                                    batch_pct=batch_pct)

        loss = compute_loss(x_test=x_test, y_test=y_test, x_train=x_train, y_train=y_train,
                            bandwidth=bandwidth, folds_n=folds_n)

    gradient = g.gradient(target=loss, sources=chi)
    return loss, gradient
