# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================
# Imports
# ===========================================================================
import numpy as np
import tensorflow as tf
from typing import Tuple
# ===========================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================
# Define the functions
# ===========================================================================


def get_train_test_ind(obs_n: int, folds_n: int, train_split: float = 0.8) -> Tuple[dict, dict]:
    total_ind = set(range(obs_n))
    train_ind = {k: np.random.choice(a=obs_n, size=int(train_split * obs_n)) for k in
                 range(folds_n)}
    test_ind = {k: np.array(list(total_ind.difference(set(train_ind[k])))) for k in range(folds_n)}
    return train_ind, test_ind


def get_train_test_partition(x: np.ndarray,
                             y: np.ndarray,
                             train_ind: dict, test_ind: dict,
                             current_fold: int,
                             batch_pct: float = 1) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    rows_in_batch = get_rows_in_batch(train_ind=train_ind, current_fold=current_fold,
                                      batch_pct=batch_pct)

    x_train = tf.constant(value=x[rows_in_batch, :], dtype=tf.float32)
    y_train = tf.constant(value=y[rows_in_batch], dtype=tf.float32)
    x_test = tf.constant(value=x[test_ind[current_fold], :], dtype=tf.float32)
    y_test = tf.constant(value=y[test_ind[current_fold]], dtype=tf.float32)

    return x_train, y_train, x_test, y_test


def get_rows_in_batch(train_ind: dict, current_fold: int, batch_pct: float) -> np.ndarray:
    train_obs_n = train_ind[current_fold].shape[0]
    batch_obs_n = int(batch_pct * train_obs_n)
    inds_in_batch = np.random.choice(a=train_obs_n, size=batch_obs_n, replace=False)
    rows_in_batch = train_ind[current_fold][inds_in_batch]
    return rows_in_batch

# ===========================================================================
