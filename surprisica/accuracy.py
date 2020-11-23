
from __future__ import (absolute_import, print_function, unicode_literals, division)

import numpy as np
from collections import defaultdict
from six import iteritems

from .utils import flatten


def rmse(predictions, verbose=False):
    """Compute RMSE (Root Mean Squared Error).

    Args:
        predictions: A list of predictions returned by :meth:`test()
            <surprisica.prediction_algorithms.algo_base.AlgoBase.test()>`.
        verbose: Whether to print details of the prediction.  Default
            is False."""
    if not predictions:
        raise ValueError('Prediction list is empty.')

    predictions = flatten(predictions)
    mse = []

    for p in predictions:
        true_r = p.r_ui
        est = p.est
        mse.append(float((true_r - est)**2))

    mse = np.mean(mse)
    rmse_ = np.sqrt(mse)

    if verbose:
        print('RMSE: {0:1.4f}'.format(rmse_))

    return rmse_


def mse(predictions, verbose=False):
    """Compute MSE (Mean Squared Error).

    Args:
        predictions: A list of predictions returned by :meth:`test()
            <surprisica.prediction_algorithms.algo_base.AlgoBase.test()>`.
        verbose: Whether to print details of the prediction.  Default
            is False."""
    if not predictions:
        raise ValueError('Prediction list is empty.')

    predictions = flatten(predictions)
    mse_ = []

    for p in predictions:
        true_r = p.r_ui
        est = p.est
        mse_.append(float((true_r - est) ** 2))

    mse_ = np.mean(mse_)

    if verbose:
        print('MSE: {0:1.4f}'.format(mse_))

    return mse_


def mae(predictions, verbose=False):
    """Compute MAE (Mean Absolute Error).

    Args:
        predictions: A list of predictions returned by :meth:`test()
            <surprisica.prediction_algorithms.algo_base.AlgoBase.test()>`.
        verbose: Whether to print details of the prediction.  Default
            is False."""
    if not predictions:
        raise ValueError('Prediction list is empty.')

    predictions = flatten(predictions)
    mae_ = []

    for p in predictions:
        true_r = p.r_ui
        est = p.est
        mae_.append(float(abs(true_r - est)))

    mae_ = np.mean(mae_)

    if verbose:
        print('MAE:  {0:1.4f}'.format(mae_))

    return mae_
