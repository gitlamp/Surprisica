
from __future__ import (absolute_import, print_function, unicode_literals, division)

import numpy as np
from collections import defaultdict
from six import iteritems

from .utils import flatten


def rmse(predictions, verbose=True):
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


def mse(predictions, verbose=True):
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


def mae(predictions, verbose=True):
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


def fcp(predictions, verbose=True):
    if not predictions:
        raise ValueError('Prediction list is empty.')

    predictions_u = defaultdict(list)
    nc_u = defaultdict(int)
    nd_u = defaultdict(int)

    for u0, _, r0, est, _ in predictions:
        predictions_u[u0].append((r0, est))

    for u0, preds in iteritems(predictions_u):
        for r0i, esti in preds:
            for r0j, estj in preds:
                if esti > estj and r0i > r0j:
                    nc_u[u0] += 1
                if esti >= estj and r0i < r0j:
                    nd_u[u0] += 1

    nc = np.mean(list(nc_u.values())) if nc_u else 0
    nd = np.mean(list(nd_u.values())) if nd_u else 0

    try:
        fcp = nc / (nc + nd)
    except ZeroDivisionError:
        raise ValueError('cannot compute fcp on this list of prediction. ' +
                         'Does every user have at least two predictions?')

    if verbose:
        print('FCP:  {0:1.4f}'.format(fcp))

    return fcp
