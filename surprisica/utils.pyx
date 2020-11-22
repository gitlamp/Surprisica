
from __future__ import (absolute_import, print_function, unicode_literals, division)

import numpy as np
cimport numpy as np
import numbers
from six import iteritems


def cosine(n_x, yr, min_support):
    # sum (r_xy * r_x'y) for common ys
    cdef np.ndarray[np.double_t, ndim=2] prods
    # number of common ys
    cdef np.ndarray[np.int_t, ndim=2] freq
    # sum (r_xy ^ 2) for common ys
    cdef np.ndarray[np.double_t, ndim=2] sqi
    # sum (r_x'y ^ 2) for common ys
    cdef np.ndarray[np.double_t, ndim=2] sqj
    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim=2] sim

    cdef int xi, xj
    cdef double ri,rj
    cdef int min_sprt = min_support

    prods = np.zeros((n_x, n_x), np.double)
    freq = np.zeros((n_x, n_x), np.int)
    sqi = np.zeros((n_x, n_x), np.double)
    sqj = np.zeros((n_x, n_x), np.double)
    sim = np.zeros((n_x, n_x), np.double)

    for y, y_ratings in iteritems(yr):
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                freq[xi, xj] += 1
                prods[xi, xj] += ri * rj
                sqi[xi, xj] += ri**2
                sqj[xi, xj] += rj**2

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                sim[xi, xj] = prods[xi, xj] / denum

            sim[xj, xi] = sim[xi, xj]

    return sim


def msd(n_x, yr, min_support):
    # sum (r_xy - r_x'y)**2 for common ys
    cdef np.ndarray[np.double_t, ndim=2] sq_diff
    # number of common ys
    cdef np.ndarray[np.int_t, ndim=2] freq
    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim=2] sim

    cdef int xi, xj
    cdef double ri, rj
    cdef int min_sprt = min_support

    sq_diff = np.zeros((n_x, n_x), np.double)
    freq = np.zeros((n_x, n_x), np.int)
    sim = np.zeros((n_x, n_x), np.double)

    for y, y_ratings in iteritems(yr):
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                sq_diff[xi, xj] += (ri - rj)**2
                freq[xi, xj] += 1

    for xi in range(n_x):
        sim[xi, xi] = 1  # completely arbitrary and useless anyway
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                # return inverse of (msd + 1) (+ 1 to avoid dividing by zero)
                sim[xi, xj] = 1 / (sq_diff[xi, xj] / freq[xi, xj] + 1)

            sim[xj, xi] = sim[xi, xj]

    return sim


def asymmetric_cosine(n_x, yr, xr, min_support):
    # sum (r_xy * r_x'y) for common ys
    cdef np.ndarray[np.double_t, ndim=2] prods
    # number of common ys
    cdef np.ndarray[np.int_t, ndim=2] freq
    # sum (r_xy ^ 2) for common ys
    cdef np.ndarray[np.double_t, ndim=2] sqi
    # sum (r_x'y ^ 2) for common ys
    cdef np.ndarray[np.double_t, ndim=2] sqj
    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim=2] sim

    cdef int xi, xj
    cdef double ri, rj
    cdef int min_sprt = min_support

    prods = np.zeros((n_x, n_x), np.double)
    freq = np.zeros((n_x, n_x), np.int)
    sqi = np.zeros((n_x, n_x), np.double)
    sqj = np.zeros((n_x, n_x), np.double)
    sim = np.zeros((n_x, n_x), np.double)

    for y, y_ratings in iteritems(yr):
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                freq[xi, xj] += 1
                prods[xi, xj] += ri * rj
                sqi[xi, xj] += ri**2
                sqj[xi, xj] += rj**2

    for xi in range(n_x):
        for xj in range(n_x):
            if xi == xj:
                sim[xi, xj] = 1
            elif freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                cos = prods[xi, xj] / denum
                asym_coeff = freq[xi, xj] / len(xr[xi])
                sorensen_dice_coeff = (2 * freq[xi, xj]) / (len(xr[xi]) + len(xr[xj]))
                sim[xi, xj] = cos * asym_coeff * sorensen_dice_coeff

    return sim


def asymmetric_msd(n_x, yr, xr, min_support, L=16):
    # sum (r_xy - r_x'y)**2 for common ys
    cdef np.ndarray[np.double_t, ndim=2] sq_diff
    # number of common ys
    cdef np.ndarray[np.int_t, ndim=2] freq
    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim=2] sim

    cdef int xi, xj
    cdef double ri, rj
    cdef int min_sprt = min_support

    sq_diff = np.zeros((n_x, n_x), np.double)
    freq = np.zeros((n_x, n_x), np.int)
    sim = np.zeros((n_x, n_x), np.double)

    for y, y_ratings in iteritems(yr):
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                sq_diff[xi, xj] += (ri - rj)**2
                freq[xi, xj] += 1

    for xi in range(n_x):
        sim[xi, xi] = 1  # completely arbitrary and useless anyway
        for xj in range(n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                # return inverse of (msd + 1) (+ 1 to avoid dividing by zero)
                msd = 1 / (sq_diff[xi, xj] / freq[xi, xj] + 1)
                msd = (L - msd) / L
                asym_coeff = freq[xi, xj] / len(xr[xi])
                sorensen_dice_coeff = (2 * freq[xi, xj]) / (len(xr[xi]) + len(xr[xj]))
                sim[xi, xj] = msd * asym_coeff * sorensen_dice_coeff

    return sim


def sorensen_idf(n_x, yv, n_y):
    # the visiting matrix
    cdef np.double_t[:] visit
    # the inverse item freq for x
    cdef np.ndarray[np.double_t, ndim=2] xiif
    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim=2] sim

    cdef int xi, xj

    visit = np.zeros(n_y, np.double)
    xiif = np.zeros((n_x, n_y), np.double)
    sim = np.zeros((n_x, n_x), np.double)
    cal_iif = np.vectorize(lambda x: np.log10(np.sum(visit) / x)
                           if x > 0 else 0)

    for y, y_ratings in iteritems(yv):
        for _, v in y_ratings:
            visit[y] += v

    iif = cal_iif(visit)

    for y, y_ratings in iteritems(yv):
        for xi, v in y_ratings:
            xiif[xi, y] += iif[y]

    for y, y_ratings in iteritems(yv):
        for xi, _ in y_ratings:
            for xj, _ in y_ratings:
                if xi == xj:
                    sim[xi, xj] = 1
                else:
                    sim[xi, xj] += 2 * (iif[y]) / (xiif[xi].sum() + xiif[xj].sum())

    return sim


def usr_influence_cos(n_x, yr, n_y, xr, min_support):
    # sum (r_xy * r_x'y) for common ys
    cdef np.ndarray[np.double_t, ndim=2] prods
    # number of common ys
    cdef np.ndarray[np.int_t, ndim=2] freq
    # sum (r_xy ^ 2) for common ys
    cdef np.ndarray[np.double_t, ndim=2] sqi
    # sum (r_x'y ^ 2) for common ys
    cdef np.ndarray[np.double_t, ndim=2] sqj
    # prime rates
    cdef np.ndarray[np.double_t, ndim=2] rprime
    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim=2] sim

    cdef int x, j, xi, xj
    cdef double r, ri,rj
    cdef int min_sprt = min_support

    prods = np.zeros((n_x, n_x), np.double)
    freq = np.zeros((n_x, n_x), np.int)
    sqi = np.zeros((n_x, n_x), np.double)
    sqj = np.zeros((n_x, n_x), np.double)
    rprime = np.zeros((n_x, n_y), np.double)
    sim = np.zeros((n_x, n_x), np.double)

    for x, x_ratings in iteritems(xr):
        avg = sum(r for _, r in x_ratings) / len(x_ratings)
        for j, r in x_ratings:
            if r >= avg:
                rprime[x, j] = 1
            else:
                rprime[x, j] = 0
    
    for y, y_ratings in iteritems(yr):
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                freq[xi, xj] += 1
                prods[xi, xj] += ri * rj
                sqi[xi, xj] += ri**2
                sqj[xi, xj] += rj**2

    for xi in range(n_x):
        for xj in range(n_x):
            if xi == xj:
                sim[xi, xj] = 1
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                cos = prods[xi, xj] / denum
                asym_coeff = freq[xi, xj] / len(xr[xi])
                sorensen_dice_coeff = (2 * freq[xi, xj]) / (len(xr[xi]) + len(xr[xj]))
                usrinf_coeff = sum(rprime[xi] * rprime[xj]) / sum(rprime[xj])
                sim[xi, xj] = cos * asym_coeff * sorensen_dice_coeff * usrinf_coeff

    return sim


def get_rng(random_state):
    """Return a 'validated' RNG.

    If random_state is None, use RandomState singleton from numpy.  Else if
    it's an integer, consider it's a seed and initialized an rng with that
    seed. If it's already an rng, return it."""
    if random_state is None:
        return np.random.mtrand._rand
    elif isinstance(random_state, (numbers.Integral, np.integer)):
        return np.random.RandomState(random_state)
    if isinstance(random_state, np.random.RandomState):
        return random_state
    raise ValueError('Wrong random state. Expecting None, an int or a numpy '
                     'RandomState instance, got a '
                     '{}'.format(type(random_state)))


def flatten(container):
    """Flatten unstructured list."""
    for i in container:
        if isinstance(i, list):
            for j in flatten(i):
                yield j
        else:
            yield i
