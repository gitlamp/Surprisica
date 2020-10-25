
from __future__ import (absolute_import, print_function, unicode_literals, division)

cimport numpy as np
import numpy as np
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
