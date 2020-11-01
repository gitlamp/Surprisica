
from __future__ import (absolute_import, print_function, unicode_literals, division)

import pyximport; pyximport.install()

import numpy as np
from abc import ABCMeta, abstractmethod
from itertools import product
from joblib import Parallel
from joblib import delayed
from six import moves, string_types, with_metaclass

from .split import get_cv
from .validation import fit_and_score
from ..dataset import DatasetUserFolds
from ..utils import get_rng


class BaseSearchCV(with_metaclass(ABCMeta)):
    @abstractmethod
    def __init__(self, algo_class, measures=['rmse', 'mae'], cv=None,
                 refit=False, return_train_measures=False, n_jobs=1,
                 pre_dispatch='2*n_jobs', joblib_verbose=0):

        self.algo_class = algo_class
        self.measures = [measure.lower() for measure in measures]
        self.cv = cv

        if isinstance(refit, string_types):
            if refit.lower() not in self.measures:
                raise ValueError('It looks like the measure you want to use '
                                 'with refit ({}) is not in the measures '
                                 'parameter')

            self.refit = refit.lower()

        elif refit is True:
            self.refit = self.measures[0]

        else:
            self.refit = False

        self.return_train_measures = return_train_measures
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.joblib_verbose = joblib_verbose

    def _parse_options(self, params):
        if 'sim_options' in params:
            sim_options = params['sim_options']
            sim_options_list = [dict(zip(sim_options, v)) for v in
                                product(*sim_options.values())]
            params['sim_options'] = sim_options_list

        if 'bsl_options' in params:
            bsl_options = params['bsl_options']
            bsl_options_list = [dict(zip(bsl_options, v)) for v in
                                product(*bsl_options.values())]
            params['bsl_options'] = bsl_options_list

        return params

    def fit(self, data):
        if self.refit and isinstance(data, DatasetUserFolds):
            raise ValueError('refit cannot be used when data has been '
                             'loaded with load_from_folds().')

        cv = get_cv(self.cv)

        delayed_list = (
            delayed(fit_and_score)(self.algo_class(**params), trainset,
                                   testset, self.measures,
                                   self.return_train_measures)
            for params, (trainset, testset) in product(self.param_combinations,
                                                       cv.split(data))
        )
        out = Parallel(n_jobs=self.n_jobs,
                       pre_dispatch=self.pre_dispatch,
                       verbose=self.joblib_verbose)(delayed_list)

        (test_measures_dicts,
         train_measures_dicts,
         fit_times,
         test_times) = zip(*out)

        test_measures = dict()
        train_measures = dict()
        new_shape = (len(self.param_combinations), cv.get_n_folds())
        for m in self.measures:
            test_measures[m] = np.asarray([d[m] for d in test_measures_dicts])
            test_measures[m] = test_measures[m].reshape(new_shape)
            if self.return_train_measures:
                train_measures[m] = np.asarray([d[m] for d in
                                                train_measures_dicts])
                train_measures[m] = train_measures[m].reshape(new_shape)

        cv_results = dict()
        best_index = dict()
        best_params = dict()
        best_score = dict()
        best_estimator = dict()
        for m in self.measures:
            # cv_results: set measures for each split and each param comb
            for split in range(cv.get_n_folds()):
                cv_results['split{0}_test_{1}'.format(split, m)] = \
                    test_measures[m][:, split]
                if self.return_train_measures:
                    cv_results['split{0}_train_{1}'.format(split, m)] = \
                        train_measures[m][:, split]

            # cv_results: set mean and std over all splits (testset and
            # trainset) for each param comb
            mean_test_measures = test_measures[m].mean(axis=1)
            cv_results['mean_test_{}'.format(m)] = mean_test_measures
            cv_results['std_test_{}'.format(m)] = test_measures[m].std(axis=1)
            if self.return_train_measures:
                mean_train_measures = train_measures[m].mean(axis=1)
                cv_results['mean_train_{}'.format(m)] = mean_train_measures
                cv_results['std_train_{}'.format(m)] = \
                    train_measures[m].std(axis=1)

            # cv_results: set rank of each param comb
            indices = cv_results['mean_test_{}'.format(m)].argsort()
            cv_results['rank_test_{}'.format(m)] = np.empty_like(indices)
            cv_results['rank_test_{}'.format(m)][indices] = np.arange(
                len(indices)) + 1  # sklearn starts rankings at 1 as well.

            # set best_index, and best_xxxx attributes
            if m in ('mae', 'rmse', 'mse'):
                best_index[m] = mean_test_measures.argmin()
            elif m in ('fcp',):
                best_index[m] = mean_test_measures.argmax()
            best_params[m] = self.param_combinations[best_index[m]]
            best_score[m] = mean_test_measures[best_index[m]]
            best_estimator[m] = self.algo_class(**best_params[m])

        # Cv results: set fit and train times (mean, std)
        fit_times = np.array(fit_times).reshape(new_shape)
        test_times = np.array(test_times).reshape(new_shape)
        for s, times in zip(('fit', 'test'), (fit_times, test_times)):
            cv_results['mean_{}_time'.format(s)] = times.mean(axis=1)
            cv_results['std_{}_time'.format(s)] = times.std(axis=1)

        # cv_results: set params key and each param_* values
        cv_results['params'] = self.param_combinations
        for param in self.param_combinations[0]:
            cv_results['param_' + param] = [comb[param] for comb in
                                            self.param_combinations]

        if self.refit:
            best_estimator[self.refit].fit(data.build_full_trainset())

        self.best_index = best_index
        self.best_params = best_params
        self.best_score = best_score
        self.best_estimator = best_estimator
        self.cv_results = cv_results

    def test(self, testset, verbose=False):
        if not self.refit:
            raise ValueError('refit is False, cannot use test()')

        return self.best_estimator[self.refit].test(testset, verbose)

    def predict(self, *args):
        if not self.refit:
            raise ValueError('refit is False, cannot use predict()')

        return self.best_estimator[self.refit].predict(*args)


class GridSearchCV(BaseSearchCV):
    def __init__(self, algo_class, param_grid, measures=['rmse', 'mae'],
                 cv=None, refit=False, return_train_measures=False, n_jobs=1,
                 pre_dispatch='2*n_jobs', joblib_verbose=0):

        super(GridSearchCV, self).__init__(
            algo_class=algo_class, measures=measures, cv=cv, refit=refit,
            return_train_measures=return_train_measures, n_jobs=n_jobs,
            pre_dispatch=pre_dispatch, joblib_verbose=joblib_verbose)

        self.param_grid = self._parse_options(param_grid.copy())
        self.param_combinations = [dict(zip(self.param_grid, v)) for v in
                                   product(*self.param_grid.values())]


class RandomizedSearchCV(BaseSearchCV):
    def __init__(self, algo_class, param_distributions, n_iter=10,
                 measures=['rmse', 'mae'], cv=None, refit=False,
                 return_train_measures=False, n_jobs=1,
                 pre_dispatch='2*n_jobs', random_state=None, joblib_verbose=0):

        super(RandomizedSearchCV, self).__init__(
            algo_class=algo_class, measures=measures, cv=cv, refit=refit,
            return_train_measures=return_train_measures, n_jobs=n_jobs,
            pre_dispatch=pre_dispatch, joblib_verbose=joblib_verbose)

        self.n_iter = n_iter
        self.random_state = random_state
        self.param_distributions = self._parse_options(
            param_distributions.copy())
        self.param_combinations = self._sample_parameters(
            self.param_distributions, self.n_iter, self.random_state)

    @staticmethod
    def _sample_parameters(param_distributions, n_iter, random_state=None):
        # check if all distributions are given as lists
        # if so, sample without replacement
        all_lists = np.all([not hasattr(v, 'rvs')
                            for v in param_distributions.values()])
        rnd = get_rng(random_state)

        # sort for reproducibility
        items = sorted(param_distributions.items())

        if all_lists:
            # create exhaustive combinations
            param_grid = [dict(zip(param_distributions, v)) for v in
                          product(*param_distributions.values())]
            combos = np.random.choice(param_grid, n_iter, replace=False)

        else:
            combos = []
            for _ in moves.range(n_iter):
                params = dict()
                for k, v in items:
                    if hasattr(v, 'rvs'):
                        params[k] = v.rvs(random_state=rnd)
                    else:
                        params[k] = v[rnd.randint(len(v))]
                combos.append(params)

        return combos
