
from __future__ import (absolute_import, print_function, unicode_literals, division)

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import minmax_scale
from surprise.dataset import Dataset

from .trainset import Trainset


class Dataset(Dataset):

    def __init__(self, reader):
        super(Dataset, self).__init__(reader)
        self.context = reader.context

    @classmethod
    def load_from_df(cls, df, reader):
        ratings = cls.create_profile(df=df, reader=reader)

        return DatasetAutoFolds(df=ratings, reader=reader)

    @classmethod
    def load_from_file(cls, path, reader):
        ratings = cls.read_ratings(cls, file_name=path, reader=reader)

        return DatasetAutoFolds(df=ratings, reader=reader)

    @classmethod
    def create_profile(cls, df, reader):
        if reader.context:
            ratings = cls.create_rating_profile(df, reader)
            contexts = cls.create_context_profile(df, reader)
            profile = ratings.merge(contexts)

        else:
            profile = cls.create_rating_profile(df, reader)

        return profile

    @staticmethod
    def create_rating_profile(df, reader):
        col = df.columns.to_list()
        uid = col[0]
        iid = col[1]
        t = col[2]
        r = df.groupby([uid, iid])[t].count().reset_index(name='rating')
        r[['rating']] = minmax_scale(r[['rating']], feature_range=reader.rating_scale)

        return r

    @staticmethod
    def create_context_profile(df, reader):
        col = df.columns.to_list()
        col.remove(col[2])
        lid = col[1]
        cnx = col[2:]
        df = df.filter(col)
        df = df.drop_duplicates()
        # Check context exists
        for c in reader.cnx_entities:
            if df[c].dtype != int:
                raise ValueError('Column "{0}" should have integer type.'.format(c))

        cnx_loc = df.groupby([lid] + cnx)[lid].count()
        all_loc = df.groupby(cnx)[lid].count()
        usr_loc = df.groupby([lid])[lid].count()
        # Term frequency
        tf = cnx_loc.div(all_loc).to_frame(name='tf').reset_index()
        # Inverse document frequency
        idf = np.log10(usr_loc.apply(lambda x: usr_loc.sum() / x)).to_frame(name='idf').reset_index()
        res = tf.merge(idf)
        res['cnx_weight'] = res['tf'] * res['idf']
        res = res.drop(['tf', 'idf'], axis=1)
        res = df.merge(res)

        return res

    def construct_trainset(self, raw_trainset):
        if not self.context:
            return super(Dataset, self).construct_trainset(raw_trainset)

        else:
            raw2inner_id_users = {}
            raw2inner_id_items = {}
            raw2inner_id_contexts = {}

            current_u_index = 0
            current_i_index = 0
            current_c_index = 0

            ur = defaultdict(list)
            ir = defaultdict(list)
            uc = defaultdict(list)
            ic = defaultdict(list)

            for ruid, riid, r, all_c, all_cw in raw_trainset:
                try:
                    uid = raw2inner_id_users[ruid]
                except KeyError:
                    uid = current_u_index
                    raw2inner_id_users[ruid] = current_u_index
                    current_u_index += 1
                try:
                    iid = raw2inner_id_items[riid]
                except KeyError:
                    iid = current_i_index
                    raw2inner_id_items[riid] = current_i_index
                    current_i_index += 1

                ur[uid].append((iid, r))
                ir[iid].append((uid, r))

                for i in range(len(all_c)):
                    rcid = all_c[i]
                    cw = all_cw[i]
                    try:
                        cid = raw2inner_id_contexts[rcid]
                    except KeyError:
                        cid = current_c_index
                        raw2inner_id_contexts[rcid] = current_c_index
                        current_c_index += 1

                    if (iid, cid) not in uc[uid]:
                        uc[uid].append((iid, cid))
                    if (cid, cw) not in ic[iid]:
                        ic[iid].append((cid, cw))

                    uc[uid].sort()
                    ic[iid].sort()

            n_users = len(ur)
            n_items = len(ir)
            n_ratings = len(raw_trainset)
            n_contexts = len(raw2inner_id_contexts)

            trainset = Trainset(ur,
                                ir,
                                uc,
                                ic,
                                n_users,
                                n_items,
                                n_ratings,
                                n_contexts,
                                self.reader.rating_scale,
                                raw2inner_id_users,
                                raw2inner_id_items,
                                raw2inner_id_contexts)
            return trainset

    def read_ratings(self, file_name, reader):
        ratings = []
        try:
            raw_ratings = pd.read_csv(file_name,
                                      sep=reader.sep,
                                      encoding='utf-8',
                                      usecols=reader.entities)
            raw_ratings = raw_ratings[reader.entities]
            ratings = self.create_profile(raw_ratings, reader)
        except Exception as err:
            print(err)

        return ratings

    def construct_testset(self, raw_testset):
        if not self.context:
            return super(Dataset, self).construct_testset(raw_testset)
        else:
            return [(ruid, riid, r_ui_trans, rcid)
                    for (ruid, riid, r_ui_trans, rcid, _) in raw_testset]


class DatasetUserFolds(Dataset):
    def __init__(self, folds_files=None, reader=None):

        Dataset.__init__(self, reader)
        self.folds_files = folds_files

        # check that all files actually exist.
        for train_test_files in self.folds_files:
            for f in train_test_files:
                if not os.path.isfile(os.path.expanduser(f)):
                    raise ValueError('File ' + str(f) + ' does not exist.')


class DatasetAutoFolds(Dataset):
    def __init__(self, df=None, reader=None):
        super(DatasetAutoFolds, self).__init__(reader=reader)

        if df is not None:
            self.df = df
        else:
            raise ValueError('Must specify rating dataframe.')

        if reader.context:
            self.raw_ratings = []
            cols = self.df.columns.to_list()

            for i, group in self.df.groupby(cols[:3]):
                uid = i[0]
                iid = i[1]
                r = i[2]
                all_c = []
                all_cw = []
                for t in group.itertuples(index=False, name=None):
                    cnum = len(reader.cnx_entities)
                    all_c.append((t[3:3+cnum]))
                    all_cw.append(float(t[-1]))

                self.raw_ratings.append((uid, iid, float(r), all_c, all_cw))

        else:
            self.raw_ratings = [(uid, iid, float(r), None, None)
                                for (uid, iid, r) in
                                self.df.itertuples(index=False)]

    def build_full_trainset(self):
        return self.construct_trainset(self.raw_ratings)
