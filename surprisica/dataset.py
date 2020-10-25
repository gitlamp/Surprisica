
from __future__ import (absolute_import, print_function, unicode_literals, division)

from collections import defaultdict

import numpy as np
from sklearn.preprocessing import minmax_scale
from surprise.dataset import Dataset

from .trainset import Trainset


class Dataset(Dataset):

    def __init__(self, reader):
        super(Dataset, self).__init__(reader)
        self.context = reader.context

    @classmethod
    def load_from_df(cls, df, reader):
        rating = cls.create_profile(df=df, reader=reader)

        return DatasetAutoFolds(df=rating, reader=reader)

    @classmethod
    def create_profile(cls, df, reader):
        if reader.context:
            rating = cls.create_rating_profile(df, reader)
            context = cls.create_context_profile(df, reader)
            profile = rating.merge(context)

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
        uid = col[0]
        lid = col[1]
        cnx = col[2:]
        df = df.filter(col)
        # Check context exists
        for c in reader.cnx_entities:
            if df[c].dtype != int:
                raise ValueError(f'Column "{c}" should have integer type.')

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
        res = df[[uid, lid] + cnx].merge(res)

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


class DatasetAutoFolds(Dataset):

    def __init__(self, df=None, reader=None):
        Dataset.__init__(self, reader)

        if df is not None:
            self.df = df
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
        else:
            raise ValueError('Must specify rating dataframe.')

    def build_full_trainset(self):
        return self.construct_trainset(self.raw_ratings)
