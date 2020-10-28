
from __future__ import (absolute_import, print_function, unicode_literals, division)

from collections import namedtuple


class PredictionImpossible(Exception):
    pass


class Prediction(namedtuple('Prediction',
                            ['uid', 'iid', 'cid', 'r_ui', 'est', 'details'])):
    __slots__ = ()

    def __str__(self):
        s = 'user: {uid:<10} '.format(uid=self.uid)
        s += 'item: {iid:<10} '.format(iid=self.iid)
        if self.r_ui is not None:
            s += 'r_ui = {r_ui:1.2f}   '.format(r_ui=self.r_ui)
        else:
            s += 'r_ui = None   '
        if self.cid is not None:
            s += 'c_ui = {cid}   '.format(cid=self.cid)
        else:
            s += 'cid = None   '
        s += 'est = {est:1.2f}   '.format(est=self.est)
        s += str(self.details)

        return s
