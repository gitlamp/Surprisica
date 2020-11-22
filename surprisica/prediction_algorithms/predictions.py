
from __future__ import (absolute_import, print_function, unicode_literals, division)

from collections import namedtuple


class PredictionImpossible(Exception):
    """Exception raised when a prediction is impossible.
    When raised, the estimation :math:`\hat{r}_{ui}` is set to the global mean
    of all ratings :math:`\mu`."""
    pass


class Prediction(namedtuple('Prediction',
                            ['uid', 'iid', 'cid', 'r_ui', 'est', 'details'])):
    """A named tuple for storing the results of a prediction.
   It's wrapped in a class, but only for documentation and printing purposes.

    Args:
       uid:() The raw user id.
       iid:() The raw item id.
       cid:(tuple) The raw context id.
       r_ui(float): The true rating :math:`r_{ui}`.
       est:(float) The estimated rating :math:`\\hat{r}_{ui}`.
       details(dict): A dictionary containing all details related to predictions."""

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
