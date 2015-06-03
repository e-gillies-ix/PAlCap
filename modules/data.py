import numpy as np
from root_numpy import root2array
#import math

"""
Notation used below:
"""


class AlCapROOT(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    def __init__(self, path="data/signal_TDR.root", treename='tree'):
        """
        Dataset provides an interface to work with MC stored in root format.
        Results of methods are either numpy.arrays or scipy.sparse objects.

        :param path: path to rootfile
        :param treename: name of the tree in root dataset
        """
        self.entry_data = root2array(path, treename=treename)
        self.hits_data = self._get_hits()

    @property
    def n_entries(self):
        return len(self.entry_data)

    def _get_hits(self):
        """
        Creates numpy array of hits, each with their own entry in the table
        """
        # Variables defined over the whole entry
        entry_variables = ["M_nHits", "evt_num", "run_num", "weight"]
        # Variables defiend for each it
        hit_variables = list(set(self.entry_data[0].dtype.names) -
                             set(entry_variables))
        # Define new array for all of the hits
        all_hits = np.zeros((sum(self.entry_data["M_nHits"])),
                                dtype=self.entry_data[0].dtype)
        i = 0
        for entry in range(self.n_entries):
            for hit in range(self.entry_data[entry]["M_nHits"]):
                for evt_var in entry_variables:
                    all_hits[i][evt_var] = self.entry_data[entry][evt_var]
                for hit_var in hit_variables:
                    all_hits[i][hit_var] = self.entry_data[entry][hit_var][hit]
                i += 1
        return all_hits

    def filter_hits(self, variable, value):
        """
        Returns the sequence of hits that pass through the named volume
        """
        variable_ids = self.hits_data[variable]
        hits_in_vol = np.where(variable_ids == value)
        return hits_in_vol[0]

    def get_measurement(self, vol_name, meas_name):
        """
        Returns requested measurement in all wires in requested entry
        :return: numpy.array of shape [total_wires]
        """
        hits_in_vol = self.filter_hits("M_volName", vol_name)
        return  self.hits_data[hits_in_vol][meas_name]
