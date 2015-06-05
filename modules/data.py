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
        self.hits_data, self.hits_to_entries, self.entry_to_hits\
                = self._get_hits()

    @property
    def n_entries(self):
        return len(self.entry_data)

    @property
    def n_hits(self):
        return sum(self.entry_data["M_nHits"])

    def _get_hits(self):
        """
        Creates numpy array of hits, each with their own entry in the table
        """

        # Variables defined over the whole entry
        entry_variables = ["M_nHits", "evt_num", "run_num", "weight"]
        # Variables defiend for each hit
        hit_variables = list(set(self.entry_data[0].dtype.names)\
                           - set(entry_variables))
        # Get all names in the correct order
        all_names = hit_variables + entry_variables

        # Create a look up table that maps from hit number to event number
        hits_to_events = np.zeros(self.n_hits)
        # Create a look up table that maps from event number to first hit in
        # that event
        event_to_hits = np.zeros(self.n_entries)
        i = 0
        for entry in range(self.n_entries):
            event_to_hits[entry] = i
            event_hits = self.entry_data[entry]["M_nHits"]
            hits_to_events[i:i+event_hits] = entry
            i += event_hits
        hits_to_events = hits_to_events.astype(int)

        # Get columns for each hit specific variable
        all_columns = [np.concatenate(self.entry_data[hit_var])\
                       for hit_var in hit_variables]

        # Stretch out the event variables across corresponding hits
        for ent_var in entry_variables:
            new_column = self.entry_data[ent_var][hits_to_events]
            all_columns.append(new_column)

        # Create a record array from these columns, use the name names as before
        hits_table = np.rec.fromarrays(all_columns, names=(all_names))
        return hits_table, hits_to_events.astype(int), event_to_hits.astype(int)

    def get_event_to_hits(self, events):
        """
        Returns the hits from the given events
        """
        hits_array = [0]
        for entry in events:
            first_hit = self.entry_to_hits[entry]
            last_hit = self.entry_data[entry]["M_nHits"] + first_hit
            hits = range(first_hit, last_hit)
            hits_array.extend(hits)
        hits_array = np.array(hits_array)
        hits_array = np.trim_zeros(hits_array)
        return hits_array.astype(int)

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
