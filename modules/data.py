import numpy as np
from root_numpy import root2array
import os
import subprocess
from collections import OrderedDict
#import math

"""
Notation used below:
"""

class AlCapROOT(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    def __init__(self, path="data/signal_TDR.root", treename='tree',
            branches=None):
        """
        Dataset provides an interface to work with MC stored in root format.
        Results of methods are either numpy.arrays or scipy.sparse objects.

        :param path: path to rootfile
        :param treename: name of the tree in root dataset
        """
        self.hits_data, self.hits_to_entries, self.entry_to_hits,\
                self.entry_to_n_hits = self._get_hits(path, treename=treename,\
                branches=branches)


        self.n_hits = len(self.hits_data)
        self.n_entries = self.hits_data["evt_num"][self.n_hits-1]

    def _get_hits(self, path, treename, branches):
        """
        Creates numpy array of hits, each with their own entry in the table
        """

        # Variables defined over the whole entry
        entry_variables = ["M_nHits", "evt_num"]
        total_branches = entry_variables + branches
        entry_data = root2array(path, treename=treename,\
                branches=total_branches)
        # Variables defiend for each hit
        hit_variables = list(set(entry_data[0].dtype.names)\
                           - set(entry_variables))
        # Get all names in the correct order
        all_names = hit_variables + entry_variables

        # Create a look up table that maps from hit number to event number
        hits_to_events = np.zeros(sum(entry_data["M_nHits"]))
        # Create a look up table that maps from event number to first hit in
        # that event
        event_to_hits = np.zeros(len(entry_data))
        event_to_n_hits = entry_data["M_nHits"].copy()
        i = 0
        for entry in range(len(entry_data)):
            event_to_hits[entry] = i
            event_hits = entry_data[entry]["M_nHits"]
            hits_to_events[i:i+event_hits] = entry
            i += event_hits
        hits_to_events = hits_to_events.astype(int)

        # Get columns for each hit specific variable
        all_columns = [np.concatenate(entry_data[hit_var])\
                       for hit_var in hit_variables]

        # Stretch out the event variables across corresponding hits
        for ent_var in entry_variables:
            new_column = entry_data[ent_var][hits_to_events]
            all_columns.append(new_column)

        # Create a record array from these columns, use the name names as before
        hits_table = np.rec.fromarrays(all_columns, names=(all_names))
        del entry_data
        return hits_table, hits_to_events.astype(int),\
                event_to_hits.astype(int), event_to_n_hits.astype(int)

    def get_event_to_hits(self, events):
        """
        Returns the hits from the given events
        """
        n_hits = sum(self.entry_to_n_hits[events])
        hits_array = np.zeros(n_hits)
        index = 0
        for entry in events:
            first_hit = self.entry_to_hits[entry]
            evt_n_hits = self.entry_to_n_hits[entry]
            hits_array[index:index+evt_n_hits] = range(first_hit, \
                                                       first_hit + evt_n_hits)
            index += evt_n_hits
        return hits_array.astype(int)

    def get_other_hits(self, hits):
        """
        Returns the hits from the same event(s) as the given hit list
        """
        events = self.hits_to_entries[hits]
        return self.get_event_to_hits([events])

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


    def get_histogram_1d(self, hits, bins=20, x_axis="M_x"):
        # Get the X distribution of the hits
        x_dist = self.hits_data[hits][x_axis]
        # Histogram the values
        data, bin_edges = np.histogram(x_dist, bins=bins)
        return data, bin_edges

    def get_histogram_2d(self, hits, bins=20, x_axis="M_x", y_axis="M_y"):
        # Get the X and Y distribution of the hits
        x_dist = self.hits_data[hits][x_axis]
        y_dist = self.hits_data[hits][y_axis]
        # Histogram these values
        histo2d, x_bins, y_bins = np.histogram2d(x_dist, y_dist, bins=bins)
        # Flatten this histogram
        data = histo2d.flatten()
        bin_edges = np.hstack((x_bins, y_bins))
        return data, bin_edges


class G4SimRunner(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    def __init__(self, geom_dir="geometry_R2015a/",
                       geom_name="geometry_Al2000_june2015"):
        """
        Dataset provides an interface to work with MC stored in root format.
        Results of methods are either numpy.arrays or scipy.sparse objects.

        :param path: path to rootfile
        :param treename: name of the tree in root dataset
        """
        # Record original directory of ipython notebook session
        self.pwd = os.path.expandvars("$PWD")

        ## Source the environment variable ##
        self.g4sim_root = os.path.normpath(self.pwd+"/../../g4sim")
        g4sim_env = self.g4sim_root+"/env.sh"
        if os.path.isfile(g4sim_env):
            self.commands = "cd "+self.g4sim_root+"; "
            self.commands += "source "+g4sim_env+"; "
        else:
            print "Cannot find the g4sim environment"
        ## Run this program in the g4sim/alcap directory
        run_dir = self.g4sim_root+"/alcap/"
        self.commands += "cd "+run_dir+"; "
        self.config_dir = run_dir+"configure/"
        self.gen_mac = "gen/PAlCap_gen"
        self.out_mac = "output/PAlCap_out"
        self.geom_mac = geom_dir+geom_name
        self.macros_dir = run_dir +"macros/"
        self.data_dir = run_dir +"output/"
        self.run_mac = "PAlCap_run"

    def run_simulation(self, data_path=""):
        final_commands = self.commands + "g4sim "+self.macros_dir+\
                                                  self.run_mac+"; "
        subprocess.call([final_commands], shell=True)
        if data_path != "":
            subprocess.call(["mv "+self.data_dir+"raw_g4sim.root "+data_path],
                    shell=True)

    def generate_gen_mac(self, mac_path="", changed_settings=None):
        # Set all as default
        settings = OrderedDict([\
        ("PhysicsList", "QGSP_BERT"),
        ("Type", "simple"),
        ("Particle", "mu-"),
        ("Theta", 0),
        ("Phi", 0),
        ("ThetaSpread", 0),
        ("PhiSpread", 0),
        ("MomAmp", 28.),
        # Defines 1.03 full width at half maximum width
        ("MomSpread", (0.357/28.) * 28.),
        ("x", 0),
        ("y", 0),
        ("z", -230),
        ("PosSpreadX", 6),
        ("PosSpreadY", 6),
        ("PosSpreadZ", 0),
        ("PosLimit", 6),
        ("RandMode", "none"),
        ("EnergyMode", "none"),
        ("PositionMode", "none"),
        ("DirectionMode", "none"),
        ("ThetaMode", "none"),
        ("PhiMode", "none"),
        ("TimeMode", "none"),
        ("pidMode", "none"),
        ("Time", 0)])
        # Check for new settings
        if changed_settings != None:
            new_vals = dict(changed_settings)
            # Loop through new settings
            for parameter, new_value in new_vals.iteritems():
                # Check this is a recognized settings
                if parameter in settings:
                    # Access the value, change it to the new value
                    settings[parameter] = new_value
                # Defines 1.03 full width at half maximum width by default
                # Default value is over ridden if explicity defined
                if  parameter == "MomAmp" and "MomSpread" not in new_vals:
                    settings["MomSpread"] = (0.357/28.)*new_vals["MomAmp"]

        gen_macro = open(self.config_dir+self.gen_mac, 'w')
        gen_macro.write("\n"+
       "# A default generator file that shouldn't be used\n"+
       "     PhysicsList: "+settings["PhysicsList"]+"\n"+
       " \n"+
       "  #General setting\n"+
       "     Type:      "+settings["Type"]+" #(simple/stable/ion)\n"+
       "     Particle:  "+settings["Particle"]+
                        " #(e+/e-/deuteron/chargedgeantino/gamma/...)\n\n"+
       "#Default Direction\n"+
       " #                 Theta    Phi\n"+
       " #                 deg      deg\n\n"+
       "  Direction:    "+str(settings["Theta"])+"  "+str(settings["Phi"])+"\n"+
       "  ThetaSpread:  "+str(settings["ThetaSpread"])+"\n"+
       "  PhiSpread:    "+str(settings["PhiSpread"])+"\n\n"+
       "#Default Momentum Amplitude\n"+
       " #                MeV\n"+
       "     MomAmp:      "+str(settings["MomAmp"])+"\n"+
       "     MomSpread:   "+str(settings["MomSpread"])+"\n"+
       "#Default Position     x     y     z\n"+
       " #                     mm    mm    mm\n"+
       "     Position:         "+str(settings["x"])+"    "+
                                 str(settings["y"])+"    "+
                                 str(settings["z"])+"\n"
       "     PosSpread:        "+str(settings["PosSpreadX"])+"    "+
                                 str(settings["PosSpreadY"])+"    "+\
                                 str(settings["PosSpreadZ"])+"\n"+
       "     PosLimit:         "+str(settings["PosLimit"])+"\n\n"+
       "     RandMode:       "+settings["RandMode"]+
                                    " #(none/root)\n"+
       "     EnergyMode:     "+settings["EnergyMode"]+\
                                    " #(root/histo/none)\n"+
       "     PositionMode:   "+settings["PositionMode"]+
                                    "  #(uniform/root/none/gRand/uRand)\n"+
       "     DirectionMode:  "+settings["DirectionMode"]+
                                    "  #(none/uniform/histo)\n"+
       "     ThetaMode:      "+settings["ThetaMode"]+
                                    "  #(none/gRand/uRand)\n"+
       "     PhiMode:        "+settings["PhiMode"]+
                                    "  #(none/gRand/uR\and)\n"+
       "     TimeMode:       "+settings["TimeMode"]+
                                    "  #(none/root)\n"+
       "     pidMode:        "+settings["pidMode"]+
                                    "  #(none/root)\n"+
       "#Default Time\n"+
       "#            ns\n"+
       "    Time:    "+str(settings["Time"]))
        gen_macro.close()
        if mac_path != "":
            subprocess.call(["cp "+self.config_dir+self.gen_mac+" "+mac_path],
                    shell=True)

    def generate_out_mac(self, mac_path="", selected_branches=None):
        # Set all as default
        if selected_branches == None:
            # TODO Save this as a data member, only let people select things
            # from this list
            branches = OrderedDict([
                ("x", "cm"), ("y", "cm"), ("z", "cm"), ("t", "ns"),
                ("local_x", "cm"), ("local_y", "cm"), ("local_z", "cm"),
                ("Ox", "cm"), ("Oy", "cm"), ("Oz", "cm"), ("Ot", "ns"),
                ("ox", "cm"), ("oy", "cm"), ("oz", "cm"), ("ot", "ns"),
                ("local_ox", "cm"), ("local_oy", "cm"), ("local_oz", "cm"),
                ("local_Ox", "cm"), ("local_Oy", "cm"), ("local_Oz", "cm"),
                ("px", "GeV"), ("py", "GeV"), ("pz", "GeV"),
                ("opx", "cm"), ("opy", "cm"), ("opz", "cm"),
                ("Opx", "GeV"), ("Opy", "GeV"), ("Opz", "GeV"),
                ("e", "GeV"), ("edep", "GeV"), ("edepAll", "GeV"),
                ("stepL", "cm"), ("kill_time", "ns"), ("stop_time", "ns")])
        else:
            branches = OrderedDict(selected_branches)
        out_macro = open(self.config_dir+self.out_mac, 'w+')
        out_macro.write("\n\
        tree_name      tree\n\
        AutoSave          0     //0: no auto save; n(n>0): call"+\
                                  " m_tree->AutoSave() every n events\n\
        Circular          0     //0: do not set circular; n(n>0):"+\
                                  " call m_tree->SetCircular(n)\n\
        Verbose           0     //for classes related to output, "+\
                                  " including MyRoot, Cdc*SD, etc.\n\
        PrintModule       10000 //for classes related to output,"+\
                                  "including MyRoot, Cdc*SD, etc.\n\
        \n\
        EVENTHEADER_SECTION\n\
            evt_num\n\
            run_num\n\
            weight\n\
        EVENTHEADER_SECTION\n\
        \n\
        \n\
        MCTRUTH_SECTION\n\
            nTracks\n\
            pid\n\
            tid\n\
            ptid\n\
            ppid\n\
            time              ns\n\
            px                GeV\n\
            py                GeV\n\
            pz                GeV\n\
            e                 GeV\n\
            x                 cm\n\
            y                 cm\n\
            z                 cm\n\
            charge\n\
            particleName\n\
            process\n\
            volume\n\
            tid2pid\n\
            tid2time\n\
            MCTRUTH_SECTION\n\
            \n\
            MCTRUTHFILTER_SECTION\n\
            #   Switch                     //if commented, then program will "+\
                                           " not generate any hit\n\
            #   nTracks       1            //maximum tracks cut\n\
            #   minp          0      GeV   //minimum momentum cut\n\
            #   mine          0      MeV   //minimum momentum cut\n\
            #   mint          0      ns    //left end of time window, 0 means"+\
                                           " no lower limit\n\
            #   maxt          0      ns    //right end of time window, 0"+\
                                           " means no upper limit\n\
            #   WL            0            //Add a PDGEncode to white list."+\
                                           " if white list is not empty then"+\
                                           " only particle on white list"+\
                                           " will be recorded. 0 means"+\
                                           " pid<1e7\n\
            #   BL            2112\n\
            MCTRUTHFILTER_SECTION\n\
        \n\
        M_SECTION\n\
            nHits\n\
            volID\n\
            volName\n\
            ovolName\n\
            oprocess\n\
            ppid\n\
            ptid\n\
            tid\n\
            pid\n\
            particleName\n\
            charge\n\
            stopped\n\
            killed\n")
        # Fill in optional branches
        for variable, unit in branches.iteritems():
            out_macro.write("\
            "+variable+"     "+unit+"\n")
        out_macro.write("\
        M_SECTION\n\
        \n\
        M_FILTERSECTION\n\
            Switch                  //if commented, then program will"+\
                                    " not generate any hit\n\
            #  neutralCut           //if not commented, then neutral "+\
                                    " tracks will not be recorded\n\
            #  stopped              //need to be stopped inside\n\
            #  maxn     0           //maximum hits cut, 0 means no limit\n\
            #  minp     0     GeV   //minimum momentum cut\n\
            #  mine     0     GeV   //minimum energy cut\n\
            #  mint     0     ns    //left end of time window, 0 means no"+\
                                    " lower limit\n\
            #  maxt     0     ns    //right end of time window, 0 means no"+\
                                    " upper limit\n\
               tres     1e10  ns    //time resolution\n\
               minedep  1     GeV   //minimum energy deposition cut\n\
            #  WL       13          //Add a PDGEncode to white list. if white"+\
                                    " list is not empty then only particle"+\
                                    " on white list will be recorded. 0"+\
                                    " means pid<1e7\n\
            #  WL       -211        //Add a PDGEncode to white list. if white"+\
                                    " list is not empty then only particle"+\
                                    " on white list will be recorded. 0"+\
                                    " means pid<1e7\n\
        M_FILTERSECTION\n\
        \n\
        V_SECTION\n\
            x              cm\n\
            y              cm\n\
            z              cm\n\
            t              ns\n\
            px             GeV\n\
            py             GeV\n\
            pz             GeV\n\
            edep           GeV\n\
            volName\n\
            ovolName\n\
            oprocess\n\
            ppid\n\
            ptid\n\
            tid\n\
            pid\n\
            particleName\n\
        V_SECTION\n\
        \n\
        V_FILTERSECTION\n\
            Switch                        //if commented, then program will"+\
                                          " not generate any hit\n\
            tres           20    ns       //time resolution\n\
            minedep        500   eV       //minimum energy deposition cut\n\
        V_FILTERSECTION\n")
        out_macro.close()
        if mac_path != "":
            subprocess.call(["cp "+self.config_dir+self.out_mac+" "+mac_path],
                    shell=True)

    def generate_run_mac(self, n_events, mac_path="", vis=False):
        run_macro = open(self.macros_dir+self.run_mac, "w")
        run_macro.write("/control/getEnv ALCAPWORKROOT\n")
        run_macro.write("\n")
        run_macro.write("# get default settings\n")
        run_macro.write("/control/execute"+
                       " {ALCAPWORKROOT}/macros/resetVerbose.mac\n")
        run_macro.write("/control/execute"+
                       " {ALCAPWORKROOT}/macros/resetCut.mac\n")
        run_macro.write("\n")
        run_macro.write("# get visualisation\n")
        if vis:
            run_macro.write("/control/execute "+
                       " {ALCAPWORKROOT}/macros/june_geom_check.mac\n\n\n")
        run_macro.write("# set output\n")
        run_macro.write("/g4sim/myAnalysisSvc/set_out_card "+
                                            self.out_mac+"\n\n")
        run_macro.write("# Set generator settings\n")
        run_macro.write("/g4sim/gun/ResetGen               "+
                                            self.gen_mac+" \n\n")
        run_macro.write("# Set geometry settings\n")
        run_macro.write("/g4sim/det/ReloadGeo              "+
                                            self.geom_mac+" \n\n\n")
        run_macro.write("#Turn on the beam!\n")
        run_macro.write("/run/beamOn  "+str(n_events)+"\n\n")
        if vis:
            run_macro.write("/vis/ogl/printEPS\n")
        run_macro.close()
        if mac_path != "":
            subprocess.call(["cp "+self.macros_dir+self.run_mac+" "+mac_path],
                    shell=True)
