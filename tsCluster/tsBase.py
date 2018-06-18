import numpy as np
from tsCluster.cyclicAnalysis import cyclic_analysis, norms, trend_removals


class TsBase:
    """A module to hold time series data collected in runs and sessions.

    This module holds data time-series data that are collected in 'runs' (run1, run2, etc.) which belong to 'sessions'
    (s1, s2, etc) from two locations (San Antonio,TX and Champaign, IL). The module loads the data from disk based on
    a naming convention, defined in 'name_structure' and a folder containing the data files, defined in 'file_list'.

    The default naming convention and data folder are defined in internal variables: __def_name_structure and
    __def_file_list.

    Each instance of TsCluster holds the following attributes:

        - name: The name of the instance
        - norm: The normalization performed on the multi variate time series data. Allowed norms are 'tv'
                (total variation) and 'sqr' ( Frobenius norm). Type is  a string.
        - trend_removal : The trend removal applied on the time series data. Allowed trend removals are 'none' (do
                nothing), 'linear' (remove a linear trend in the data), 'bias' (reapply mean centering). Type is a
                string.
        - samples: A list of python dictionary objects that hold the actual data. Each sample has the following keys
            - Name: A subject identifier.
            - Group; The particular group the subject belongs to - tinnitus, hearing loss, controls, tinnitus with
                hearing loss by default
            - Location: Location where the particular data was recorded - San Antonio or Champaign by default
            - Run: The particular run in which the datum was collected - default values are 'run1', 'run2', 'run3'
            - Session: The session to which the particular datum belongs - 's1', or 's2' by default.
            - TimeSeries: A numpy array that holds the multivariate time series
            - ULM: Un-sorted lead matrix
            - Phases: Leading eigenvector of the lead matrix
            - Permutation: Ordering recovered from components of the leading eigenvector
            - SLM: Sorted lead matrix, sorted according to recovered ordering
            - Eigenvalues: Eigenvalues corresponding to Phases
            - FlatULM: Lead matrix flattened into a vector for faster computation
            - FlatSLM
            - NormedTS: The normalized time series data
            - CM: The correlation matrix from the time-series data
            - FlatCM
        - covariance_matrix: The covariance matrix from the samples in consideration during analysis
        - projectors: The leading eigenvectors of the covariance matrix
        - cov_eigenvalues: The eigenvalues of the covariance matrix
        - groups: The set of groups that the samples belong to
        - runs: The set of runs the samples belong to
        - sessions: The set of sessions the samples belong to
        - locations: The set of locations that samples belong
        - removed_counter: A helper dictionary object for adding and removing samples to analysis
        - data_counter: A helper dictionary object for adding and removing samples to analysis
        - removed_samples: A list to hold samples removed from analysis
        - deleted_samples: A list to hold samples deleted from analysis
        - rois: A list of strings corresponding to regions of interest in consideration.

    Each instance of TsCluster has the following methods:

        - save_to_file(): Save the instance to disk using pickle and gzip
        - reset(): A helper function to recalculate covariance structure after adding or removing samples
        - recompute(): A helper function recompute lead matrices after making any changes to time series data
        - from_directory(): A constructor to load data set from a directory with a naming convention.
        - from_manual(): A constructor to load data set while interactively specifying naming convention
        - mod_samples(): A function to add or remove subjects into analysis
        - mod_data_rem(): A function to remove runs or sessions from analysis
        - mod_data_add(): A function to add back runs or sessions from analysis
        - mod_loc(): A function to add or remove locations from analysis
        - mod_group(): A function to add or remove groups from analysis
        - edit_time_series: A function edit a particular time series in the analysis.
        - remove_roi(): A function that removes a region of interest from analysis.

    """
    __def_name_structure__ = {'groups': ['ctr', 'tin+hl', 'tin', 'hl'], 'locations': ['san', 'chm'],
                              'runs': ['run1', 'run2', 'run3'], 'sessions': ['s1', 's2'], 'short_name': 'sub',
                              'file_ext': '.txt', 'norm': 'tv', 'trend_removal': 'none'}
    __def_file_list__ = './tsCluster/data'

    def __init__(self, name=None):
        """ Initializes an instance of the TsCluster module.

            Keyword Arguments:
                name -- A name for the instance, if not specified tries to infer name of variable it is being assigned

            When called as:
                var_name = TsCluster()
            it initializes all the necessary attributes of the instance so that the call to constructor methods
            from_directory and from_manual can be made.
        """
        import traceback
        if name is None:
            (filename, line_number, function_name, text) = traceback.extract_stack()[-2]
            def_name = text[:text.find('=')].strip()
            self.name = def_name
        self.norm = ""
        self.trend_removal = ""
        self.samples = []
        self.covariance_matrix = []
        self.projectors = []
        self.cov_eigenvalues = []
        self.groups = set()
        self.locations = set()
        self.runs = set()
        self.data_counter = {}
        self.sessions = set()
        self.removed_samples = []
        self.removed_counter = {}
        self.deleted_samples = []
        self.rois = ['lamygdala', 'lanteriorinsula', 'lcuneus', 'lfrontaleyefield', 'linferiorfrontallobe',
                     'linferiorparietallobe', 'lmidfrontalgyrus', 'lparahippocampus', 'lposteriorintraparietalsulcus',
                     'lprimaryauditorycortex', 'lprimaryvisualcortex', 'lsuperioroccipitallobe',
                     'lsuperiortemporaljunction', 'lsuperiortemporalsulcus', 'lventralintraparietalsulcus',
                     'medialprefrontalcortex', 'posteriorcingulatecortex', 'precuneus', 'ramygdala', 'ranteriorinsula',
                     'rcuneus', 'rfrontaleyefield', 'rinferiorfrontallobe', 'rinferiorparietallobe', 'rmidfrontalgyrus',
                     'rparahippocampus', 'rposteriorintraparietalsulcus', 'rprimaryauditorycortex',
                     'rprimaryvisualcortex', 'rsuperioroccipitallobe', 'rsuperiortemporaljunction',
                     'rsuperiortemporalsulcus', 'rventralintraparietalsulcus']

    def __repr__(self):
        """For use with evaluation in CLI"""
        return "TimeSeries clustering object" + self.name

    def __str__(self):
        """For use with the str method or print method in CLI"""
        if not self.samples:
            return 'Uninitialized'
        else:
            str0 = '----\t' + __name__ + ':' + self.name + '\t----\n'
            str1 = 'Num of samples: \t' + str(len(self.samples)) + '\n'
            group_data = {group: len([sample['Name'] for sample in self.samples if
                                      sample['Group'] == group]) for group in self.groups}
            str2 = 'Groups in analysis: \t' + str(group_data) + '\n'
            str3 = 'Runs in analysis: \t' + str(self.runs) + '\n'
            str4 = 'Sessions in analysis:\t' + str(self.sessions) + '\n'
            str5 = 'Locations in analysis:\t' + str(self.locations) + '\n'
            str6 = 'Data counter: \t' + str(self.data_counter) + '\n'
            str7 = 'Number of deleted samples: \t' + str(len(self.deleted_samples)) + '\n'
            str8 = 'Number of removed samples: \t' + str(len(self.removed_samples)) + '\n'
            str9 = 'Removal counter: \t' + str(self.removed_counter) + '\n'

        return str0 + str1 + str2 + str3 + str4 + str5 + str6 + str7 + str8 + str9

    def __default_constructor__(self, file_list, name_structure):
        """ The default constructor for the class.  Calls load_data() and called by from_directory() or  from_manual().

            Arguments:
                file_list -- Usually  a list of strings, where each string is a valid filename pointing to a text file
                    that holds the multivariate time series data. It can also be a dictionary whose keys are groups in
                    the analysis and values are list of strings corresponding to file names in the analysis.
                name_structure -- A dictionary specifying the naming convention in use.
        """
        self.norm = name_structure['norm']
        self.trend_removal = name_structure['trend_removal']

        # If its a dict, then it likely came from constructor from_manual, and groups were specified
        if type(file_list) == dict:
            self.groups = list(file_list.keys())
            self.__load_data__(file_list, name_structure)
        # If its a list, then it likely came from constructor from_directory and groups were not specified
        elif type(file_list) == list:
            self.groups = []  # Let default constructor parse the groups present
            self.__load_data__({'': file_list},
                               name_structure)  # load_data() MUST always get a dict in the first argument
        else:
            print('Error! file_list must be a dict or a list')
            raise ValueError
        self.reset()

        # Initialize the data_counter object after all the data is loaded
        for location in self.locations:
            for item in self.sessions:
                self.data_counter[location + '-' + item] = set(sample['Run'] for sample in self.samples
                                                               if sample['Session'] == item and
                                                               sample['Location'] == location)

    def __load_data__(self, file_list, name_structure):
        """ Function that actually loads the data. Called by default_constructor()

            Arguments:
                file_list -- A dictionary specifying groups in the data set and the associated file names.
                name_structure -- A dictionary specifying naming convention in use.
        """

        def parse_name(file_name):
            """ Function to parse a file name given a naming convention. """
            short_name, f_ext = name_structure['short_name'], name_structure['file_ext']
            run_list, session_list = name_structure['runs'], name_structure['sessions']
            group_list, loc_list = name_structure['groups'], name_structure['locations']

            # Parse identifier string
            try:
                sample_name = short_name + file_name[(file_name.index(short_name) +
                                                      len(short_name)):file_name.index(f_ext)]
            except ValueError:
                print("Error: String not found in '" + str(file_name) + "' !")
                raise LookupError

            # Parse identifier number
            try:
                sample_run_number = next(run for run in run_list if run in file_name.split('/')[-1])
            except StopIteration:
                print('The runs: ' + str(run_list) + ' were not found in' + file_name.split('/')[-1])
                raise LookupError

            # Parse sample's session id
            try:
                sample_session_number = next(session for session in session_list if session in file_name.split('/')[-1])
            except StopIteration:
                print('The sessions: ' + str(session_list) + ' were not found in' + file_name.split('/')[-1])
                raise LookupError

            # Parse sample's group
            try:
                sample_group = next(item for item in group_list if item in file_name.split('/')[-1])
                if sample_group == 'tin+hl':  # We don't want '+' in a key value, groups are keys in data_counter
                    sample_group = 'tin_hl'
            except StopIteration:
                print('The groups: ' + str(group_list) + ' were not found in' + file_name.split('/')[-1])
                raise LookupError

            # Parse sample's location
            try:
                sample_loc = next(item for item in loc_list if item in file_name.split('/')[-1])
            except StopIteration:
                print('The groups: ' + str(group_list) + ' were not found in' + file_name.split('/')[-1])
                raise LookupError

            return sample_session_number, sample_run_number, sample_group, sample_name, sample_loc

        for groups in file_list:  # That is keys in the dictionary file_list
            for sample in file_list[groups]:  # Iterate over the list of file_names corresponding to group
                if groups == '':  # from_directory was called and groups were not given/parsed
                    try:
                        session_number, run_number, group, name, loc = parse_name(file_name=sample)
                    except UnboundLocalError:
                        print('Error: There is a property/filename mismatch!!')
                        raise ValueError
                else:  # Groups and associated file names are given so no need to parse groups.
                    try:
                        session_number, run_number, _, name, loc = parse_name(file_name=sample)
                    except UnboundLocalError:
                        print('Error: There is a property/filename mismatch!!')
                        raise ValueError
                time_series = np.genfromtxt(sample, delimiter=',')

                # Don't want the normalized time series to have matched ends
                ret, _ = cyclic_analysis(time_series, p=1, normalize=name_structure['norm'],
                                         trend_removal=name_structure['trend_removal'])

                # If you DO want matched ends in the time series then comment below, and
                # replace _ with normed_time_series above
                (_, normalize) = norms[self.norm]
                (_, de_trend) = trend_removals[self.trend_removal]
                normed_time_series = de_trend(normalize(time_series))
                lm, phases, perm, sorted_lm, evals = ret
                (_, n) = lm.shape
                cm = np.corrcoef(normed_time_series)

                # Have all the data, so append it to samples.
                self.samples.append({"Name": name, "Group": group, "Location": loc, "Run": run_number,
                                     "Session": session_number, "TimeSeries": time_series, "ULM": lm, "Phases": phases,
                                     "Permutation": perm, "SLM": sorted_lm, "Eigenvalues": evals,
                                     "FlatULM": lm[np.triu_indices(n, 1)], "FlatSLM": sorted_lm[np.triu_indices(n, 1)],
                                     "NormedTS": normed_time_series, "CM": cm, 'FlatCM': cm[np.triu_indices(n, 1)]})

    def save_to_file(self, file_name=None):
        """ Saves the instance to disk.

            If file name is not provided it opens a TK dialog box to save the file to disk. Filename can also be
            provided as a string with a valid path.
        """
        import pickle
        import gzip
        from tkinter import Tk, filedialog
        root = Tk()
        if not file_name:
            file_name = filedialog.asksaveasfilename()
        root.destroy()
        with gzip.open(file_name, 'wb') as file:
            pickle.dump(self, file, protocol=-1)

    def reset(self):
        """ Re-calculates covariance structure.

            The reset() method is useful after removing or adding data to the analysis - often we remove outliers or
            subjects that showed too much motion.
        """
        if not self.samples:
            print('Error: Data has not been loaded yet!')
        else:
            self.groups = set([sample['Group'] for sample in self.samples])
            self.sessions = set([sample['Session'] for sample in self.samples])
            self.runs = set([sample['Run'] for sample in self.samples])
            self.locations = set([sample['Location'] for sample in self.samples])
            temp_var = [self.samples[i]["FlatULM"] for i in range(len(self.samples))]
            self.covariance_matrix = np.cov(np.asarray(temp_var).T)
            self.projectors, self.cov_eigenvalues, _ = np.linalg.svd(self.covariance_matrix)

    def recompute(self):
        """ Re-computes lead matrices and calls reset()

            The recompute() method is used whenever the time series data itself are edited. This means the lead
            matrices will change, and so need to be re-calculated. Since the lead matrices change the covariance
            structure also changes, and hence we need to call reset().
        """
        if not self.samples:
            print('Error: Data has not been loaded yet!')
        else:
            for sample in self.samples:
                ret, normed_time_series = cyclic_analysis(sample['TimeSeries'], p=1, normalize=self.norm,
                                                          trend_removal=self.trend_removal)
                lm, phases, perm, sorted_lm, eigenvalues = ret
                cm = np.corrcoef(normed_time_series)
                (_, n) = lm.shape
                sample['SLM'] = sorted_lm
                sample['ULM'] = lm
                sample['Eigenvalues'] = eigenvalues
                sample['Phases'] = phases
                sample['Permutation'] = perm
                sample['CM'] = cm
                sample['NormedTS'] = normed_time_series
                sample['FlatULM'] = lm[np.triu_indices(n, 1)]
                sample['FlatSLM'] = sorted_lm[np.triu_indices(n, 1)]
                sample['FlatCM'] = cm[np.triu_indices(n, 1)]

        self.reset()

    def from_directory(self, name_structure=None, directory=None):
        """ Constructor to load data from a directory with a naming convention. Calls default_constructor(). 

            Keyword arguments:
                name_structure -- A dictionary specifying the naming convention in use in the directory
                directory -- A string specifying the directory to load data form

            If arguments are not provided uses defaults specified by the class variables.
        """
        from os import listdir, path

        def listdir_fp(d):
            """ Gives a list of path+file names from the directory d."""
            return [path.join(d, f) for f in listdir(d)]

        # See if keyword arguments given, else use defaults.
        if not name_structure:
            name_convention = self.__def_name_structure__
        else:
            assert set(name_structure.keys()) == set(self.__def_name_structure__.keys())
        if not directory:
            directory = self.__def_file_list__

        # Try to load the data files in the directory
        try:
            file_list = listdir_fp(directory)
            self.__default_constructor__(file_list, name_convention)
        except ValueError:
            print('No files found in directory!')
        except FileNotFoundError as e:
            print('Got error: FileNotFound: ' + e.filename)
        except (TypeError, IndexError):
            print('Got a TypeError or IndexError: perchance file_list is empty?')
        except AssertionError:
            print('Got an AssertionError: perchance name_structure is faulty')

    def from_manual(self):
        """ Constructor to load data while interactively specifying the naming convention. Calls default_constructor().

            This will use TkInter dialog boxes to interactively let the user specify the naming convention in use. The
            function will then generate the necessary dictionary object to be used with default constructor.
        """
        from tkinter import Tk
        import tsCluster.guiLoader as guiLoader
        root = Tk()
        gui = guiLoader.LoadGroups(root)
        root.mainloop()
        self.__default_constructor__(gui.fileList, gui.nameStruc)

    def mod_samples(self, op, idxs):
        """ Function to add or remove samples from analysis. 
        
            Arguments:
                op -- The operation to perform, 'add' or 'rem' (remove). Type is string.
                idxs -- The names of the subjects that you want to add or remove from analysis. 
                    Type is a list of strings
        """
        if not type(idxs) == list:
            print('Error: mod_samples must be called with a list of sample names!')
        if op == 'rem':
            for idx in idxs:
                idx_list = [sample for sample in self.samples if sample['Name'] == idx]
                for item in idx_list:
                    self.deleted_samples.append(item)
                    self.samples.remove(item)
        if op == 'add':
            for idx in idxs:
                idx_list = [sample for sample in self.samples if sample['Name'] == idx]
                for item in idx_list:
                    self.samples.append(item)
                    self.deleted_samples.remove(item)
        self.reset()

    def mod_data_rem(self, idx_session, idx_run=None):
        """ A function to remove collections of data from analysis. 
        
            Arguments:
                idx_session: The largest collection that can be removed is a session. This argument must be provided.
                    Type is string.
                idx_run: One can be more specific than a session by also specifying a run in that session. If no run is
                    specified then all data corresponding to given session is removed. 
            
            Note that session from all locations will be removed; and also technically a location is the largest
            collection that can be removed but that is done in another function.
        """
        if not idx_session and not idx_run:
            print('Error: Nothing to do!')

        if not idx_run:  # Removing an entire session
            if idx_session in self.sessions:
                temp = [sample for sample in self.samples if sample['Session'] == idx_session]
                self.samples = [sample for sample in self.samples if sample['Session'] != idx_session]
                if not self.removed_samples:
                    self.removed_samples = temp
                else:
                    for sample in temp:
                        self.removed_samples.append(sample)

                # Update removal counter for samples from Champaign.
                try:
                    self.removed_counter['chm-' + idx_session] = set(
                        [sample['Run'] for sample in self.removed_samples if sample['Location'] == 'chm'])
                    self.data_counter.pop('chm-' + idx_session)
                except KeyError:  # Probably because no samples recorded at Champaign are in data set.
                    pass

                # Update removal counter for samples from San Antonio.
                try:
                    self.removed_counter['san-' + idx_session] = set(
                        [sample['Run'] for sample in self.removed_samples if sample['Location'] == 'san'])
                    self.data_counter.pop('san-' + idx_session)
                except KeyError:  # Probably because no samples recorded in San Antonio are in data set
                    pass
                self.reset()
            else:
                raise LookupError
        elif idx_session in self.sessions and idx_run in self.runs:  # Only removing a single run.
            temp1 = [sample for sample in self.samples if sample['Session'] != idx_session]
            temp2 = [sample for sample in self.samples if sample['Session'] == idx_session and sample['Run'] != idx_run]
            temp3 = [sample for sample in self.samples if sample['Session'] == idx_session and sample['Run'] == idx_run]
            self.samples = temp1 + temp2
            if not self.removed_samples:
                self.removed_samples = temp3
            else:
                for sample in temp3:
                    self.removed_samples.append(sample)

            # Update data and removal counter for samples from Champaign. The counters are dictionaries that have as
            # keys location + session (e.g. 'chm-s1') and as values the runs in the session e.g (['run1', 'run2'])
            try:
                self.removed_counter['chm-' + idx_session] = set(
                    [sample['Run'] for sample in self.removed_samples if sample['Location'] == 'chm'])
                self.data_counter['chm-' + idx_session] = set(
                    [sample['Run'] for sample in self.samples if sample['Session'] == idx_session
                     and sample['Location'] == 'chm'])

                # Remove empty sets
                if not self.data_counter['chm-' + idx_session]:
                    self.data_counter.pop('chm-' + idx_session)
                if not self.removed_counter['chm-' + idx_session]:
                    self.removed_counter.pop('chm-' + idx_session)
            except KeyError:  # Probably because no samples recorded at Champaign are in data set.
                pass

            # Update data and removal counter for samples from San Antonio
            try:
                self.removed_counter['san-' + idx_session] = set([sample['Run'] for sample in self.removed_samples
                                                                  if sample['Location'] == 'san'])
                self.data_counter['san-' + idx_session] = set([sample['Run'] for sample
                                                               in self.samples if sample['Session'] == idx_session
                                                               and sample['Location'] == 'san'])
                # Remove empty sets
                if not self.data_counter['san-' + idx_session]:
                    self.data_counter.pop('san-' + idx_session)
                if not self.removed_counter['san-' + idx_session]:
                    self.removed_counter.pop('san-' + idx_session)
            except KeyError:  # Probably because no samples recorded in San Antonio are in data set
                pass
            self.reset()
        else:
            raise LookupError

    def mod_data_add(self, idx_session, idx_run=None):
        """ Function to add session or run level data back in to the analysis.

            Arguments:
                idx_session -- The session identifier for the session that is to be removed.
            Keyword arguments:
                idx_run -- The specific run in idx_session to be removed.

            If idx_run is not provided, all matches of idx_session amongst the removed samples are added back in to
            the analysis (regardless of location)
        """
        if not idx_run:  # No run specified
            if any(idx_session in key_list for key_list in list(self.removed_counter.keys())):
                temp = [sample for sample in self.removed_samples if sample['Session'] == idx_session]
                self.removed_samples = [sample for sample in self.removed_samples if
                                        sample['Session'] != idx_session]
                for item in temp:
                    self.samples.append(item)

                    # Session level data was added back in, so keys must exist in counter as is.
                    try:
                        self.data_counter['chm-' + idx_session] = self.removed_counter.pop('chm-' + idx_session)
                    except KeyError:
                        pass
                    try:
                        self.data_counter['san-' + idx_session] = self.removed_counter.pop('san-' + idx_session)
                    except KeyError:
                        pass
                self.reset()
            else:
                print('Error! Specified session not found amongst removed samples')
                raise LookupError
        elif any(idx_session in key_list for key_list in list(self.removed_counter.keys())) and \
                (idx_run in self.removed_counter['chm-' + idx_session] or idx_run in self.removed_counter[
                    'san-' + idx_session]):
            temp = [sample for sample in self.removed_samples
                    if sample['Session'] == idx_session and sample['Run'] == idx_run]
            for item in temp:
                self.samples.append(item)
                self.removed_samples.remove(item)
            try:
                # Update removed counter
                self.removed_counter['san-' + idx_session] = set([sample['Run'] for sample in self.removed_samples
                                                                  if sample['Location'] == 'san'])
                # Delete empty sets
                if not self.removed_counter['san-' + idx_session]:
                    self.removed_counter.pop('san-' + idx_session)
                # Update data counter
                self.data_counter['san-' + idx_session] = set([sample['Run'] for sample in self.samples
                                                               if sample['Session'] == idx_session and
                                                               sample['Location'] == 'san'])
                # Data counter is unlikely to have empty sets
            except KeyError:
                pass
            try:
                # Update removed counter
                self.removed_counter['chm-' + idx_session] = set([sample['Run'] for sample in self.removed_samples
                                                                  if sample['Location'] == 'chm'])
                # Delete empty sets
                if not self.removed_counter['chm-' + idx_session]:
                    self.removed_counter.pop('chm-' + idx_session)
                # Update data counter
                self.data_counter['chm-' + idx_session] = set([sample['Run'] for sample in self.samples
                                                               if sample['Session'] == idx_session and
                                                               sample['Location'] == 'chm'])
            except KeyError:
                pass
            self.reset()
        else:
            raise LookupError

    def mod_loc(self, op, loc):
        """ Function to add or remove samples from a location to the data set.

            Arguments:
                op: Operation to perform, either we 'add' or we 'rem' (remove)
                loc: The location to be removed, usually 'chm' or 'san'
        """
        if op == 'rem':
            if loc in set([sample['Location'] for sample in self.samples]):  # Check if location exists in samples
                temp = [sample for sample in self.samples if sample['Location'] == loc]
                self.samples = [sample for sample in self.samples if sample['Location'] != loc]
                if not self.removed_samples:
                    self.removed_samples = temp
                else:
                    for item in temp:
                        self.removed_samples.append(item)
                self.reset()
                to_pop = []

                # Update the data counter, since we removed a location, we need to remove those keys from data counter
                # and add them to removed counter
                for key in self.data_counter.keys():
                    if loc in key:
                        to_pop.append(key)

                # Update the removed counter
                for key in to_pop:
                    self.removed_counter[key] = self.data_counter.pop(key)
            else:
                raise LookupError
        elif op == 'add':
            if loc in set([sample['Location'] for sample in self.removed_samples]):
                temp = [sample for sample in self.removed_samples if sample['Location'] == loc]
                for item in temp:
                    self.samples.append(item)
                    self.removed_samples.remove(item)
                self.reset()
                to_pop = []

                # Update the removed counter
                for key in self.removed_counter.keys():
                    if loc in key:
                        to_pop.append(key)

                # Update the data counter
                for key in to_pop:
                    self.data_counter[key] = self.removed_counter.pop(key)
            else:
                raise LookupError
        else:
            print('Error! Location not in data set')
            raise LookupError

    def mod_group(self, op, group):
        """ Function to add or remove a particular group from analysis. Useful for test train data sets.

            Arguments:
                op -- String 'add' for adding and 'rem' for removing.
                group -- String specifying which group to add or remove.

            Note that there is no removed_group counter. Instead self.groups shows groups that are present.
        """
        if op == 'rem':
            if group in self.groups:
                temp = [sample for sample in self.samples if sample['Group'] == group]
                self.samples = [sample for sample in self.samples if sample['Group'] != group]
                if not self.removed_samples:
                    self.removed_samples = temp
                else:
                    for item in temp:
                        self.removed_samples.append(item)
                self.reset()
            else:
                raise LookupError
        if op == 'add':
            if group in set([sample['Group'] for sample in self.removed_samples]):
                temp = [sample for sample in self.removed_samples if sample['Group'] == group]
                for item in temp:
                    self.removed_samples.remove(item)
                    self.samples.append(item)
                self.reset()
            else:
                raise LookupError

    def edit_time_series(self, sub, run, session, edit_idx):
        """ Function to edit a specific time series entity. MUST CALL recompute manually.

            Arguments:
                sub -- Name of subject to remove. String.
                run -- Specific run of that subject to edit. String.
                session -- The session the specified run belongs to. String.
                edit_idx -- Numpy s_ objects corresponding to slices in the time series that one wishes to KEEP.

            All arguments are required to prevent accidental edits to time series data.
        """

        for sample in self.samples:
            if sample['Name'] == sub and sample['Run'] == run and sample['Session'] == session:
                if type(edit_idx) == slice:
                    sample['TimeSeries'] = sample['TimeSeries'][:, edit_idx]
                elif type(edit_idx) == tuple:
                    first = True
                    for idx in edit_idx:
                        if first:
                            temp = sample['TimeSeries'][:, idx]
                            first = False
                        else:
                            temp = np.concatenate((temp, sample['TimeSeries'][:, idx]), axis=1)
                    sample['TimeSeries'] = temp
                else:
                    print('Error: Given slicing is not comprehensible!')

    def remove_roi(self, roi):
        """ Function to remove a region of interest from the data set (from all samples). Must call recompute.

            Arguments:
                roi -- The roi to be removed specified as a string.

            Function is destructive, a roi removed from analysis cannot be added back in.  The instance's recompute
            method must be called manually.
        """

        if roi in self.rois:
            idx = self.rois.index(roi)
            for sample in self.samples:
                sample['TimeSeries'] = np.delete(sample['TimeSeries'], obj=idx, axis=0)
            # self.recompute()
            self.rois.remove(roi)
        else:
            raise LookupError
