from tsCluster import tsBase as Base
import numpy as np
from numpy import random
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as mpl


class TsCluster(Base.TsBase):
    """ A class to interactively examine time series data collected in runs and sessions.

        This extends the TsBase (time series base) class by adding plotting routines and other features.

        Methods:
            get_projected_labelled: A method to get data along with its group labels
            get_test_train: A method to get data in testing and training splits along with labels
            scatter: A method to see a scatter plot by group of data projected to 2D
            show_time_series: A method to examine the multi-variate time series of a subject
            show_mat: A method to visualize the various matrices associated with the time series data
            plot_auto_corr: A function that shows the auto-correlation function
    """

    def __init__(self):
        super().__init__()

    def __do_partition__(self, named_data, key, value):
        """ Function that partitions data (all of the same group) into test_train pairs.

            Arguments:
                named_data : A list of tuples with name, and data
                key :  A list of groups (i.e. a list of strings)
                value: A float specifying how much of a particular group to include in training.
        """
        train_label, train_data, test_label, test_data, k, j = list(), list(), list(), list(), 1, 1
        (names, data) = zip(*named_data)

        # We don't want the same subject showing up in test and train samples
        unique_names = set(names)
        threshold = int(value * len(unique_names))
        o_names = list()
        for a_name in unique_names:
            idxs = [idx for idx, name in enumerate(names) if a_name == name]
            if k < threshold:       # k, j make sure the train and test samples are equal sized.
                for idx in idxs:
                    train_data.append(data[idx])
                    train_label.append(key)
                    o_names.append(k - 1)
                k += 1
            elif j < threshold:
                for idx in idxs:
                    test_data.append(data[idx])
                    test_label.append(key)
                j += 1
        return train_data, train_label, o_names, test_data, test_label

    def get_projected_labeled(self, n=12, dec=None):
        """ Function to get projected data by group.

            Keyword arguments:
                n -- The number of dimensions to project down to
                dec -- A decorator function to for additional processing that acts on lists.

            Returns a list or dec(list)
        """
        data_grouped = list()
        for group in self.groups:
            temp_var = [self.samples[i]["FlatULM"] for i in range(len(self.samples))
                        if self.samples[i]["Group"] == group]
            data = np.matmul(np.asarray(temp_var), self.projectors[:, 0:n])
            (m, _) = data.shape
            for row in range(m):
                data_grouped.append((group, data[row, :]))
        if dec:
            return dec(data_grouped)
        else:
            return data_grouped

    def get_test_train(self, n=12, mat='FlatULM', project=False, **kwargs):
        """ Function to generate test train data sets.

            Keyword arguments:
                n -- The number of projections to project down to, has effect only if project=True and mat='FlatULM'
                mat -- The type of matrix to get, either 'FlatULM','FlatSLM' or 'FlatCM'
                project -- Whether to perform dimension reduction using PCA, default is False

            Function also takes string and a float between 0 and 1 as keyword argument pairs. The float specifies the
            percentage of samples of the group specified by the string to have in the training data set. The test data
            set will be size matched to the training data set. If not specified then will use 0.5 as defaults.
        """
        test_label, train_label, test_data, train_data, train_names = list(), list(), list(), list(), list()

        if kwargs is not None:
            percentages = kwargs
        else:
            percentages = dict.fromkeys(self.groups, 0.5)

        for key, value in percentages.items():
            try:
                temp = [(self.samples[i]['Name'], self.samples[i][mat]) for i in range(len(self.samples))
                        if self.samples[i]["Group"] == key]
            except KeyError:
                print('Error! "mat" must be "FlatULM", "FlatSLM" or "FlatCM"')
                raise KeyError
            sorted_temp = sorted(temp, key=lambda x: x[0])  # Sort them so the same subject is consecutive in the list
            names, temp_var = zip(*sorted_temp)
            if len(temp_var) == 0:
                print('Error! Group not found!')
                raise LookupError

            # To project or not to project?
            if project and mat == 'ULM':
                data = np.matmul(np.asarray(temp_var), self.projectors[:, 0:n])
            elif project and mat != 'ULM':
                print('Error! Projectors only defined for lead matrices')
                raise KeyError
            elif not project:
                data = np.asarray(temp_var)
            else:
                raise LookupError

            # Call the partitioning function.
            gp_train, gp_train_label, g_names, gp_test, gp_test_label = self.__do_partition__(zip(names, data),
                                                                                              key, value)

            # Append data and label
            train_data.extend(gp_train)
            train_label.extend(gp_train_label)
            test_data.extend(gp_test)
            test_label.extend(gp_test_label)
            train_names.extend(g_names)
        return train_label, train_data, train_names, test_label, test_data

    def scatter(self, fig_num=1):
        """ Function to make a 2D scatter plot of data by group using covariance structure to project.

            Keyword arguments:
                fig -- Specifies a figure number for call to matplotlib module
        """
        import mpldatacursor

        def plotted_artists(ax):
            artists = (ax.lines + ax.patches + ax.collections + ax.images + ax.containers)
            return artists

        def formatter(**kwargs):
            return kwargs['point_label'].pop()

        col = ['b', 'g', 'r', 'c', 'm', 'y', 'k']   # List of colors
        used = []                                   # List of used colors
        per_scatter_label = list()                  # We need a list of labels per call to scatter

        fig = mpl.figure(fig_num)
        ax = mpl.axes()

        for group in self.groups:
            by_group = [sample for sample in self.samples if sample["Group"] == group]
            (label, data) = zip(*[(sample['Name'] + '\n' + sample['Session'] + '-' + sample['Run'] + '-' +
                                   sample['Location'] + '\n' + sample['Group'],
                                   sample['FlatULM']) for sample in by_group])
            xy = np.matmul(np.asarray(data), self.projectors[:, 0:2])  # Project down to 2 Dimension
            c = random.choice(col)
            while c in used:
                c = random.choice(col)
            used.append(c)
            ax.scatter(xy[:, 0], xy[:, 1], c=c, label=group)
            per_scatter_label.append(label)
            mpl.show(block=False)

        axes = [ax for ax in fig.axes]
        scatters = [artist for ax in axes for artist in plotted_artists(ax)]

        point_labels = dict(zip(scatters, per_scatter_label))
        mpldatacursor.datacursor(formatter=formatter, point_labels=point_labels)
        ax.legend()

        ax.grid(True)
        mpl.xlabel('Principal direction 1')
        mpl.ylabel('Principal direction 2')
        mpl.title('Scatter plot')
        mpl.show()

    def show_time_series(self, sub, run=None, session=None, rois=None):
        """ Function to display the time series data.

            Arguments:
                sub -- The name of the subject whose time series data should be displayed.
            Keyword arguments:
                run -- Which run the plot should correspond to
                session -- Which session the plot should correspond to

            If run and session are not specified, plots it for all matches of subject name.
        """
        import mpldatacursor

        def plotter(sample):
            """ Helper function that does actually plotting. """
            if not rois:
                mpl.figure()
                for index, row in enumerate(sample['TimeSeries'].tolist()):
                    mpl.plot(row, label=self.rois[index])
            else:
                mpl.figure()
                for roi in rois:
                    try:
                        idx = self.rois.index(roi)
                        mpl.plot(sample['TimeSeries'][idx,:].tolist(), label=roi)
                    except ValueError:
                        print('Error! ROI ' + roi + ' not found!')
                        raise KeyError
            mpl.title(sample['Name'] + '-' + sample['Session'] + '-' + sample['Run'])
            mpl.xlabel('Time')
            mpldatacursor.datacursor(formatter='{label}'.format)
            mpl.show()

        # Get all samples with name match
        temp = [sample for sample in self.samples if sample['Name'] == sub]

        if temp is None:
            print('Error! No such time series found!')
            raise LookupError

        # Depending on arguments plot appropriately
        for sample in temp:
            if not run and not session:
                if sample['Name'] == sub:
                    plotter(sample)
            elif not session:
                if sample['Name'] == sub and sample['Run'] == run:
                    plotter(sample)
            elif not run:
                if sample['Name'] == sub and sample['Session'] == session:
                    plotter(sample)
            else:
                if sample['Name'] == sub and sample['Run'] == run and sample['Session'] == session:
                    plotter(sample)

    def show_mat(self, sub, mat='ULM', session=None, run=None):
        """ A function to visualize the matrices associated with samples.

            Arguments:
                sub -- The subject identifier as a string

            Keyword arguments:
                mat -- The matrix to display. A string which must be one of 'ULM', 'CM' and 'SLM'
                session -- Session identifier for the matrix of interest
                run -- Run in the session specified.

            If keyword arguments are not specified function displays the unsorted lead matrix for all instances of the
            subject that it found.
        """
        import mpldatacursor
        matplotlib.rcParams.update({'font.size': 16})

        def plot(sample, matrix):
            """ Helper function that does the actual plotting.

                Arguments:
                    sample -- An element of self.samples
                    matrix -- A string specifying which matrix to visualize. Must be 'ULM', 'SLM' of 'CM'
            """
            mat_name = {'CM': 'Correlation Matrix - ', 'ULM': 'Lead Matrix - ', 'SLM': 'Sorted Lead Matrix - '}
            mpl.figure()
            try:
                mpl.imshow(sample[matrix], interpolation=None)
            except KeyError:
                print('Error! Matrix not found. "mat" must be ULM, CM or SLM')
            mpl.title(mat_name[matrix] + sample['Name'] + ':' + sample['Session'] + '-' + sample['Run'] +
                      '(' + sample['Location'] + ')')
            mpl.colorbar()
            mpldatacursor.datacursor(bbox=dict(alpha=1, fc='w'), formatter=label_point)

        def label_point(**kwargs):
            """ A helper function for `mpldatacursor`.
            """
            if kwargs is not None:
                try:
                    return 'row = ' + self.rois[kwargs['i']] + '\ncol = ' + self.rois[kwargs['j']]
                except KeyError:
                    pass

        # Get all name matches
        sub_samples = [sample for sample in self.samples if sample['Name'] == sub]

        if sub_samples is None:
            print('Error! Subject not found')
        else:
            # Plot as per input arguments
            for sample in sub_samples:
                if session is None:
                    plot(sample, mat)
                elif sample['Session'] == session and run is None:
                    plot(sample, mat)
                elif sample['Session'] == session and sample['Run'] == run:
                    plot(sample, mat)
        mpl.show(block=True)
        matplotlib.rcdefaults()

    def plot_auto_corr(self, sub, roi=None, norm=False):
        """ Function to plot the auto-correlation of a single time series.

            Arguments:
                sub -- Subject identifier as string
            Keyword arguments:
                roi -- An integer or string specifying the region of interest to plot auto-correlation over. If not
                    specified then it is 0.
                norm -- A boolean specifying whether to normalize the time series or not before auto-correlation is
                    calculated. Default is False.
        """

        # Convert roi index to roi name and vice versa
        if not roi:
            roi_idx = 0
            roi = self.rois[0]
        elif type(roi) is int:
            roi_idx = roi
            if not (0 <= roi_idx <= len(self.rois)):
                print('Error! ROI index not found')
                raise LookupError
            roi = self.rois[roi_idx]
        elif type(roi) is str:
            try:
                roi_idx = self.rois.index(roi)
            except ValueError:
                print('Error! ROI index not found')
                raise LookupError
        else:
            print('Error! name must be ROI index or a ROI name')
            raise LookupError

        # Import norm functions as necessary
        from tsCluster.cyclicAnalysis import quad_norm, tv_norm
        norms = {'tv': tv_norm, 'sqr': quad_norm}

        def mean_center(z):
            """Mean centers the time series

                Arguments:
                    Z -- A matrix or a vector. If matrix then each row is a variable, and columns are instances at
                        different times.
            """
            return z - np.mean(z, axis=1)[:, np.newaxis]

        def serial_corr(wave, lag=1):
            """ Computes serial correlation given a wave and a lag amount.

                Arguments:
                    wave -- A single variable time series, i.e. a vector.
                Keyword arguments:
                    lag -- How much to lag the time series by. Default is 1.
            """
            n = len(wave)
            y1 = wave[lag:]
            y2 = wave[:n - lag]
            corr = np.dot(y1, y2) / n
            return corr

        def auto_corr(wave):
            """ Computes the auto correlation from lag = 0 to lag = len(wave) -1

                Arguments:
                    wave -- A vector
            """
            time_lags = range(len(wave) - 1)
            correlates = [serial_corr(wave, lag) for lag in time_lags]
            return time_lags, correlates

        samples = [sample for sample in self.samples if sample['Name'] == sub]

        for sample in samples:
            item = sample['TimeSeries'][roi_idx, np.newaxis]

            if not norm:
                std_item = mean_center(item)
            else:
                std_item = norms[self.norm](item)
            lags, corrs = auto_corr(std_item.flatten().tolist())
            mpl.figure()
            mpl.stem(lags, corrs)
            mpl.title(sub + ' in ' + sample['Run'] + ' of ' + 'Session-' + sample['Session'] + '\n' + roi)
        mpl.show(block=True)
