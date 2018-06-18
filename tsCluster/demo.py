from tsCluster import *
from numpy import s_
from copy import deepcopy
from tkinter import Tk, filedialog
from matplotlib import pyplot as mpl

# The dictionary below defines the conventions used in the data files. The data contains all groups, all locations,
# all runs and so on. So we specify this in the dictionary. The file names are like:
#
#           s1_run1_aus_ctr_sub_101.txt
#
# here 'sub' is the short name, if file was named s1_test1_aus_ctr_sample_101.txt then the short name would be 'sample'
# and we would change: 'runs': ['test1', 'run2', 'run3']. This allows for flexible use of this module with different
# naming conventions
#
# We can also specify options for lead matrix creation. The allowed norms are 'tv' and 'sqr', allowed trend removals are
# 'none', 'linear', 'bias'

convention = {'groups': ['ctr', 'tin+hl', 'tin', 'hl'], 'locs': ['aus', 'chm'], 'runs': ['run1', 'run2', 'run3'],
              'sessions': ['s1', 's2'], 'short_name': 'sub', 'file_ext': '.txt', 'norm': 'tv', 'trend_removal': 'none'}

# Specify the directory holding the data
directory = './data'

# Load the data
all_data = TsCluster()  # Initialize instance
all_data.from_directory(name_structure=convention, directory=directory)  # Call constructor

# The above arguments are default, so if the folder structure is exactly as this repo, then above constructor call
# is equivalent to:         all_data.from_directory()
#
# One can also define the dictionary and the data files manually using GUI
# (gets tedious if you have to do it many times)

# all_data_manual = TsCluster()
# all_data_manual.from_manual()


print(all_data)  # Get summary statistics

# Remove subjects
rem_subject_list = ['sub_157', 'sub_158', 'sub_158', 'sub_162', 'sub_169', 'sub_182', 'sub_188', 'sub_210',
                    'sub_217', 'sub_903']  # Repeated sub_ids from Austin had their first digit changed to 9

all_data.mod_samples('rem', rem_subject_list)
print(all_data)  # Confirm removal

# Remove music run
all_data.mod_data(op='rem', idx_session='s1', idx_run='run3')

# Usually the constructor will initialize the name of the instance with the variable it is being bound to, but we can
# manually change it to show whats been done.
all_data.name = 'all_data after removing music run'
print(all_data)

# You can add the data back in using:
#
#           all_data.mod_data(op='add', idx_session='s1', idx_run='run3')
# or
#           all_data.mod_data(op='rem', idx_session='s1')

# Remove Austin data
all_data.mod_loc(op='rem', loc='aus')
all_data.name = 'all_data after removing run3 and austin'
print(all_data)

# View and edit faulty time series
# all_data.show_time_series(sub='sub_132', run='run2', session='s2')
all_data.edit_time_series(sub='sub_132', run='run2', session='s2', edit_idx=s_[0:180])
# all_data.show_time_series(sub='sub_132', run='run2', session='s2')

# all_data.show_time_series(sub='sub_120', run='run1', session='s1')
all_data.edit_time_series(sub='sub_120', run='run1', session='s1', edit_idx=s_[0:290])
# all_data.show_time_series(sub='sub_120', run='run1', session='s1')

# all_data.show_time_series(sub='sub_135', run='run2', session='s1')
all_data.edit_time_series(sub='sub_135', run='run2', session='s1', edit_idx=s_[0:135, 180:230, 180:300])
# all_data.show_time_series(sub='sub_135', run='run2', session='s1')

# Compare to MATLAB scatter to see if same  -YES!
all_data.scatter()

# Returns a list of tuples, each tuple contains label and projected data (as ndarray)
temp1 = all_data.get_projected_labeled(n=12)  # n is the number of dimensions to project down to, default 12

# At this point with all_data.samples is a list of dicts. Using pythons dictionary and list comprehension  returning
# almost any subset of the data is easy. Say we want to get a the time-series corresponding to each group

ts_tin = [sample['TimeSeries'] for sample in all_data.samples if sample['Group'] == 'tin']
ts_ctr = [sample['TimeSeries'] for sample in all_data.samples if sample['Group'] == 'ctr']
ts_tinhl = [sample['TimeSeries'] for sample in all_data.samples if sample['Group'] == 'tin+hl']
ts_hl = [sample['TimeSeries'] for sample in all_data.samples if sample['Group'] == 'hl']

try:
    assert len(ts_ctr) + len(ts_tin) + len(ts_tinhl) + len(ts_hl) == 188
except AssertionError:
    print('Oops!')

# If we want it along with the label
ts_tin_labelled = [(sample['Group'], sample['TimeSeries']) for sample in all_data.samples if sample['Group'] == 'tin']


# Because of this only the get_projected_ method is implemented since the projection vectors change each time data is
# added or removed from the whole set. This method also accepts a optional  decorator that can work on the list of
# tuples. For e.g


def vectorize(list_of_tuples):
    label = list()
    data = list()
    for item in list_of_tuples:
        item_label, item_data = item
        label.append(item_label)
        data.append(item_data.tolist())
    return label, data


# Returns a list of labels and a list containing projected data
label, data = all_data.get_projected_labeled(dec=vectorize)

# Finally we can add or remove groups from the analysis, for example to remove hearing loss samples:
all_data.mod_group(op='rem', group='hl')
all_data.name = 'all_data with hl removed'
print(all_data)

# But note that now:
all_data.mod_group(op='add', group='hl')
all_data.name = 'all_data with hl added back has run3 in it'
print(all_data)

# has run3 in it, so to restore state, we need to do
all_data.mod_data(op='rem', idx_session='s1', idx_run='run3')
all_data.mod_loc(op='rem', loc='aus')
all_data.name = 'restored state'
# again.
print(all_data)

# Point is mixing and matching removals/additions must be done carefully. For this purpose it is good practice to create
# a backup using the deepcopy method
backup = deepcopy(all_data)
backup.name = 'Backup of all_data'
# before embarking on removal/addition business. Samples removed using mod_sample as in the beginning, remain deleted
# (they are held in a different list in the class).

all_data.mod_data(op='rem', idx_session='s1')
all_data.mod_data(op='add', idx_session='s1')  # Now has data from session 1 in Austin
all_data.name = 'all_data with s1 added back, has s1-austin in it'
print(backup)
print(all_data)
# But the deepcopied backup remains unchanged.

# Finally there is a helper function to get labels and data as test/train sets. It lets you specify the percentage of
# samples from a particular group to use for training

train_label1, train_data1, test_label1, test_data1 = all_data.get_test_train(n=12, tin=0.4, ctr=0.5, hl=0.5, tin_hl=0.3)

n_hl_train = len([label for label in train_label1 if label=='tin_hl'])
n_hl_test = len([label for label in test_label1 if label=='tin_hl'])

print(str(n_hl_train/(n_hl_train+n_hl_test)))

# Here n is number of dimensions to project down to for dimension reduction (using PCA), omitting a group will cause
# the constructor to not include a group, this lets us be flexible in generating test/train sets for different
# classes included or not e.g.

train_label2, train_data2, test_label2, test_data2 = all_data.get_test_train(n=12, hl=0.5, ctr=0.5, tin_hl=0.3)

# will not have any data from pure tinnitus subjects
print(train_label2)




