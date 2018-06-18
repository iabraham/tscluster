from tsCluster import tsBase as tsc
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
import gzip
import pickle

# all_data = tsc.TsCluster()
# all_data.from_directory()
# all_data.remove_roi('lcuneus')  # Desctructive, cannot add it back in
# all_data.remove_roi('rcuneus')
# all_data.recompute()
# print(all_data)
# print(all_data.rois)
# print(all_data.samples[0]['TimeSeries'].shape)
# all_data.save_to_file()

file_name = './binaries/data_binary_roi33.bin'
with gzip.open(file_name, 'rb') as file:
    all_data = pickle.load(file)
# ============================== HOUSEKEEPING

# 2. Un-fixable Motion removal
remList = ['sub_157', 'sub_158', 'sub_158', 'sub_132', 'sub_120', 'sub_135', 'sub_162', 'sub_169',
           'sub_182', 'sub_188', 'sub_210', 'sub_217', 'sub_903']
all_data.mod_samples(op='rem', idxs=remList)
all_data.mod_data_rem(idx_session='s1', idx_run='run3')

to_sort = [(item['Group'], item['Name'], item['SLM']) for item in all_data.samples]
sorted_slm = sorted(to_sort, key=lambda x: (x[0], x[1]))
just_slm = [item[2] for item in sorted_slm]
just_label = [item[0] for item in sorted_slm]  #Piece of trust going on here. 
data = dict()
data['labels'] = just_label
data['data'] = np.asarray(just_slm)

print('Shape of data is: ' + str(data['data'].shape))
factored = parafac(data['data'], rank=1089)
approx = tl.kruskal_to_tensor(factored)
error = np.square(approx - just_slm).sum()
print(error)

with open('factored_full_slm.bin', 'wb') as file:
	pickle.dump(factored, file, protocol=pickle.HIGHEST_PROTOCOL)

with open('data_w_labels_slm.bin', 'wb') as file:
	pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

# ts_data = dict()
# group_ranks = {'hl': [792], 'ctr': [792, 1980], 'tin': [792, 1584], 'tin_hl': [792, 3762]}
# group_ranks = {'hl': [792, 1089], 'ctr': [1089], 'tin': [1089], 'tin_hl': [1089]}
# factored = dict()
# random_data = dict()
# uts_data = dict()

# for group in all_data.groups:
#    uts_data[group] = [(sample['Name'],sample['TimeSeries']) for sample in all_data.samples if sample['Group']==group]

#for group in uts_data:
#    sorted_data = sorted(uts_data[group], key=lambda x: x[0])
#    ts_data[group] = np.asarray([item[1] for item in sorted_data])

# # ts_data['hl'] = np.asarray([sample['TimeSeries'] for sample in all_data.samples if sample['Group'] == 'hl'])
#
# for group in ts_data:
#    print('Doing group: ' + group)
#    factored[group] = dict()
#    random_data[group] = dict()
#    print('Shape is ' + str(ts_data[group].shape))
#    for rank in group_ranks[group]:
#        print('Doing (' + str(rank) + ')')
#        random_data[group][str(rank)] = np.random.random(ts_data[group].shape)
#        factored[group]['rank'+str(rank)] = parafac(ts_data[group], rank=rank)
#        factored[group]['rand'+str(rank)] = parafac(random_data[group][str(rank)], rank=rank)
#
# with open('facsorted_by_groups.bin', 'wb') as file:
#    pickle.dump(factored, file, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('random_lm_by_groups.bin', 'wb') as file:
#    pickle.dump(random_data, file, protocol=pickle.HIGHEST_PROTOCOL)

##############################
# factored = parafac(ts_data['ctr'], rank=1980)
# import matplotlib.pyplot as plt
# plt.figure() 
# plt.plot(factored[0][:,0]) 
# plt.show()
