from tsCluster import tsCluster as tsc
import gzip, pickle

# all_data = tsc.TsCluster()
# all_data.from_directory()

file_name = './data_binary_roi33.bin'
print('Starting gzip')
with gzip.open(file_name, 'rb') as file:
    print('Starting pickle')
    all_data = pickle.load(file)
    print('End pickle')

hl_samples = [sample['TimeSeries'] for sample in all_data.samples if sample['Group']=='hl' and sample['Run']!='run3']

with open('./factorized_by_groups.bin', 'rb') as file:
    factor_list = pickle.load(file)

