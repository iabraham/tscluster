from tsCluster import tsCluster as tsc
import gzip, pickle

# all_data = tsc.TsCluster()
# all_data.from_directory()

file_name = './binaries/data_binary_roi33.bin'
print('Starting gzip')
with gzip.open(file_name, 'rb') as file:
    print('Starting pickle')
    all_data = pickle.load(file)
    print('End pickle')

