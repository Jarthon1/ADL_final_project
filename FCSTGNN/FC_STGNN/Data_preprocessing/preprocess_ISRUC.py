import numpy as np
import scipy.io as scio
from os import path
from scipy import signal
import h5py

path_Extracted = '/home/ubuntu/deep-sleep-pytorch/data/extracted_channels/ISRUC_mat_data'
path_RawData   = '/home/ubuntu/deep-sleep-pytorch/data'
path_output    = '/home/ubuntu/FCSTGNN/FC_STGNN/data'
channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1',
            'LOC_A2', 'ROC_A1','X1', 'X2']

# test1 = [0,0,0,0,0,0,0,5]
# test2 = "greetings"
# test3 = [[0,0],[1,1]]

# with open(path.join(path_output, 'test_savez.npz'), 'wb') as f:
#     np.savez(f, data1 = test1, data2 = test2, data3 = test3)



def read_psg(path_Extracted, sub_id, channels, resample=3000):
    psg = scio.loadmat(path.join(path_Extracted, 'subject%d.mat' % (sub_id)))
    psg_use = []
    for c in channels:
        psg_use.append(
            np.expand_dims(signal.resample(psg[c], resample, axis=-1), 1))
    psg_use = np.concatenate(psg_use, axis=1)
    return psg_use


def read_label(path_RawData, sub_id, ignore=30):
    label = []
    with open(path.join(path_RawData, '%d/%d_1.txt' % (sub_id, sub_id))) as f:
        s = f.readline()
        while True:
            a = s.replace('\n', '')
            label.append(int(a))
            s = f.readline()
            if s == '' or s == '\n':
                break
    return np.array(label[:-ignore])


'''
output:
    save to $path_output/ISRUC_S3.npz:
        Fold_data:  [k-fold] list, each element is [N,V,T]
        Fold_label: [k-fold] list, each element is [N,C]
        Fold_len:   [k-fold] list
'''

fold_label = []
fold_psg = []
fold_len = []

for sub in range(1, 60):
    print('Read subject', sub)
    label = read_label(path_RawData, sub)
    psg = read_psg(path_Extracted, sub, channels)
    print('Subject', sub, ':', label.shape, psg.shape)
    assert len(label) == len(psg)

    # in ISRUC, 0-Wake, 1-N1, 2-N2, 3-N3, 5-REM
    label[label==5] = 4  # make 4 correspond to REM
    fold_label.append(np.eye(5)[label]) # This is just a way to get a one hot vector for the label
    fold_psg.append(psg)
    fold_len.append(len(label))
print('Preprocess over.')

print('Trying to save')

with h5py.File(path.join(path_output, 'ISRUC_S3.h5'), 'w') as f:
    fold_data_group = f.create_group('Fold_data')
    for i, data in enumerate(fold_psg):
        fold_data_group.create_dataset(str(i), data=data)

    fold_label_group = f.create_group('Fold_label')
    for i, label in enumerate(fold_label):
        fold_label_group.create_dataset(str(i), data=label)

    f.create_dataset('Fold_len', data=np.array(fold_len))

print('Saved to', path.join(path_output, 'ISRUC_S3.h5'))