import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

def download_modelnet40(DATA_DIR):
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        os.mkdir(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048'))
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

def check_ln_modelnet40(DATA_DIR):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        # /home/chengle/worksapce/data
    dataset_dir = '/home/chengle/worksapce/data/modelnet40_ply_hdf5_2048'
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        os.system('ln -s ' + dataset_dir + ' ' + DATA_DIR)

def load_data_cls(DATA_DIR, partition):
    # download_modelnet40()
    check_ln_modelnet40(DATA_DIR)
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', '*%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    # xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

class ModelNet40(Dataset):
    def __init__(self, root, num_points=1024, partition='train'):
        self.data, self.label = load_data_cls(root, partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            #pointcloud = rotate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        # pointcloud[:, 0:3] = pc_normalize(pointcloud[:, 0:3])
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ScanObjectNN(Dataset):
    def __init__(self, root, partition='training', num_points=1024):
        super().__init__()
        self.data, self.label = self.load_scanobjectnn_data(root, partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'training':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

    def download_scanobjectnn():
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        if not os.path.exists(os.path.join(DATA_DIR, 'h5_files')):
            # note that this link only contains the hardest perturbed variant (PB_T50_RS).
            # for full versions, consider the following link.
            www = 'https://web.northeastern.edu/smilelab/xuma/datasets/h5_files.zip'
            # www = 'http://103.24.77.34/scanobjectnn/h5_files.zip'
            zipfile = os.path.basename(www)
            os.system('wget %s  --no-check-certificate; unzip %s' % (www, zipfile))
            os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
            os.system('rm %s' % (zipfile))

    def check_ln_scanobjectnn(self, DATA_DIR):
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        dataset_dir = '/home/chengle/worksapce/data/h5_files'
        if not os.path.exists(os.path.join(DATA_DIR, 'h5_files')):
            os.system('ln -s ' + dataset_dir + ' ' + DATA_DIR)
            
    def load_scanobjectnn_data(self, DATA_DIR, partition):
        # download()
        # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.check_ln_scanobjectnn(DATA_DIR)
        BASE_DIR = DATA_DIR
        all_data = []
        all_label = []
        if partition == 'train':
            partition = 'training'

        h5_name = BASE_DIR + '/h5_files/main_split/' + partition + '_objectdataset_augmentedrot_scale75.h5'
        f = h5py.File(h5_name, mode="r")
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return all_data, all_label

def dataset_get(DATA_PATH, model_name=None, num_points=1024):
    if model_name == 'modelnet40':
        TRAIN_DATASET = ModelNet40(root=DATA_PATH, partition='train', num_points=num_points)
        TEST_DATASET = ModelNet40(root=DATA_PATH, partition='test', num_points=num_points)
    elif model_name == 'scanobjectnn':
        TRAIN_DATASET = ScanObjectNN(root=DATA_PATH, partition='training', num_points=num_points)
        TEST_DATASET = ScanObjectNN(root=DATA_PATH, partition='test', num_points=num_points)
    return TRAIN_DATASET,TEST_DATASET
