import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import csv
import scipy.io as sio
from opts import parse_opts
from utils import OsJoin
import codecs
import pickle
opt = parse_opts()
data_type = opt.data_type
csv_dir = OsJoin(opt.data_root_path, 'csv', data_type, opt.category)
def nii_loader(path):
    img_pil = nib.load(path)
    img_arr = np.array(img_pil.get_fdata())
    if len(img_arr.shape) > 3:
        img_arr = np.sum(img_arr, axis=3)
    # img_arr = np.array(img_pil.get_fdata()) change the function get_data() to get_fdata()
    img_arr_cleaned = np.nan_to_num(img_arr)  # Replace NaN with zero and infinity with large finite numbers.

    # if path.split('/')[-1] == 's20090904_03_ZangYF_LSF_LiuZhuo-0003-00001-000128-01.nii' or 's20090526_09_ZangYF_LSF_ShuiCaiKong-0003-00001-000128-01.nii':
    #     img_arr_cleaned.resize((256,256,128))   # resize bad samples
    img_pil = torch.from_numpy(img_arr_cleaned)
    # max_ = torch.max(torch.max(torch.max(img_pil )))
    # img_pil = img_pil / max_
    return img_pil
def dfc_loader(path):
    mat_arr = sio.loadmat(path)
    mat_arr = np.array((mat_arr['dfc_tmp']))
    mat_arr_cleaned = np.nan_to_num(mat_arr)
    mat_arr = torch.from_numpy(mat_arr_cleaned)
    # max_ = np.max(np.max(mat_arr))
    # max_ = torch.max(torch.max(torch.max(mat_arr)))
    # min_ = torch.min(torch.min(torch.min(mat_arr)))
    # mat_arr = (mat_arr-min_)/ max_
    # mat_arr = torch.where(torch.isinf(mat_arr), torch.full_like(mat_arr , 0), mat_arr)
    # mat_arr = torch.where(torch.isnan(mat_arr), torch.full_like(mat_arr, 0), mat_arr)
    return mat_arr

def sfc_loader(path):
    mat_arr = sio.loadmat(path)
    mat_arr = np.array((mat_arr['z_matrix_sub']))
    mat_arr_cleaned = np.nan_to_num(mat_arr)
    mat_arr = torch.from_numpy(mat_arr_cleaned)
    # max_ = torch.max(torch.max(mat_arr))
    # mat_arr = mat_arr / max_
    mat_arr = torch.unsqueeze(mat_arr, 2)
    return mat_arr

def default_loader(path):
    data = np.loadtxt(path)
    # with open(path, 'rb') as f:
    #     data = f.read()
    # with open(path, 'rb') as f:
    #     datadict = pickle.load(f, encoding='latin1')
    #     x = datadict['data']
    # img_pil = nib.load(path)
    # img_arr = np.array(img_pil.get_fdata())
    # if len(img_arr.shape)>3:
    #     img_arr=np.sum(img_arr,axis=3)
    #img_arr = np.array(img_pil.get_fdata()) change the function get_data() to get_fdata()
    data_arr_cleaned = np.nan_to_num(data)
    # max_ = torch.max(torch.max(data_arr_cleaned))# Replace NaN with zero and infinity with large finite numbers.
    # data_arr_cleaned = data_arr_cleaned / max_
    # if path.split('/')[-1] == 's20090904_03_ZangYF_LSF_LiuZhuo-0003-00001-000128-01.nii' or 's20090526_09_ZangYF_LSF_ShuiCaiKong-0003-00001-000128-01.nii':
    #     img_arr_cleaned.resize((256,256,128))   # resize bad samples
    # img_pil = torch.from_numpy(data_arr_cleaned)
    img_pil = torch.from_numpy(data_arr_cleaned)
    # max_ = torch.max(torch.max(torch.max(img_pil )))
    # img_pil = img_pil / max_
    img_pil = torch.unsqueeze(img_pil, 2)
    return img_pil


class TrainSet(Dataset):

    def __init__(self, fold_id, loader = default_loader, dfc_loader=dfc_loader, sfc_loader=sfc_loader, nii_loader=nii_loader):
        with open(csv_dir + '/train_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            #for col in reader:
            '''
            need to complete, the structure is stubborn,
            need to re-open file
            '''
            file_train_alff = ([OsJoin(opt.data_root_path, row[0]) for row in reader])
        with open(csv_dir + '/train_fold%s.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_train_dfc = ([OsJoin(opt.data_root_path, row[1]) for row in reader])
        with open(csv_dir + '/train_fold%s.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_train_fa = ([OsJoin(opt.data_root_path, row[2]) for row in reader])
        with open(csv_dir + '/train_fold%s.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_train_fc = ([OsJoin(opt.data_root_path, row[3]) for row in reader])
        # with open(csv_dir + '/train_fold%s.csv' % str(fold_id), 'r') as csvfile:
        #     reader = csv.reader(csvfile)
        #     file_train_fc = ([OsJoin(opt.data_root_path, row[3]) for row in reader])
        # with open(csv_dir + '/train_fold%s.csv' % str(fold_id), 'r') as csvfile:
        #     reader = csv.reader(csvfile)
        #     file_train_fb = ([OsJoin(opt.data_root_path, row[4]) for row in reader])
        # with open(csv_dir + '/train_fold%s.csv' % str(fold_id), 'r') as csvfile:
        #     reader = csv.reader(csvfile)
        #     file_train_reho = ([OsJoin(opt.data_root_path, row[5]) for row in reader])

        file_train = np.array([file_train_alff, file_train_dfc, file_train_fa, file_train_fc])
            #file_train=[file_train file_train_tmp]
            #row=(row[0:5] for row in reader)
            #file_train_tmp=([OsJoin(opt.data_root_path, row[1]) for row in reader])
        with open(csv_dir + '/train_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_train = [row[4] for row in reader]
        self.image = file_train
        self.label = label_train
        self.loader = loader
        self.dfc_loader = dfc_loader
        self.sfc_loader = sfc_loader
        self.nii_loader = nii_loader
    def __getitem__(self, index):
        fn = self.image[:, index]
        img_arr=[]
        for fea in fn:
            # if "nii" in fea:
            #   img_arr_tmp = self.loader(fea)
            #   img_arr.append(img_arr_tmp)
            if "dfc" in fea:
              img_arr_tmp = self.dfc_loader(fea)
              img_arr.append(img_arr_tmp)
            elif "sfc" in fea:
              img_arr_tmp = self.sfc_loader(fea)
              img_arr.append(img_arr_tmp)
            elif "Alff" in fea:
               img_arr_tmp = self.nii_loader(fea)
               img_arr.append(img_arr_tmp)
            else:
                img_arr_tmp = self.loader(fea)
                img_arr.append(img_arr_tmp)
        #img = self.loader(fn)
        label = self.label[index]
        return img_arr, label

    def __len__(self):
        return len(self.image[1])

class ValidSet(Dataset):
    def __init__(self, fold_id, loader=default_loader, dfc_loader=dfc_loader, sfc_loader=sfc_loader, nii_loader=nii_loader):
        with open(csv_dir + '/val_fold%s.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            # for col in reader:
            '''
            need to complete, the structure is stubborn,
            need to re-open file
            '''
            file_val_alff = ([OsJoin(opt.data_root_path, row[0]) for row in reader])
        with open(csv_dir + '/val_fold%s.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_val_dfc = ([OsJoin(opt.data_root_path, row[1]) for row in reader])
        with open(csv_dir + '/val_fold%s.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_val_fa = ([OsJoin(opt.data_root_path, row[2]) for row in reader])
        with open(csv_dir + '/val_fold%s.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_val_fc = ([OsJoin(opt.data_root_path, row[3]) for row in reader])
        file_val = np.array([file_val_alff, file_val_dfc, file_val_fa, file_val_fc])
        # file_val=[file_val file_val_tmp]
        # row=(row[0:5] for row in reader)
        # file_val_tmp=([OsJoin(opt.data_root_path, row[1]) for row in reader])
        with open(csv_dir + '/val_fold%s.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_val = [row[4] for row in reader]
        self.image = file_val
        self.label = label_val
        self.loader = loader
        self.dfc_loader = dfc_loader
        self.sfc_loader = sfc_loader
        self.nii_loader = nii_loader
    def __getitem__(self, index):
        fn = self.image[:, index]
        img_arr = []
        for fea in fn:
            if "dfc" in fea:
              img_arr_tmp = self.dfc_loader(fea)
              img_arr.append(img_arr_tmp)
            elif "sfc" in fea:
              img_arr_tmp = self.sfc_loader(fea)
              img_arr.append(img_arr_tmp)
            elif "Alff" in fea:
               img_arr_tmp = self.nii_loader(fea)
               img_arr.append(img_arr_tmp)
            else:
                img_arr_tmp = self.loader(fea)
                img_arr.append(img_arr_tmp)
        # img = self.loader(fn)
        label = self.label[index]
        return img_arr, label

    def __len__(self):
        return len(self.image[1])

class TestSet(Dataset):

    def __init__(self,loader=default_loader, dfc_loader=dfc_loader, sfc_loader=sfc_loader, nii_loader=nii_loader):
        file_test = []
        with open(csv_dir + '/test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_test_alff = ([OsJoin(opt.data_root_path, row[0]) for row in reader])
        with open(csv_dir + '/test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_test_dfc = ([OsJoin(opt.data_root_path, row[1]) for row in reader])
        with open(csv_dir + '/test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_test_fa = ([OsJoin(opt.data_root_path, row[2]) for row in reader])
        with open(csv_dir + '/test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_test_fc = ([OsJoin(opt.data_root_path, row[3]) for row in reader])
        with open(csv_dir + '/test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_test = [row[4] for row in reader]

        file_test = np.array([file_test_alff, file_test_dfc, file_test_fa, file_test_fc])
        self.image = file_test
        self.label = label_test
        self.loader = loader
        self.dfc_loader = dfc_loader
        self.sfc_loader = sfc_loader
        self.nii_loader = nii_loader
    def __getitem__(self, index):
        fn = self.image[:, index]
        img_arr = []
        for fea in fn:
            if "dfc" in fea:
              img_arr_tmp = self.dfc_loader(fea)
              img_arr.append(img_arr_tmp)
            elif "sfc" in fea:
              img_arr_tmp = self.sfc_loader(fea)
              img_arr.append(img_arr_tmp)
            elif "Alff" in fea:
               img_arr_tmp = self.nii_loader(fea)
               img_arr.append(img_arr_tmp)
            else:
                img_arr_tmp = self.loader(fea)
                img_arr.append(img_arr_tmp)
        # img = self.loader(fn)
        label = self.label[index]
        return img_arr, label

    def __len__(self):
        return len(self.image[1])
