# import os.path
# import torchvision.transforms as transforms
# from data.base_dataset import BaseDataset, get_transform
from data.base_dataset import BaseDataset
# from data.image_folder import make_dataset
# from PIL import Image
# import PIL
import h5py
import random
import torch
import numpy
import math
# import skimage.transform
import time
import scipy.io as sio
import os
import util.util as util
import time

class MRFDataset(BaseDataset):
    def initialize(self, opt):
        self.flipimMRF = False
        self.initialize_base(opt)

    def name(self):
        return 'TestData'
    

    def get_paths(self):
        if self.opt.onMAC:
            d_root = '/Users/zhenghanfang/Desktop/standard_MRF/DataNewDictionary/'
        else:
            d_root = '/shenlab/lab_stor6/yilinliu/MB4_TestData/Raw_2304_slice1/Raw_2304/'
        
        subject_N = [5]

        test_i = self.opt.test_i
        if self.opt.set_type == 'train':
            person = list(range(1,test_i))+list(range(test_i+1,2))
        else:
            person = list(range(test_i,test_i+1))
        # person = list(range(1,7))

        self.data_paths = []
        a = os.listdir(d_root+'data')
        a.sort(key=lambda f: int(filter(str.isdigit, f)))
        for p in a:
            if p[0] == '.':
               a.remove(p)
        if self.opt.set_type == 'train':
           for j in range(7):
                self.data_paths.append({
                'imMRF': d_root+'data'+'/'+a[j],
                'Tmap': d_root+'goals'+'/'+a[j], # sparse dict
                # 'Tmap': d_root+person_path[person[i]-1]+'/'+a[j]+'/patternmatching_densedict.mat', # dense dict
                'mask': d_root+'goals'+'/'+a[j] # large mask
                # 'mask': d_root+'Data_Qian_skull_h5/'+str(person[i])+'/'+str(j+1)+'-skull.mat' # small mask
                })

        else:
            for j in range(11):
                self.data_paths.append({
                'imMRF': d_root+'data'+'/'+a[0],
                'Tmap': d_root+'goals'+'/'+a[0], # sparse dict
                # 'Tmap': d_root+person_path[person[i]-1]+'/'+a[j]+'/patternmatching_densedict.mat', # dense dict
                'mask': d_root+'goals'+'/'+a[0] # large mask
                # 'mask': d_root+'Data_Qian_skull_h5/'+str(person[i])+'/'+str(j+1)+'-skull.mat' # small mask
                })
