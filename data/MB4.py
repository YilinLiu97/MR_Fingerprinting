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
        return 'multiband'
    

    def get_paths(self):
        if self.opt.onMAC:
            d_root = '/Users/zhenghanfang/Desktop/standard_MRF/DataNewDictionary/'
        else:
            d_root = '/shenlab/lab_stor6/yilinliu/multiband/'
        person_path = ['180114', '180124', '180131', '180131_2', '180202']
        self.n_Network = self.opt.n_Network
     
        slice_N = [12,12,12,12,12]
#        slice_N = [1,1,1,1,1]
        test_i = self.opt.test_i
        if self.opt.set_type == 'train':
            person = list(range(1,test_i))+list(range(test_i+1,len(person_path)+1))
        else:
            person = list(range(test_i,test_i+1))
        # person = list(range(1,7))

        self.data_paths = []
        for i in range(len(person)):
            a = os.listdir(d_root+'MB4/'+person_path[person[i]-1])
            #a.sort(key=lambda f: int(filter(str.isdigit, f)))
            a = natsorted(a)
            label_dir = os.listdir(d_root+'MB4/Labels/'+person_path[person[i]-1])
            label_dir = natsorted(label_dir)
            #label_dir.sort(key=lambda f: int(filter(str.isdigit, f)))
            mask_dir = os.listdir(d_root+'training/Masks/'+person_path[person[i]-1])
            mask_dir = natsorted(mask_dir)
            #mask_dir.sort(key=lambda f: int(filter(str.isdigit, f)))
            print('%%%%%%%%%%%%%%%%% label_dir: ', label_dir)
            print('%%%%%%%%%%%%%%%%% mask_dir: ', mask_dir)
            print('%%%%%%%%%%%%%%%% len(a): ', len(a))
            for p in a:
                if p[0] == '.':
                    a.remove(p)
            for j in range(slice_N[person[i]-1]):
      
#                print('person: ', person)
#                print('slice_N: ', slice_N)
#                print('person[3]: ', person[3])
                if j+3*(self.n_Network-1) >= slice_N[person[i]-1]:
                   #print('slice_N[person[i]]: ', slice_N[person[i]])
                   print('j-(slice_N[person[i]-1]-3*(self.n_Network-1)): ', j-(slice_N[person[i]-1]-3*(self.n_Network-1)))
                   self.data_paths.append({
                        'imMRF': d_root+'MB4/'+ person_path[person[i]-1]+'/'+a[j]+'/imMRF_all_simulated.mat',
                        'Tmap': d_root+'MB4/Labels/'+person_path[person[i]-1]+'/'+label_dir[j-(slice_N[person[i]-1]-3*(self.n_Network-1))]+'/patternmatching.mat', # sparse dict
                        # 'Tmap': d_root+person_path[person[i]-1]+'/'+a[j]+'/patternmatching_densedict.mat', # dense dict
                        'mask': d_root+'training/Masks/'+person_path[person[i]-1]+'/'+mask_dir[j-(slice_N[person[i]-1]-3*(self.n_Network-1))]+'/immask.mat' # large mask
                        # 'mask': d_root+'Data_Qian_skull_h5/'+str(person[i])+'/'+str(j+1)+'-skull.mat' # small mask
                        })
                else:
                   #print('j+3*(self.n_Network-1): ', j+3*(self.n_Network-1))
                   self.data_paths.append({
                        'imMRF': d_root+'MB4/'+ person_path[person[i]-1]+'/'+a[j]+'/imMRF_all_simulated.mat',
                        'Tmap': d_root+'MB4/Labels/'+person_path[person[i]-1]+'/'+label_dir[j+3*(self.n_Network-1)]+'/patternmatching.mat', # sparse dict
                        # 'Tmap': d_root+person_path[person[i]-1]+'/'+a[j]+'/patternmatching_densedict.mat', # dense dict
                        'mask': d_root+'training/Masks/'+person_path[person[i]-1]+'/'+mask_dir[j+3*(self.n_Network-1)]+'/immask.mat' # large mask
                        # 'mask': d_root+'Data_Qian_skull_h5/'+str(person[i])+'/'+str(j+1)+'-skull.mat' # small mask
                        })
             
