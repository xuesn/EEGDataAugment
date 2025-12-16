import os
import random
import numpy as np
import torch
import logging

from .utils_dataset import *


class My_Dataset_nopreload_preload(torch.utils.data.Dataset):

    #主构造函数
    def __init__(self,
            eeg_path_dataset=None,label_dataset=None,imgNO_dataset=None,
            eeg_npy_dir=None, label_npy_dir=None,imgNO_npy_dir=None,
            ori_timepoint_num=None,ori_electrode_num=None,ori_class_num=None,
            preproc_dir_path=None,subject_list=None,session_list=None,
            clamp_thres=None,epoch_point_st=None,epoch_point_end=None,
            norm_per_sample=None,norm_per_electrode=None,norm_per_2sample_electrode=None,
            data_dir_list=[]):

        self.preproc_dir_path = preproc_dir_path
        self.subject_list = subject_list
        self.session_list = session_list

        #getitem时做二次预处理
        self.clamp_thres = clamp_thres
        self.epoch_point_st = epoch_point_st
        self.epoch_point_end = epoch_point_end
        self.norm_per_sample = norm_per_sample #是否每个样本单独归一化
        self.norm_per_electrode = norm_per_electrode #是否分电极归一化
        self.norm_per_2sample_electrode = norm_per_2sample_electrode #是否每个样本单独分电极归一化

        self.eeg_path_dataset = eeg_path_dataset
        self.label_dataset = label_dataset # 从0开始
        self.imgNO_dataset = imgNO_dataset # 从0开始

        # 24-10-24
        self.class_num = ori_class_num
        if eeg_path_dataset is not None:
            sample_num = len(eeg_path_dataset)
            self.label_onehot_dataset = np.zeros([sample_num,ori_class_num])
            for i in range(sample_num):
                classNO = label_dataset[i]
                self.label_onehot_dataset[i,classNO]=1
        
        # 25-04-16
        self.data_dir_list = data_dir_list
        self.eeg_dataset = None
        self.ori_electrode_num = ori_electrode_num
        self.ori_timepoint_num = ori_timepoint_num
        
    def __len__(self):
        return len(self.eeg_path_dataset)

    def preprocess_twice_dataset(self, eeg_dataset):
        if (self.epoch_point_st is not None) and  (self.epoch_point_end is not None) :
            eeg_dataset = np.array(eeg_dataset)[:,self.epoch_point_st:self.epoch_point_end, :]
        # print('eeg_dataset:',eeg_dataset)
        # clamp
        if (not self.clamp_thres == 'none') and (not self.clamp_thres is None):
            eeg_dataset=clamp_per_electrode(eeg_dataset,self.clamp_thres)
            # print('eeg_dataset:',eeg_dataset)
        #归一化
        if self.norm_per_sample:
            eeg_dataset=normalize_per_sample(eeg_dataset)
        if self.norm_per_electrode:
            eeg_dataset=normalize_per_electrode(eeg_dataset)
        if self.norm_per_2sample_electrode:
            eeg_dataset=normalize_per_sample_electrode(eeg_dataset)
        return eeg_dataset

    def preprocess_twice(self, eeg):
        eeg = np.array(eeg)[self.epoch_point_st:self.epoch_point_end, :]
        # print('eeg:',eeg)
        # clamp
        if (not self.clamp_thres == 'none') and (not self.clamp_thres is None):
            eeg=(eeg-np.mean(eeg, axis=0))#必须先去均值，否则clamp可能偏差很大
            eeg[eeg > self.clamp_thres] = self.clamp_thres
            eeg[eeg < -self.clamp_thres] = -self.clamp_thres
            # print('eeg:',eeg)
        #归一化
        if self.norm_per_sample:
            eeg =  (eeg - np.mean(eeg)) / (np.std(eeg)+1e-5)
            # print(eeg_npy_path,' np.std(eeg):',np.std(eeg))
        #归一化
        if self.norm_per_electrode:
            eeg = (eeg - np.mean(eeg, axis=0)) / (np.std(eeg, axis=0)+1e-5)
        return eeg

    def __getitem__(self, idx):  
        if  len(self.imgNO_dataset)==0:
            imgNO = 9999
        else:
            #imgNO
            imgNO=self.imgNO_dataset[idx,:]
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
        #label
        label_0_x=self.label_dataset[idx]
        label_onehot=np.zeros([self.class_num])
        label_onehot[label_0_x]=1
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
        #eeg
        if self.eeg_dataset  is not None:
            eeg=self.eeg_dataset[idx,:,:]
            normed_eeg = eeg #提前预处理好了

        elif self.eeg_dataset  is  None:
            eeg_path=self.eeg_path_dataset[idx]
            eeg = np.load(os.path.join(self.preproc_dir_path,eeg_path))
            eeg=eeg.astype(np.float32)
            normed_eeg = self.preprocess_twice(eeg)

        subNO=-99 #这回未记录subNO
        return (torch.tensor(normed_eeg).float(),
            torch.tensor(label_onehot).float(), 
            imgNO)
            # imgNO要为int

    def divide(self,sample_num_per_class):
        # 1、取一个aug的做divide
        all_augment_sample_num, class_num = self.label_onehot_dataset.shape
        aug_num = len(self.data_dir_list )
        one_augment_sample_num = int(all_augment_sample_num/aug_num)
        assert all_augment_sample_num%aug_num ==0 ,'sample_num不能被aug_num整除 '

        one_augment_label_onehot_dataset = self.label_onehot_dataset[0:one_augment_sample_num]

        when_balance_class__max_sample_num_per_class = int(one_augment_sample_num/self.class_num)
        if (sample_num_per_class is not None)   and  (not sample_num_per_class ==  when_balance_class__max_sample_num_per_class):
            
            # 2、计算单augment中divide出的样本号
            # 先设定好随机数
            seed = 2025
            # split_num = len(split_proportion_list)
            sample_num, class_num = one_augment_label_onehot_dataset.shape
            perm_shuffle = np.arange(sample_num)
            np.random.seed(seed)
            np.random.shuffle(perm_shuffle)
            # print(perm_shuffle)
            
            # # --------------------------------------------------------------------
            # 原数据shuffle后，按类别sort，然后逐类切分
            label_0_x_dataset = np.argmax(self.label_onehot_dataset,axis=1).squeeze()
            label_0_x_dataset_shuffled = label_0_x_dataset[perm_shuffle]
            perm_sort = np.argsort(label_0_x_dataset_shuffled)
            
            perm_shuffle_per_class = perm_shuffle[perm_sort]
                
            sampleNO_divide=[]
            sampleNO_st = 0
            for classNO in range(class_num):
                sample_num_this_class = np.sum(label_0_x_dataset_shuffled==classNO)
                # print(sample_num_this_class)
                
                if sample_num_per_class>sample_num_this_class:
                    assert False,'sample_num_per_class 不能大于 sample_num_this_class！'+'\n sample_num_per_class:'+str(sample_num_per_class)+' sample_num_this_class:'+str(sample_num_this_class)

                st_idx = sampleNO_st
                end_idx = st_idx + sample_num_per_class

                #注意这里不能用+=，会出现错误结果，导致整体都+了
                sampleNO_divide=sampleNO_divide + perm_shuffle_per_class[st_idx:end_idx].tolist()

                sampleNO_st+=sample_num_this_class

            # 3、对所有augment的取对应位的样本
            current_aug_sampleNO_divide = sampleNO_divide
            for i in range(aug_num-1):
                current_aug_sampleNO_divide = list(np.array(current_aug_sampleNO_divide)+one_augment_sample_num)
                sampleNO_divide = sampleNO_divide+current_aug_sampleNO_divide

            # SPECIAL~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
            # 选出部分样本
            # self.eeg_dataset = self.eeg_dataset[sampleNO_divide,:,:]
            self.label_onehot_dataset = self.label_onehot_dataset[sampleNO_divide,:]
            # self.imgNO_dataset = self.imgNO_dataset[sampleNO_divide,:] # 15S
            if  not len(self.imgNO_dataset)==0:
                self.imgNO_dataset = self.imgNO_dataset[sampleNO_divide,] # 21P 22G
            
            if self.eeg_path_dataset is not None:
                self.eeg_path_dataset = list(np.array(self.eeg_path_dataset)[sampleNO_divide])
                
            if self.label_dataset is not None:
                self.label_dataset = list(np.array(self.label_dataset)[sampleNO_divide])        
            print('origin sample_num=',all_augment_sample_num,'divided to ',self.label_onehot_dataset.shape[0])

        # divide后就load-eeg-data以加快训练速度
        self.load_eeg_data()

    def load_eeg_data(self,):
        sample_num = len(self.eeg_path_dataset)
        assert sample_num==len(self.label_dataset),'eeg_path与label数不等'
        print('需要load的eeg_path数为：',sample_num)
        
        eeg_dataset = np.zeros([sample_num, self.ori_timepoint_num, self.ori_electrode_num,])

        for sample_load_NO,eeg_path in enumerate(self.eeg_path_dataset):
            eeg_dataset[sample_load_NO,:,:] = np.load(os.path.join(self.preproc_dir_path,eeg_path))
            if sample_load_NO % 5000 == 4999:
                print('已load样本数为：',sample_load_NO)

        import time
        start_time = time.time()
        normed_eeg_dataset = self.preprocess_twice_dataset( eeg_dataset)
        print( '二次预处理用时:', start_time - time.time())
        normed_eeg_dataset=normed_eeg_dataset.astype(np.float32)
        self.eeg_dataset = normed_eeg_dataset

