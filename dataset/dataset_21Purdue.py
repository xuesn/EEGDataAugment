from .dataset_nopreload_preload_21P import *
import os
import numpy as np
import json
from utils.utils import main_dir


preproc_dir_path = main_dir + '/data/eeg_preprocess/augment/'

dataset_str = '21Purdue'
ori_class_num = 40
ori_electrode_num=96

# 2.1s
# 250Hz
ori_timepoint_num=525


def get_train_set_fold_list(data_dir, foldNO_val ):
    train_fold_list = []
    fold_num = 10
    if data_dir in ['22_Germany-mean18',
                    '22_Germany-mean36',
                    '22_Germany-mean72',]:
        train_fold_list = ['fold'+str(foldNO_val).zfill(2)]

    elif data_dir in ['22_Germany-mean2',
                    '22_Germany-mean4',
                    '22_Germany-mean8',]:
        for foldNO in range(fold_num):
            if not foldNO==foldNO_val:
                fold_dir = 'fold' + str(foldNO).zfill(2)
                train_fold_list.append(fold_dir)
    elif '-origin-10fold' in data_dir:
        for foldNO in range(fold_num):
            if not foldNO==foldNO_val:
                fold_dir = 'fold' + str(foldNO).zfill(2)
                train_fold_list.append(fold_dir)
    return train_fold_list


def load_json_21Purdue(
        data_dir_list, sub_list,
        foldNO_val, train_or_val,):
    eeg_path_dataset = []
    label_0_39_dataset = []
    imgNO_dataset = []
    img_str_dataset = []
    for data_dir in data_dir_list:
        if train_or_val == 'train':
            fold_list = get_train_set_fold_list(data_dir, foldNO_val)
        elif train_or_val == 'val':
            fold_list = ['fold'+str(foldNO_val).zfill(2)]
        for sub_dir in sub_list:
            for fold_dir in fold_list:
                # 文件命名
                save_eeg_path_fname = '_'.join([sub_dir,fold_dir,'eeg_path.json'])
                save_label_0_39_fname = '_'.join([sub_dir,fold_dir,'label_0_39.json'])
                # save_imgNO_fname = '_'.join([sub_dir,fold_dir,'imgNO.json'])
                save_img_str_fname = '_'.join([sub_dir,fold_dir,'img_str.json'])

                # 
                change_data_dir = data_dir
                if '-origin-10fold_' in data_dir:
                    change_data_dir = data_dir.split('-origin-10fold_')[0]+'-origin-10fold'

                # eeg_path 由于文件夹不同，分别生成了
                save_dir_fold = os.path.join(preproc_dir_path, data_dir, sub_dir,  fold_dir)
                save_eeg_path_fpath = os.path.join(save_dir_fold,save_eeg_path_fname)

                # label 只在origin数据里存了
                save_dir_fold = os.path.join(preproc_dir_path, change_data_dir, sub_dir,  fold_dir)
                save_label_0_39_fpath = os.path.join(save_dir_fold,save_label_0_39_fname)
                # save_imgNO_fpath = os.path.join(save_dir_fold,save_imgNO_fname)
                save_img_str_fpath = os.path.join(save_dir_fold,save_img_str_fname)
                # load
                with open(save_eeg_path_fpath, "r",encoding='UTF-8') as f:
                    eeg_path_dataset_this_sub = json.load(f)
                with open(save_label_0_39_fpath, "r",encoding='UTF-8') as f:
                    label_0_39_dataset_this_sub = json.load(f)
                # with open(save_imgNO_fpath, "r",encoding='UTF-8') as f:
                #     imgNO_dataset_this_sub = json.load(f)
                with open(save_img_str_fpath, "r",encoding='UTF-8') as f:
                    img_str_dataset_this_sub = json.load(f)
                # add
                eeg_path_dataset += eeg_path_dataset_this_sub
                label_0_39_dataset += label_0_39_dataset_this_sub
                # imgNO_dataset += imgNO_dataset_this_sub
                img_str_dataset += img_str_dataset_this_sub
                if sub_dir == 'exp99':
                    print(data_dir, sub_dir, fold_dir, ' added!!')
    return eeg_path_dataset,label_0_39_dataset, imgNO_dataset, img_str_dataset
    

def dataset_21Purdue(
        data_dir_list,
        sub_list,
        foldNO_val, train_or_val,
        clamp_thres,
        norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,):

    # 注意imgNO_dataset我早期处理的数据没有imgNO，以后要用可以找代码09-19-15S-21P-22G视觉特征提取

    # load json
    (eeg_path_dataset, label_0_39_dataset,
        imgNO_dataset, img_str_dataset) = load_json_21Purdue(data_dir_list,sub_list, foldNO_val, train_or_val)
    label_dataset=np.array(label_0_39_dataset).astype(np.int32)       
    imgNO_dataset=np.array(imgNO_dataset).astype(np.int32)

    # print(imgNO_dataset)  == []
    
    eeg_npy_dir = None
    label_npy_dir = None
    imgNO_npy_dir = None
    epoch_point_st = None
    epoch_point_end = None
    subject_list = None
    session_list = None
    whole_set = My_Dataset_nopreload_preload(
            eeg_path_dataset,label_dataset,imgNO_dataset,
            eeg_npy_dir, label_npy_dir,imgNO_npy_dir,
            ori_timepoint_num,ori_electrode_num,ori_class_num,
            preproc_dir_path,subject_list,session_list,
            clamp_thres,epoch_point_st,epoch_point_end,
            norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,data_dir_list)
    return whole_set

