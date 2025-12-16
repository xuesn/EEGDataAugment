
from .dataset_preload_15S import *

import os
import numpy as np
import json


from utils.utils import main_dir




preproc_dir_path = main_dir + '/data/eeg_preprocess/augment/15Stanford/'

dataset_str = '15Stanford'

ori_timepoint_num=32# # 0.5s   62.5Hz  
ori_electrode_num=124



def get_this_fold(foldNO_val,):
    train_fold_list = ['fold'+str(foldNO_val).zfill(2)]
    return train_fold_list
def get_other_fold(foldNO_val,fold_num):
    train_fold_list = []
    for foldNO in range(fold_num):
        if not foldNO==foldNO_val:
            fold_dir = 'fold' + str(foldNO).zfill(2)
            train_fold_list.append(fold_dir)
    return train_fold_list
def get_train_set_fold_list(data_dir, foldNO_val ):
    if ('Germany' in data_dir) or ('Purdue' in data_dir) :
        fold_num = 10
    elif ('Stanford' in data_dir) :
        fold_num = 9
    # 
    if (data_dir in ['22_Germany-mean18',
                    '22_Germany-mean36',
                    '22_Germany-mean72',]):
        train_fold_list = get_this_fold(foldNO_val,)

    elif data_dir in ['22_Germany-mean2',
                    '22_Germany-mean4',
                    '22_Germany-mean8',]:
        train_fold_list = get_other_fold(foldNO_val,fold_num)

    elif '-origin-10fold' in data_dir: # 22G和21P
        train_fold_list = get_other_fold(foldNO_val,fold_num)

    elif '15Stanford-origin-9fold' in data_dir: # 15S
        train_fold_list = get_other_fold(foldNO_val,fold_num)

    elif '15Stanford-mean' in data_dir: # 15S
        train_fold_list = get_this_fold(foldNO_val,)
    return train_fold_list

    
def load_npy_json_15Stanford(
        data_dir_list, sub_list,
        foldNO_val, train_or_val,):
    eeg_dataset = np.zeros([0,ori_timepoint_num,ori_electrode_num])
    label_0_5_dataset = []
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
                # save_eeg_path_fname = '_'.join([sub_dir,fold_dir,'eeg_path.json'])
                save_eeg_data_fname = '_'.join([sub_dir,fold_dir,'eeg_data.npy'])
                save_label_0_5_fname = '_'.join([sub_dir,fold_dir,'label_0_5.json'])
                save_imgNO_fname = '_'.join([sub_dir,fold_dir,'imgNO.json'])
                # save_img_str_fname = '_'.join([sub_dir,fold_dir,'img_str.json'])

                # 
                change_data_dir = data_dir
                if '_' in data_dir:  # 代表是augment的数据
                    change_data_dir = data_dir.split('_')[0]

                # eeg_path 由于文件夹不同，分别生成了
                save_dir_fold = os.path.join(preproc_dir_path, data_dir, sub_dir,  fold_dir)
                # save_eeg_path_fpath = os.path.join(save_dir_fold,save_eeg_path_fname)
                save_eeg_data_fpath = os.path.join(save_dir_fold,save_eeg_data_fname)

                # label 只在origin数据里存了
                save_dir_fold = os.path.join(preproc_dir_path, change_data_dir, sub_dir,  fold_dir)
                save_label_0_5_fpath = os.path.join(save_dir_fold,save_label_0_5_fname)
                save_imgNO_fpath = os.path.join(save_dir_fold,save_imgNO_fname)
                # save_img_str_fpath = os.path.join(save_dir_fold,save_img_str_fname)

                # load
                eeg_dataset_this_sub = np.load(save_eeg_data_fpath)
                #
                # with open(save_eeg_path_fpath, "r",encoding='UTF-8') as f:
                #     eeg_path_dataset_this_sub = json.load(f)
                with open(save_label_0_5_fpath, "r",encoding='UTF-8') as f:
                    label_0_5_dataset_this_sub = json.load(f)
                with open(save_imgNO_fpath, "r",encoding='UTF-8') as f:
                    imgNO_dataset_this_sub = json.load(f)
                # with open(save_img_str_fpath, "r",encoding='UTF-8') as f:
                #     img_str_dataset_this_sub = json.load(f)

                # add
                eeg_dataset = np.concatenate((eeg_dataset,eeg_dataset_this_sub),axis=0)
                # eeg_path_dataset += eeg_path_dataset_this_sub
                label_0_5_dataset += label_0_5_dataset_this_sub
                imgNO_dataset += imgNO_dataset_this_sub
                # img_str_dataset += img_str_dataset_this_sub
                print(data_dir, sub_dir, fold_dir, ' added!!')
    return eeg_dataset,label_0_5_dataset, imgNO_dataset, img_str_dataset
    

def dataset_15Stanford_6class(
        data_dir_list,
        sub_list,
        foldNO_val, train_or_val,
        clamp_thres,
        norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,):

    # load json
    (eeg_dataset, label_0_5_dataset,
        imgNO_dataset, img_str_dataset) = load_npy_json_15Stanford(data_dir_list,sub_list, foldNO_val, train_or_val)
    label_dataset=np.array(label_0_5_dataset).astype(np.int32)       
    imgNO_dataset=np.array(imgNO_dataset).astype(np.int32)

    class_num=6

    whole_set = My_Dataset_array_preload(
                eeg_dataset,label_dataset,imgNO_dataset,
                preproc_dir_path, preproc_dir_path,preproc_dir_path,
                ori_timepoint_num,ori_electrode_num,class_num,
                None, sub_list,None,
                clamp_thres,None,None,            
                norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,)
    return whole_set


def dataset_15Stanford_72pic(
        data_dir_list,
        sub_list,
        foldNO_val, train_or_val,
        clamp_thres,
        norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,):

    # load json
    (eeg_dataset, label_0_5_dataset,
        imgNO_dataset, img_str_dataset) = load_npy_json_15Stanford(data_dir_list,sub_list, foldNO_val, train_or_val)
    label_dataset=np.array(label_0_5_dataset).astype(np.int32)       
    imgNO_dataset=np.array(imgNO_dataset).astype(np.int32)

    # 与15S6的区别仅在于此
    class_num=72
    label_dataset = np.copy(imgNO_dataset)

    whole_set = My_Dataset_array_preload(
                eeg_dataset,label_dataset,imgNO_dataset,
                preproc_dir_path, preproc_dir_path,preproc_dir_path,
                ori_timepoint_num,ori_electrode_num,class_num,
                None, sub_list,None,
                clamp_thres,None,None,            
                norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,)
    return whole_set
