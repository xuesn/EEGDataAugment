




from .dataset_nopreload_22G import *
import os
import numpy as np
import json
from utils.utils import main_dir


# 24-10-24
preproc_dir_path = main_dir + '/data/augment/'


dataset_str = '22Germany_mean'
ori_class_num = 1854  # train的标签也是从0~1853但是有200个标签没有
ori_electrode_num = 63

# -0.1s ~ 0.5s  250Hz
# 0.0s ~ 1.0s  250Hz
ori_timepoint_num=250  # 虽然预处理是1s，但是实际训用的500ms
ori_timepoint_num=125  # augment时改为500ms


# 记得重新存一个仅含200个图像的visual-embedding
def convert_label_imgNO(label_0_1853_dataset, imgNO_dataset):
    # dict
    # label
    test_label_0_1853_list = [
        9,26,50,58,63,79,81,83,86,95,96,107,120,123,132,148,154,156,163,171,
        186,194,195,212,213,216,220,227,239,254,278,281,284,286,290,298,306,312,315,320,
        328,339,353,354,369,371,375,378,382,400,408,425,428,439,440,441,442,446,450,451,
        461,469,471,481,514,517,522,529,537,545,546,554,560,570,580,608,625,634,640,642,
        645,658,680,689,691,693,694,695,706,713,740,741,745,768,777,792,811,819,820,843,
        844,857,868,878,887,889,893,902,920,929,964,974,990,996,1003,1011,1020,1034,1041,1042,
        1059,1074,1076,1077,1078,1081,1084,1099,1106,1117,1133,1149,1160,1166,1168,1171,1173,1205,1207,1223,
        1225,1228,1239,1249,1255,1259,1274,1284,1297,1306,1318,1330,1340,1347,1352,1356,1365,1372,1373,1382,
        1395,1400,1401,1449,1460,1462,1464,1467,1485,1508,1519,1542,1550,1570,1577,1580,1606,1608,1617,1624,
        1641,1655,1656,1673,1674,1676,1682,1706,1724,1733,1746,1765,1767,1777,1778,1805,1808,1820,1823,1830,]
    test_label_0_1853_list.sort()
    label_proj_dict = { new_label_index: test_label_0_1853
        for new_label_index, test_label_0_1853 in enumerate(test_label_0_1853_list)
    }
    # imgNO
    test_imgNO_list = list(range(16540,16739+1))
    test_imgNO_list.sort()
    imgNO_proj_dict = { test_imgNO: new_imgNO 
        for new_imgNO, test_imgNO in enumerate(test_imgNO_list)
    }
    # convert
    converted_label_0_199_dataset = [label_proj_dict[label_0_1853] for label_0_1853 in label_0_1853_dataset]
    converted_imgNO_dataset = [imgNO_proj_dict[imgNO] for imgNO in imgNO_dataset]
    return converted_label_0_199_dataset, converted_imgNO_dataset


# data_dir_list可选的例子
# data_dir_list = ['22_Germany-origin-10fold'] # 原始80次重复放映的数据
# average_num_list = [2,4,8,18,36,72] # 在叠加平均后，再保存新的eeg、label、imgNO
# for average_num in average_num_list:
#     data_dir_list.append('22_Germany-mean'+str(average_num))
def get_train_set_fold_list(data_dir, foldNO_val ):
    train_fold_list = []
    fold_num = 10
    if data_dir in ['22_Germany-mean18',
                    '22_Germany-mean36',
                    '22_Germany-mean72',]:
        train_fold_list = ['fold'+str(foldNO_val).zfill(2)]

    elif data_dir in ['22_Germany-origin-10fold',
                    '22_Germany-mean2',
                    '22_Germany-mean4',
                    '22_Germany-mean8',]:
        for foldNO in range(fold_num):
            if not foldNO==foldNO_val:
                fold_dir = 'fold' + str(foldNO).zfill(2)
                train_fold_list.append(fold_dir)
    elif '22_Germany-origin-10fold_' in data_dir:
        for foldNO in range(fold_num):
            if not foldNO==foldNO_val:
                fold_dir = 'fold' + str(foldNO).zfill(2)
                train_fold_list.append(fold_dir)
    elif '22_Germany-shift_mean' in data_dir:
        for foldNO in range(fold_num):
            train_fold_list = ['fold'+str(foldNO_val).zfill(2)]
    return train_fold_list


def load_json_22Germany(train_or_test_22G,
        data_dir_list, sub_list,
        foldNO_val, train_or_val,
        preproc_dir_path):
    eeg_path_dataset = []
    label_0_1853_dataset = []
    imgNO_dataset = []
    img_str_dataset = []
    for data_dir in data_dir_list:
        if train_or_val == 'train':
            fold_list = get_train_set_fold_list(data_dir, foldNO_val)
        elif train_or_val == 'val':
            fold_list = ['fold'+str(foldNO_val).zfill(2)]
        elif train_or_val == 'cross_sub':
            fold_num=10
            fold_list = ['fold'+str(i).zfill(2) for i in range(fold_num)]
        for sub_dir in sub_list:
            for fold_dir in fold_list:
                # 文件命名
                save_eeg_path_fname = '_'.join([sub_dir,train_or_test_22G,fold_dir,'eeg_path.json'])
                save_label_0_1853_fname = '_'.join([sub_dir,train_or_test_22G,fold_dir,'label_0_1853.json'])
                save_imgNO_fname = '_'.join([sub_dir,train_or_test_22G,fold_dir,'imgNO.json'])
                save_img_str_fname = '_'.join([sub_dir,train_or_test_22G,fold_dir,'img_str.json'])
                # 存在对应fold文件夹内

                # 
                change_data_dir  = data_dir
                if '22_Germany-origin-10fold_' in data_dir:
                    change_data_dir = '22_Germany-origin-10fold'

                save_dir_fold = os.path.join(preproc_dir_path, change_data_dir, sub_dir, train_or_test_22G, fold_dir)
                save_eeg_path_fpath = os.path.join(save_dir_fold,save_eeg_path_fname)
                save_label_0_1853_fpath = os.path.join(save_dir_fold,save_label_0_1853_fname)
                save_imgNO_fpath = os.path.join(save_dir_fold,save_imgNO_fname)
                save_img_str_fpath = os.path.join(save_dir_fold,save_img_str_fname)
                # load
                with open(save_eeg_path_fpath, "r",encoding='UTF-8') as f:
                    eeg_path_dataset_this_sub_temp = json.load(f)
                    
                eeg_path_dataset_this_sub = [fn.replace(change_data_dir,data_dir) for fn in eeg_path_dataset_this_sub_temp] 

                with open(save_label_0_1853_fpath, "r",encoding='UTF-8') as f:
                    label_0_1853_dataset_this_sub = json.load(f)
                with open(save_imgNO_fpath, "r",encoding='UTF-8') as f:
                    imgNO_dataset_this_sub = json.load(f)
                with open(save_img_str_fpath, "r",encoding='UTF-8') as f:
                    img_str_dataset_this_sub = json.load(f)
                # add
                eeg_path_dataset += eeg_path_dataset_this_sub
                label_0_1853_dataset += label_0_1853_dataset_this_sub
                imgNO_dataset += imgNO_dataset_this_sub
                img_str_dataset += img_str_dataset_this_sub
                print(data_dir, sub_dir, fold_dir, ' added!!')
    return eeg_path_dataset,label_0_1853_dataset, imgNO_dataset, img_str_dataset


def dataset_22Germany(train_or_test_22G ,
        data_dir_list,
        sub_list,
        foldNO_val, train_or_val,
        clamp_thres,
        norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,):
    # load json
    (eeg_path_dataset, label_0_1853_dataset,imgNO_dataset, 
     img_str_dataset) = load_json_22Germany(
                            train_or_test_22G,
                            data_dir_list,sub_list,
                            foldNO_val, train_or_val,
                            preproc_dir_path)
    # convert
    (converted_label_0_199_dataset,
        converted_imgNO_dataset) = convert_label_imgNO(label_0_1853_dataset, imgNO_dataset)
    #
    label_0_199_dataset=(np.array(converted_label_0_199_dataset)).astype(np.int32)
    imgNO_dataset=np.array(converted_imgNO_dataset).astype(np.int32).reshape(-1,1)

    # 暂时不用的参数（以后如果存在了不同文件夹里再用）
    eeg_npy_dir = None
    label_npy_dir = None
    imgNO_npy_dir = None

    # epoch_point_st = 25
    # epoch_point_end = 150
    # 0830
    epoch_point_st = 0  # 0:0ms
    epoch_point_end = 125  # 125:500ms
    subject_list = None
    session_list = None

    whole_set = My_Dataset_nopreload(
            eeg_path_dataset,label_0_199_dataset,imgNO_dataset,
            eeg_npy_dir, label_npy_dir,imgNO_npy_dir,
            ori_timepoint_num,ori_electrode_num,ori_class_num,
            preproc_dir_path,subject_list,session_list,
            clamp_thres,epoch_point_st,epoch_point_end,
            norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,
            data_dir_list)
        
    return whole_set


def dataset_22Germany_test(data_dir_list,
        sub_list,
        foldNO_val, train_or_val,
        clamp_thres,
        norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,):
    train_or_test_22G = 'test'
    return dataset_22Germany(data_dir_list,
        sub_list,
        foldNO_val, train_or_val,
        clamp_thres,
        norm_per_sample,norm_per_electrode,norm_per_2sample_electrode,)

