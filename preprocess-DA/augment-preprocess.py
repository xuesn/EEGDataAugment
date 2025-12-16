

from colorama import init, Fore,  Back,  Style
init(autoreset = True)

from utils_augment import *
from utils_augment_batch import *

import os
import json
import numpy as np




sub_list = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10',]
fold_num = 9

# main_dir = '/data1/snxue/data/eeg_preprocess/augment/15Stanford/'
main_dir = '/mnt/data/snxue/eeg_preprocess/augment/15Stanford/'


data_dir_list = ['15Stanford-origin-9fold'] # 原始的数据
# data_dir_list = [] 
average_num_list = [2,3,4,5,6,7,8,9,10] # 在叠加平均后，再保存新的eeg、label、imgNO
for average_num in average_num_list:
    data_dir_list.append('15Stanford-mean'+str(average_num))

# 
for data_dir in data_dir_list:
        # break
# data_dir = '21Purdue-origin-10fold'
    for sub_dir in sub_list:
        # break

        for foldNO in range(fold_num):
            # break
            fold_dir = 'fold' + str(foldNO).zfill(2)

            save_dir_fold = os.path.join(main_dir, data_dir, sub_dir,  fold_dir)
            file_list = os.listdir(save_dir_fold)

            # 读入数据
            # fname
            npy_fname = sub_dir+'_fold'+str(foldNO).zfill(2)+'_eeg_data.npy'
            eeg_batch_sample = np.load(os.path.join(save_dir_fold,npy_fname))

            # 不同的augment方法
            # for augment_str in ['amp_flip','time_flip','shift',]:
            for augment_str in ['noise','mask',]:
                
                def inline_func_save_augment_eeg():
                    augment_data_dir = data_dir + '_' + augment_str + suffix_str
                    augment_save_dir_fold = os.path.join(main_dir, augment_data_dir, sub_dir,  fold_dir)
                                        
                    # 创建文件夹
                    def create_dir(dir_path):
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                            print(dir_path, ' created!')
                    create_dir(augment_save_dir_fold)

                    augment_save_npy_path = os.path.join(augment_save_dir_fold, npy_fname)
                    #
                    # np.save(augment_save_npy_path, augment_eeg)
                    np.save(augment_save_npy_path, augment_eeg.astype(np.float16))


                if augment_str == 'ch_symmetry':
                    suffix_str = ''                            
                    augment_eeg = electrode_reorder_batch(eeg_batch_sample,symmetry_ch_order_22G)
                    inline_func_save_augment_eeg()

                elif augment_str == 'amp_flip':
                    suffix_str = ''                            
                    augment_eeg = amplitude_flip_batch(eeg_batch_sample,)
                    inline_func_save_augment_eeg()

                elif augment_str == 'time_flip':
                    suffix_str = ''                            
                    augment_eeg = time_flip_batch(eeg_batch_sample,)
                    inline_func_save_augment_eeg()

                elif augment_str == 'shift':
                    # 21P:  timepoint_num, electrode_num = 525, 96
                    # shift_timepoint_num_list = [10,20,50,100,525-10,525-20,525-50,525-100]
                    # shift_timepoint_num_list = [10,20,30,525-10,525-20,525-30,]

                    # 15S:  timepoint_num, electrode_num = 32,124
                    shift_timepoint_num_list = [1,2,3,32-1,32-2,32-3,]
                    for shift_timepoint_num in shift_timepoint_num_list:
                        suffix_str = str(shift_timepoint_num)                                
                        augment_eeg = shift_batch(eeg_batch_sample,shift_timepoint_num)
                        inline_func_save_augment_eeg()

                elif augment_str == 'noise':
                    # scale_list = [0.1,0.2,0.3]
                    scale_list = [0.05,0.04,0.03,0.02,0.01]
                    # scale_list = [0.005,0.0025,]
                    for scale in scale_list:
                        suffix_str = str(scale).replace('.','p')                                
                        augment_eeg = add_noise_batch(eeg_batch_sample,scale)
                        inline_func_save_augment_eeg()

                elif augment_str == 'mask':
                    # 21P:  timepoint_num, electrode_num = 525, 96
                    # mask_num_len_list = [ (1,10) , (2,10) , (4,10), 
                    #                         (1,20), (2,20), 
                    #                         (1,40),]

                    # 15S:  timepoint_num, electrode_num = 32,124
                    mask_num_len_list = [ (1,1) , (2,1) , (4,1), 
                                            (1,2), (2,2), 
                                            (1,4),]
                    for mask_num,mask_len in mask_num_len_list:
                        suffix_str = str(mask_num)+'-'+str(mask_len)
                        augment_eeg = continous_mask_batch(eeg_batch_sample  ,mask_num,mask_len)
                        inline_func_save_augment_eeg()
                        



                

            print('\t',Fore.GREEN+fold_dir,'saved!')
        print(Fore.RED+sub_dir,'saved!')
    print(Fore.RED+data_dir,'saved!')








