import numpy as np
import random


 

# 电极对称
# 22G的电极对称
electrode_list_22G = ['Fp1','F3','F7','FT9','FC5','FC1','C3','T7','TP9','CP5','CP1','Pz','P3','P7','O1','Oz','O2','P4','P8','TP10','CP6','CP2','Cz','C4','T8','FT10','FC6','FC2','F4','F8','Fp2','AF7','AF3','AFz','F1','F5','FT7','FC3','FCz','C1','C5','TP7','CP3','P1','P5','PO7','PO3','POz','PO4','PO8','P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8','F6','F2','AF4','AF8',]
def get_symmetry_ch_order(electrode_list):
    origin_ch_order_dict = { ch_name:index for index, ch_name in enumerate(electrode_list) }
    symmetry_ch_order = []
    for chNO, ch_name in enumerate(electrode_list):
        symmetry_ch_name = ''
        # 根据命名规律找出对应ch名
        if ch_name[-1] == 'z': # 不变
            symmetry_ch_name = ch_name
        elif (ch_name[-1] in ['0','1','2','3','4','5','6','7','8','9',]):
            if (ch_name[-2] in ['0','1','2','3','4','5','6','7','8','9',]): # 后两位都是数字
                character_str = ch_name[:-2]
                number_str = ch_name[-2:]
            else: # 只有最后一位是数字
                character_str = ch_name[:-1]
                number_str = ch_name[-1:]
            number_ch = int(number_str)
            if (number_ch%2)==0: # 偶数
                symmetry_number_ch = number_ch-1
            elif (number_ch%2)==1: # 奇数
                symmetry_number_ch = number_ch+1
            symmetry_ch_name = character_str+str(symmetry_number_ch)
        else:
            assert False,'Unhandlable channel name: '+ch_name
        # 根据symmetry_ch名得到原始ch名对应的index
        symmetry_ch_index = origin_ch_order_dict[symmetry_ch_name]
        symmetry_ch_order.append(symmetry_ch_index)              
    return symmetry_ch_order

symmetry_ch_order_22G = get_symmetry_ch_order(electrode_list_22G)
def electrode_reorder(eeg_a_sample,new_ch_order):
    timepoint_num, electrode_num = eeg_a_sample.shape
    augment_eeg = np.zeros([timepoint_num, electrode_num])
    augment_eeg = eeg_a_sample[:,new_ch_order]
    return augment_eeg


##
def amplitude_flip(eeg_a_sample,):
    timepoint_num, electrode_num = eeg_a_sample.shape
    augment_eeg = np.zeros([timepoint_num, electrode_num])
    augment_eeg = -1 * eeg_a_sample
    return augment_eeg


##
def time_flip(eeg_a_sample,):
    timepoint_num, electrode_num = eeg_a_sample.shape
    augment_eeg = np.zeros([timepoint_num, electrode_num])
    for i in range(timepoint_num):
        augment_eeg[i,:] = eeg_a_sample[timepoint_num-i-1,:]
    return augment_eeg


##
shift_timepoint_num_list = [100,200,300,400]
def shift(eeg_a_sample,shift_timepoint_num):
    timepoint_num, electrode_num = eeg_a_sample.shape
    augment_eeg = np.zeros([timepoint_num, electrode_num])
    for i in range(timepoint_num):
        shifted_i = (i+shift_timepoint_num)%timepoint_num
        augment_eeg[i,:] = eeg_a_sample[shifted_i,:]
    return augment_eeg
def shift_back(eeg_a_sample,shift_timepoint_num):
    timepoint_num, electrode_num = eeg_a_sample.shape
    augment_eeg = np.zeros([timepoint_num, electrode_num])
    for i in range(timepoint_num):
        shifted_i = (i-shift_timepoint_num+timepoint_num)%timepoint_num
        augment_eeg[i,:] = eeg_a_sample[shifted_i,:]
    return augment_eeg


##
def add_noise(eeg_a_sample,scale=0.1,seed=1):
    timepoint_num, electrode_num = eeg_a_sample.shape
    augment_eeg = np.zeros([timepoint_num, electrode_num])
    np.random.seed(seed)
    gauss_noise = np.random.normal(loc=0, scale=1, size=eeg_a_sample.shape)
    augment_eeg = eeg_a_sample + scale * gauss_noise
    return augment_eeg


##
def continous_mask( eeg_a_sample  ,mask_num,mask_len):
    time_num,electrode_num=eeg_a_sample.shape
    augment_eeg = eeg_a_sample.copy()

    #每个样本mask的数量*电极数  
    total_mask_num=mask_num*electrode_num
    rand_size=(mask_num,electrode_num)
    # [low, high)
    rand_mask_start = np.random.randint( 
        low=0, high=time_num-mask_len+1, size=rand_size)
    #不同电极 加上等差数列 方便后面拉成一维后实现批量mask
    arithmetic_progression=np.arange(start=0,stop=electrode_num*time_num ,step=time_num, dtype=np.int32)
    #加上等差数列
    rand_mask_start_plus=(rand_mask_start+arithmetic_progression)

    # 拉成一维，方便实现批量mask
    # 由于electrode维位于time后面，按照python的规则，我要先调换顺序，才符合我的间隔time_num的想法  
    #### 或者不调换顺序，前面的等差数列用 63为步长
    augment_eeg=augment_eeg.transpose(1,0).reshape(-1,)

    # 置零
    mask_st_list = list(rand_mask_start_plus.reshape(-1,))
    random.shuffle(mask_st_list)
    # mask_num_type1_zero=int(0.8*total_mask_num)
    mask_num_type1_zero = total_mask_num
    #split
    mask_st_list_1=mask_st_list[0:mask_num_type1_zero]
    #转回numpy  并添加mask_len长度   # 1.如果shape相同，则是逐个元素运算    # 2. 如果shape不同，需要考虑广播broadcast
    range_mask_len=np.arange(mask_len).reshape(1,-1)
    mask_1=np.array(mask_st_list_1).reshape(-1,1)+range_mask_len
    #广播后多了一个维度，要reshape（顺序不重要，不需要transpose）
    mask_1= mask_1.reshape(-1,)
    #mask1 置零
    augment_eeg[mask_1]=0    

    augment_eeg=augment_eeg.reshape(electrode_num,time_num).transpose(1,0)
    return augment_eeg 



