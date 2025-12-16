
import os
import logging
from torch import nn
import numpy as np

from .loss import SupervisedContrastiveLoss_myrevised, BatchAllTripletLoss,InfoNCE,KLDivLoss
from utils.utils import main_dir


def parameter_dataset():
    clamp=500
    norm_type='norm_per_sample'
    
    mask_num=1
    mask_len=1
    logging.info('mask_num:{},mask_len:{}'.format(mask_num, mask_len))

    # 数据
    sub_list = None
    return clamp, norm_type, mask_num, mask_len, sub_list


def parameter_save(learn_rate,train_batch_size,world_size,seed,
dataset_str_for_save, single_multi_cross,
dropout_fc,dropout_conv,weight_decay,
fc_hid_list): 
    # ~ 运行前需修改的 ~ ~ ~ ~ ~
    # 存储路径 代码 数据集 模型
    save_share_path = main_dir+'/models/'
    save_code_dir='04-16-21P-数据增广-不同数据、增广量'
    # save_dataset_dir=dataset_str_for_save + '-' + single_multi_cross
    save_dataset_dir=dataset_str_for_save 
    save_model_dir='newSCT'

    # ~ 不需设定的 ~ ~ ~ ~ ~
    # 参数信息
    # 仅限train相关，模型和数据集参数要手动设定save_model_dir和save_dataset_dir
    parameter_str_short =  'loss_' + \
                '_lr'+str(learn_rate) + \
                '_bs'+str(train_batch_size) + \
                '_wd'+str(weight_decay) + \
                '_sd'+str(seed)  +  \
                '_dFC'+str(dropout_fc) + \
                '_dCv'+str(dropout_conv) + \
                '_fH'+str(fc_hid_list) 

    # 文件夹路径
    save_dir_ckpt = os.path.join(save_share_path,'checkpoint',save_code_dir, save_dataset_dir, save_model_dir, parameter_str_short)
    save_dir_csv = os.path.join(save_share_path,'loss_acc',save_code_dir, save_dataset_dir, save_model_dir, parameter_str_short)
    save_dir_csv_batch = os.path.join(save_share_path,'loss_acc_batch',save_code_dir, save_dataset_dir, save_model_dir, parameter_str_short)
    return save_code_dir, save_model_dir, save_dataset_dir, parameter_str_short, save_dir_ckpt, save_dir_csv, save_dir_csv_batch


def parameter_train():
    # 随机种子
    seed = 50
    # 迭代
    start_epoch = 0
    end_epoch = 100
    # 续跑参数 
    load_latest_model = 1
    csv_overwrite_flag = 0
    # batch_size
    train_batch_size = 125  
    val_batch_size=int(train_batch_size*2)#测试时无梯度，可以大一点    
    return seed, start_epoch, end_epoch, load_latest_model, csv_overwrite_flag, train_batch_size, val_batch_size


def parameter_regularize():
    #dropout
    dropout_trans=0.1
    dropout_conv=0.25
    dropout_fc=0
    weight_decay = 0
    logging.info('weight_decay:{},dropout_trans:{},dropout_conv:{},dropout_fc:{},'.format(weight_decay,dropout_trans,dropout_conv,dropout_fc))
    return dropout_trans, dropout_conv, dropout_fc, weight_decay   


def parameter_optimizer():
    # 优化器参数
    # 平滑常数\beta_1和\beta_2
    beta_1 = 0.9
    beta_2 = 0.999
    # eps加在分母上防止除0
    eps=1e-08
    # 学习率
    learn_rate = 1e-5*100
    logging.info('learn_rate:{}'.format(learn_rate))
    gamma=1  # opt_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1)
    return beta_1, beta_2, eps, learn_rate, gamma 


def parameter_loss():
    criterion_ce = nn.CrossEntropyLoss()  
    return criterion_ce
    

