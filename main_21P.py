
sample_num_per_class_list = [125,250,500,1000-100]
sample_num_per_class_list = [1000-100]


import os
cudaNO_list = [        替换用         ]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cudaNO_list))  # 一般在程序开头设置
os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '123'+str(cudaNO_list[0])+str(cudaNO_list[0])
os.environ['MASTER_PORT'] = '123'+str(cudaNO_list[0])+str(cudaNO_list[0]+1)
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
# 常用库 + torch
import time
import logging
from colorama import init, Fore,  Back,  Style
init(autoreset = True)
from sklearn.model_selection import StratifiedKFold,KFold
import torch
from torch import optim
from torch.optim import lr_scheduler
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
#分布式 混合精度
import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
#my-code #~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
from parameter.parameter import *
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

from model.fc_model import ProjNet_FC
from model.model_linear import ModelLinear
from model.reweight_model import ModelLinearReweight
# #~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
from train.train_ce import *
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
from save.utils_loss_acc_dict import *
from save.utils_save import *
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
from utils.utils import *
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
from torch.cuda.amp import GradScaler
from accelerate import Accelerator, DistributedType

scaler = GradScaler()
#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
accelerator = Accelerator()
accelerator_device = accelerator.device
print('accelerator_device:{}'.format(accelerator_device))
distributed_type = accelerator.distributed_type
print('distributed_type:{}'.format(distributed_type))

#
world_size =  len(cudaNO_list)
rank=0

data_dir_list_list = [
            [ '21Purdue-origin-10fold','21Purdue-origin-10fold_mask1-4'],
            [ '21Purdue-origin-10fold','21Purdue-origin-10fold_noise0p02'],
            ['21Purdue-origin-10fold',],
            ['21Purdue-origin-10fold','21Purdue-origin-10fold_noise0p03',] ,  
]
target_fold_list = [0,1,2,3,4,5,6,7,8,9]
fold_num = 10
target_fold_list = [target_fold_list[替换用]]

# 一、parameter
# dataset
clamp_thres, norm_type, mask_num, mask_len, sub_list = parameter_dataset()

# train
seed, start_epoch, end_epoch, load_latest_model, csv_overwrite_flag, train_batch_size, val_batch_size = parameter_train()
# regularize
dropout_trans, dropout_conv, dropout_fc, weight_decay = parameter_regularize()
# optimizer
beta_1, beta_2, eps, learn_rate, gamma = parameter_optimizer()




# ~ ~ ~ ~ ~ ~
# 选择数据集
# 15Stanford_6class
# 15Stanford_72pic
# 21Purdue
# 22Australia_test
# 22Germany_test
dataset_str = '21Purdue'
end_epoch = 50
func_dataset,sub_list_list = choose_dataset(dataset_str)

single_multi_cross = 'singleSub'

# ~ ~ ~ ~ ~ ~

# loss
criterion_ce = parameter_loss()

for data_dir_list in data_dir_list_list:
    print('已完成：{} done!'.format(str(data_dir_list)))

    for sample_num_per_class in sample_num_per_class_list:
        # 
        lr_list = [ 1e-3, ]
        bs_list = [ 256 ]  # 25790MB  val_batch_size=1334      38308MB  val_batch_size=1334 

        wd_list = [ 0 ]
        dp_fc_list = [ 0.1 ]
        dp_conv_list = [ 0.25,]
        # dp_trans = 0.1

        fc_hid_list_list = [
            [256],
            ]
        # 逐一配对版
        parameter_list_whole = [ (a,b,c,d,e,f)    
                                    for a in   lr_list 
                                    for b in   bs_list 
                                    for c in   wd_list
                                    for d in   dp_fc_list 
                                    for e in   dp_conv_list
                                    for f in   fc_hid_list_list   ] 
        parameter_list_this_cuda = parameter_list_whole[:]
        for ( learn_rate, train_batch_size , 
            weight_decay , dropout_fc , dropout_conv,
            fc_hid_list ) in parameter_list_this_cuda:

            dataset_str_for_save = get_dataset_str_for_save(data_dir_list,dataset_str)

            # 归一化
            norm_per_sample = False
            norm_per_electrode = False
            norm_per_2sample_electrode = False
            if norm_type=='norm_per_sample':
                norm_per_sample = True
                norm_type_short= 'nps'
            elif norm_type=='norm_per_electrode':
                norm_per_electrode = True
                norm_type_short= 'npe'
            elif norm_type=='norm_per_2sample_electrode':
                norm_per_2sample_electrode = True
                norm_type_short= 'np2se'
            elif norm_type=='none':
                norm_per_2sample_electrode = True
                norm_type_short= 'none'

            # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
            # 二、data    选择模型   

            # 选择模型
            model_str_list=['linear', 'conv', 'ShallowConvNet', 'EEGNet', 'MetaEEG', 'NICE', 'ATMS_50', 'SCT','SCT_pool', ]
            model_str_list=['SCT', ]
            model_str_list=['SCT_pool', ]
            # 22A 22G不pool性能好   # 15S 21P 可能 pool性能好

            from itertools import product
            result = product(sub_list_list, model_str_list, )
            for sub_list, model_str in result:
                print('sub_list:',sub_list, '\tmodel_str:', model_str)

                # ~ 根据前面的参数，设定保存的parameter_str_short ~ ~ ~ ~ ~
                # save
                save_code_dir, save_model_dir, save_dataset_dir, parameter_str_short, save_dir_ckpt, save_dir_csv, save_dir_csv_batch = parameter_save(learn_rate,train_batch_size,world_size,seed,
                    dataset_str_for_save+'_'+str(sample_num_per_class), single_multi_cross,
                    dropout_fc,dropout_conv,weight_decay,
                    fc_hid_list)

                sub_num = len(sub_list)
                save_dir_ckpt_this_sub = save_dir_ckpt.replace(save_dataset_dir,save_dataset_dir+'_subNum'+str(sub_num)+'_'+sub_list[0])
                save_dir_csv_this_sub = save_dir_csv.replace(save_dataset_dir,save_dataset_dir+'_subNum'+str(sub_num)+'_'+sub_list[0])
                save_dir_csv_batch_this_sub = save_dir_csv_batch.replace(save_dataset_dir,save_dataset_dir+'_subNum'+str(sub_num)+'_'+sub_list[0])

                for foldNO_val in range(fold_num):
                    #重新将start_epoch置零
                    start_epoch = 0

                    # 跳过非指定fold
                    if not foldNO_val in target_fold_list:
                        continue

                    # save命名
                    fold_str = '_foldNO-'+str(foldNO_val).zfill(2)
                    # 文件名(loss)，ckpt的要记录epoch和loss/acc
                    time_str = timestamp_to_date(time_stamp=time.time(), format_string = "%Y-%m-%d-%H-%M-%S")
                    csv_fname = fold_str+'_'+ time_str + '.csv'
                    checkpoint_fname_rear =  fold_str+'_'+ '_checkpoint.pt'

                    # dataset
                    start_time = time.time()
                    train_or_val = 'train'
                    trainset = func_dataset(data_dir_list,
                                            sub_list,
                                            foldNO_val, train_or_val,
                                            clamp_thres,
                                            norm_per_sample, norm_per_electrode, norm_per_2sample_electrode, )
                    print(' load time:', time.time() - start_time)
                    sample_num = len(trainset)
                    print(train_or_val, ' 样本数:', sample_num)
                    #
                    start_time = time.time()
                    train_or_val = 'val'
                    val_data_dir_list = ['21Purdue-origin-10fold']
                    valset = func_dataset(val_data_dir_list,
                                            sub_list,
                                            foldNO_val, train_or_val,
                                            clamp_thres,
                                            norm_per_sample, norm_per_electrode, norm_per_2sample_electrode, )
                    print(' load time:', time.time() - start_time)
                    sample_num = len(valset)
                    print(train_or_val, ' 样本数:', sample_num)

                    # 样本维度
                    eeg, label_onehot, imgNO = valset.__getitem__(1)  # 测试，ps：python并无私有
                    timepoint_num, electrode_num = eeg.shape
                    class_num = label_onehot.shape[0]
                    print('timepoint_num:{} electrode_num:{} class_num:{}'.format(timepoint_num, electrode_num, class_num))

                    # # 缩小训练集的训练样本量
                    # trainset.divide(sample_num_per_class) 
                    # valset.divide(None) 

                    datasets={ 'train':    trainset,
                                'val':       valset, } 
                    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
                    print(' train_size:{} val_size:{}'.format(
                        len(trainset), len(valset)))
                    #  
                    val_batch_size=int(train_batch_size*2)#测试时无梯度，可以大一点
                    val_batch_size=1334 
                    val_batch_size=2000 
                    print('train_batch_size:{} val_batch_size:{} '.format(
                        train_batch_size, val_batch_size))
                    # num_workers = 8,
                    # pin_memory = True,
                    # prefetch_factor = 4,
                    trainloader = torch.utils.data.DataLoader(trainset, train_batch_size, shuffle=True)
                    valloader = torch.utils.data.DataLoader(valset, val_batch_size,shuffle=False)
                    dataloaders = {'train':    trainloader,
                                    'val':      valloader, }    

                    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
                    # 三、model   
                    print('开始构建模型训练') 
                    # model 
                    pool_type = 'mean'  # 好很多
                    seed_everything(seed)
                    subject_num = 1 # MetaEEG用，但是我这里没有区分被试
                    model = get_model_encoder(model_str, dataset_str, 
                        timepoint_num, electrode_num, 
                        class_num, 
                        subject_num,
                        vis_emb_dim=None)

                    print('Start -- model.to(rank)')
                    start_time = time.time()
                    model = model.to(rank)
                    print('model.to(rank) cost:{}'.format(time.time()-start_time))
                    # 优化器
                    optimizer = optim.Adam(model.parameters(), lr=learn_rate,
                                        betas=(beta_1, beta_2), eps=eps, weight_decay=weight_decay)
                    opt_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma) 

                    # # ##############
                    in_shape = [timepoint_num, electrode_num ]
                    out_shape,proj_in_dim = test_model_output_shape(model,rank,in_shape)
                    # # ##############
                    #加分类头
                    fc_in_dim = proj_in_dim
                    fc_out_dim = class_num
                    hid_dim_list = fc_hid_list
                    
                    fc_model=  ProjNet_FC(
                        fc_in_dim,fc_out_dim,
                        hid_dim_list,
                        activate_func='relu',
                        whether_last_layer_act=True,
                        last_layer_act_func='softmax',
                        dropout_fc=dropout_fc)
                    fc_model = fc_model.to(rank)
                    fc_optimizer = optim.Adam(fc_model.parameters(), lr=learn_rate,
                                        betas=(beta_1, beta_2), eps=eps, weight_decay=weight_decay)
                    fc_scheduler = lr_scheduler.ExponentialLR(fc_optimizer, gamma)
                    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
                    model_list = [model]+[fc_model]
                    optimizer_list = [optimizer]+[fc_optimizer]
                    opt_scheduler_list =[opt_scheduler]+[fc_scheduler]

                    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
                    # 四、train + logging + save
                    val_min_acc = 0
                    val_min_loss = float('inf')
                    csv_overwrite_flag_this_train=csv_overwrite_flag
                    # epoch自适应
                    acc_not_increase_flag_list = {'train':0,'val':0} 
                    max_acc_list = {'train':0,'val':0} 

                    # epoch自适应
                    for epoch in range(start_epoch, end_epoch*100):
                        start_time = time.time()
                        list_acc_loss = []
                        list_acc_loss.append(epoch)
                        print("epoch:{}".format(epoch)) 
                        for train_or_val_phase in ['val', 'train']:
                            dataloader = dataloaders[train_or_val_phase]
                            (dict_loss_acc_epoch, confusion_mat_epoch)  = train(
                                    train_or_val_phase, 
                                    model_list,  optimizer_list,  opt_scheduler_list, 

                                    criterion_ce,  

                                    dataloader, 

                                    scaler, 
                                    rank, world_size, 

                                    epoch, 
                                    save_path_csv_batch=save_dir_csv_batch_this_sub,  
                                    csv_fname=csv_fname, 
                                    overwrite_flag=csv_overwrite_flag, 
                                    parameter_info_thorough=parameter_str_short, 
                                        )
                            # 参数 csv_fname, overwrite_flag, parameter_info_thorough,  并未使用
                                    
                            list_loss_acc_epoch=dict_tolist_loss_acc(dict_loss_acc_epoch)
                            list_acc_loss+=list_loss_acc_epoch
                            '''
                            # 用于save best模型 
                            if train_or_val_phase=='val':
                            # if epoch%1 ==0:
                                val_loss=dict_loss_acc_epoch['loss_total']
                                val_acc=dict_loss_acc_epoch['accuracy']
                                # 保存当前的 及 最好的测试性能的 模型和优化器
                                # val_loss = list_acc_loss[pos_val_loss]#!!!这里一定要注意val_loss在list_acc_loss中的位置!!!

                                # acc和loss各存一个
                                best_acc_or_loss = 'loss'
                                val_min_loss = save_checkpoint_best_and_current(
                                                    save_dir_ckpt_this_sub, checkpoint_fname_rear,
                                                    epoch, model, optimizer,
                                                    val_loss, val_min_loss, 
                                                    best_acc_or_loss,
                                                    accelerator,
                                                    whether_remove_old_model=True)

                                # acc和loss各存一个
                                best_acc_or_loss = 'acc'
                                val_min_acc = save_checkpoint_best_and_current(
                                                    save_dir_ckpt_this_sub, checkpoint_fname_rear,
                                                    epoch, model, optimizer,
                                                    val_acc, val_min_acc, 
                                                    best_acc_or_loss,
                                                    accelerator,
                                                    whether_remove_old_model=True)'''

                            # print
                            step=None
                            if epoch == start_epoch:#打印1个epoch的用时
                                time_duration=time.time()-start_time
                            else:
                                time_duration
                            # 新加入了batch_num，是epoch内查看进度用的
                            print_loss_acc_dict(dict_loss_acc_epoch, train_or_val_phase, epoch, step, 
                                                batch_num=len(dataloader), time_duration=time_duration)
                            # # 保存当前train_or_val_phase的confusion矩阵
                            # save_confusion_matrix(save_path_confusion_mat_this_para, epoch, train_or_val_phase, confusion_mat_epoch_nomask,  'nomask')
                            # save_confusion_matrix(save_path_confusion_mat_this_para, epoch, train_or_val_phase, confusion_mat_epoch_mask,  'mask')

                            # epoch自适应
                            acc_this_epoch  = dict_loss_acc_epoch['accuracy']
                            if acc_this_epoch > max_acc_list[train_or_val_phase]:
                                max_acc_list[train_or_val_phase] = acc_this_epoch
                                acc_not_increase_flag_list[train_or_val_phase]  = 0
                            else:
                                acc_not_increase_flag_list[train_or_val_phase]  += 1            

                        # 保存loss 总的准确率 每一类准确率（train和test）
                        save_acc_loss(
                            save_dir_csv_this_sub, csv_fname,
                            list_acc_loss,
                            csv_overwrite_flag_this_train,
                            parameter_info_thorough=parameter_str_short)
                        #*-重要-*  csv_overwrite_flag_this_train
                        csv_overwrite_flag_this_train=0#保存过1次后，就置零

                        # epoch自适应
                        # 假如训练集验证集准确率都不再增长，则终止训练
                        if epoch>end_epoch:
                            if acc_not_increase_flag_list['train']>20 or acc_not_increase_flag_list['val']>30:
                                print(Fore.RED+'Training ended at epoch',epoch)
                                break

                    print('foldNO {} done!'.format(foldNO_val))
                print('{} {} done!'.format(sub_list, model_str))
            print('已完成：{} done!'.format(parameter_str_short))












