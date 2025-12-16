

import os

os.environ["TORCH_DISTRIBUTED_DEBUG"] = 'DETAIL'

import time
import numpy as np
import torch

from save.utils_save import save_acc_loss_batch
from save.utils_loss_acc_dict import *   # 注意未放在train路径下，因为main中也用得到

from .utils_loss import *  


# train / val
def  train(
        train_or_val_phase, 
        model_list,  optimizer_list,  opt_scheduler_list, 

        criterion_ce,  

        dataloader, 

        scaler, 
        rank, world_size, 

        epoch, 
        save_path_csv_batch,  
        csv_fname, 
        overwrite_flag, 
        parameter_info_thorough, ): 
    #获取类别数，以初始化confusion_mat
    eeg, label_onehot, imgNO= dataloader.dataset.__getitem__(1)
    timepoint_num,  electrode_num = eeg.shape
    class_num = label_onehot.shape[0]
    confusion_mat_epoch = np.zeros(
        [class_num,  class_num],  dtype=np.int32)  

    #训练 or 测试
    if train_or_val_phase == 'train':
        for model in model_list:
            model.train()
    elif (train_or_val_phase == 'val') or (train_or_val_phase == 'test'):
        for model in model_list:
            model.eval()
    else:
        assert 'train_or_val_phase can only be '+'train'+' or ' + 'val'+' or ' + 'test'
    dict_loss_acc_epoch=initial_loss_acc_dict()
    sample_num = len(dataloader.dataset)
    # print('sample_num:{}'.format(sample_num)) 
    batch_num = len(dataloader)
    # print('batch_num:{}'.format(batch_num)) 
    for step,  ( eeg, label_onehot,  imgNO ) in enumerate(dataloader):
        # break
        #保存每个batch的loss acc
        dict_loss_acc_batch=initial_loss_acc_dict()
        start_time=time.time()
        with torch.set_grad_enabled(train_or_val_phase == 'train'):
            eeg = eeg.to(rank)

            label_onehot = label_onehot.to(rank)         

            # eeg和mask_eeg一起过模型 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
            # 之前调的22G的参数应该是concat版的，如果效果不佳可以改回来
            
            dec_out_normed, emb_norm = model_list[0](eeg)
            last_emb_eeg = dec_out_normed
            prob = model_list[-1](last_emb_eeg)




            # 计算loss和acc
            (loss_ce, 
                right_num, accuracy_batch, 
                confusion_mat_batch,)=get_loss_acc_ce(rank, world_size, 
                    emb_norm, prob, 
                    eeg, label_onehot, 
                    criterion_ce,   )

            #模型参数更新    # 每个batch更新一次参数
            if train_or_val_phase == 'train':
                for optimizer in optimizer_list:
                    optimizer.zero_grad()

                scaler.scale(loss_ce).backward()
                for optimizer in optimizer_list:
                    scaler.step(optimizer)
                        
                # 看是否要增大scaler
                scaler.update()
            

        #confusion_mat 不每个batch保存了，内存也占的大
        # 行：模型预测的类别  列：真实类别
        confusion_mat_epoch+=confusion_mat_batch
        #保存每个batch的loss acc
        # # SPECIAL！！concat后要除2
        # right_num = int(right_num/2)
        dict_loss_acc_batch=tensor_loss_acc_dict(dict_loss_acc_batch,    
                        loss_ce, 
                        right_num,)                    
        list_loss_acc_batch=[epoch, step, train_or_val_phase]
        list_loss_acc_batch += dict_tolist_loss_acc(dict_loss_acc_batch)
        #
        # save_acc_loss_batch(save_path_csv_batch,  
        #     csv_fname, 
        #     list_loss_acc_batch, 
        #     overwrite_flag, 
        #     parameter_info_thorough)
        overwrite_flag=0#保存过1次后，就置零

        #打印信息
        time_duration = time.time() - start_time
        #
        # if step%2 ==0:
        if step%10_000 ==9_999:
            print_loss_acc_dict(dict_loss_acc_batch,  train_or_val_phase,  epoch,  step, batch_num, time_duration)

        #加入epoch的loss里
        dict_loss_acc_epoch=add_loss_acc_dict(dict_loss_acc_epoch, dict_loss_acc_batch)

    # 每个epoch的学习率调节
    if train_or_val_phase == 'train':
        # gamma=1，其实暂未做学习率控制
        for opt_scheduler in opt_scheduler_list:
            opt_scheduler.step()

    # loss除以batch数，因为loss是对batch内的样本数取过平均的，而不是每个样本的loss相加
    dict_loss_acc_epoch=divide_loss_acc_dict(dict_loss_acc_epoch, batch_num, sample_num)

    return dict_loss_acc_epoch,  confusion_mat_epoch
    



