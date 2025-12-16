


from colorama import init, Fore,  Back,  Style
init(autoreset = True)

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
# loss acc 记录 相关函数
def initial_loss_acc_dict():
    keys = [ 'loss_ce',   'accuracy',]
    values = [0,0,]
    multi_loss_acc_dict=dict(zip(keys,values))
    return multi_loss_acc_dict

def tensor_loss_acc_dict(loss_acc_dict, loss_ce, 
                        right_num,):
    # print(loss_total)
    loss_acc_dict['loss_ce'] += loss_ce.item()
    loss_acc_dict['accuracy'] += right_num
    return loss_acc_dict

def add_loss_acc_dict(dict_loss_acc_epoch,dict_loss_acc_batch):
    for key in dict_loss_acc_epoch.keys():
        dict_loss_acc_epoch[key] +=  dict_loss_acc_batch[key]
    return dict_loss_acc_epoch

def divide_loss_acc_dict(dict_loss_acc_epoch,batch_num,sample_num):
    for key in dict_loss_acc_epoch.keys():
        if 'accuracy' in key:
            dict_loss_acc_epoch[key] /= sample_num
        else:
            dict_loss_acc_epoch[key] /= batch_num
    return dict_loss_acc_epoch

def print_loss_acc_dict(loss_acc_dict, train_or_val_phase, epoch, step, batch_num,time_duration):
    if train_or_val_phase == 'train':
        fore_color = Fore.GREEN 
        # back_color = Back.GREEN 
    elif train_or_val_phase == 'val':
        fore_color = Fore.RED 
        # back_color = Back.RED 

    #打印用时
    # print('epoch:{} step:{} batch_num:{} time_duration:{}'.format(epoch,step,batch_num,time_duration))
    #打印loss
    print(fore_color + '{}:  '.format(train_or_val_phase))
    if loss_acc_dict['accuracy'] >= 1: #说明是right_num，而不是accuracy。 除非right_num=0
        print(fore_color + '    loss_ce_____:{:.6f} \t accuracy:{:d}'.format(
            loss_acc_dict['loss_ce'],  int(loss_acc_dict['accuracy']) ) )
    else:  # accuracy按百分比打印
        print(fore_color + '    loss_ce_____:{:.6f} \t accuracy:{:.2f}% '.format(
            loss_acc_dict['loss_ce'],  loss_acc_dict['accuracy']*100) )



def dict_tolist_loss_acc(dict_loss_acc):
    list_loss_acc =[ dict_loss_acc['loss_ce'],dict_loss_acc['accuracy'],]
    return list_loss_acc


