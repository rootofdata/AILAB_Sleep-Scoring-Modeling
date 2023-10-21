from loader import data_loader
from multiprocessing import cpu_count
import argparse
import csv
import torch
from torch import nn, optim
from tqdm import tqdm
from model_resnet import Resnet50, Resnet34, Resnet18
from model_efficientnet import EfficientNet
from torch.autograd import Variable
import torch.autograd.profiler as profiler
from torch.optim.lr_scheduler import StepLR
from grad_cam import GradCam
import datetime
import os
import slack
import json
import requests

def compute_result(prob, label):
    TP, FP, FN, TN = 0, 0, 0, 0
    Total = 0
    for p, l in zip(prob, label):
        l = int(l)
        if p==1 and l==1 :
            TP+=1
        elif p!=1 and l==1 :
            FN +=1
        elif p==1 and l!=1 :
            FP+=1
        elif p!=1 and l!=1 :
            TN+=1
        Total = TP+FN+FP+TN
    log = 'TP : {}, FN : {}, FP : {}, TN : {} \n'.format(TP, FN, FP, TN)
    
    try:
        Acc = round((TP+TN)/Total*100,2)
    except ZeroDivisionError as identifier:
        Acc = -1
    try:
        Pre = round(TP/(TP+FP)*100,2)
    except ZeroDivisionError as identifier:
        Pre = -1
    try:
        Recall = round(TP/(TP+FN)*100,2)
    except ZeroDivisionError as identifier:
        Recall = -1
    try:
        Spec = round(TN /(TN+FP)*100,2)
    except ZeroDivisionError as identifier:
        Spec = -1
    try:
        F1 = round(2*Pre*Recall/(Pre+Recall),2)
    except ZeroDivisionError as identifier:
        F1 = -1


    return Acc, Pre, Recall, F1, Spec, log

def save_model(model, optimizer, scheduler, epoch, time, memo, fold, f2, std, slack_flag) : 
    model_cpu = model.to('cpu')
    state = {
        'model' : model_cpu.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()
    }
    if not(os.path.isdir('./saved_model')) : os.mkdir('./saved_model')
    if not(os.path.isdir('./saved_model/'+time+'_fold'+str(fold)+'_'+memo)) : os.mkdir('./saved_model/'+time+'_fold'+str(fold)+'_'+memo)

    try:
        if not(os.path.isdir('./saved_model')) : os.mkdir('./saved_model')
        if not(os.path.isdir('./saved_model/'+time+'_fold'+str(fold)+'_'+memo)) : os.mkdir('./saved_model/'+time+'_fold'+str(fold)+'_'+memo)
        torch.save(state, './saved_model/'+time+'_fold'+str(fold)+'_'+memo+'/'+str(epoch)+'_'+std+'.pth')
        text = 'Model_'+std+' save is sucessed\n'
        print(text)
        f2.write(text)  
    except :
        text = 'Model_'+std+' save is failed\n'
        print(text)
        f2.write(text)
        if slack_flag : send_task_stat(memo, text, '', '', '')
        pass

#나중에 이 쪽 수정해야함 (url 비공개)
web_hook_url = 'https://hooks.slack.com/services/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'
def send_msg(msg):
    data = {'text': msg}
    response = requests.post(
        web_hook_url, data=json.dumps(data), headers={'Content-Type': 'application/json'}
    )
    return response


def send_task_stat(memo, log1, log2, log3, log4):
    msg = f'{memo}\n{log1} \n {log2} \n {log3} \n {log4}'
    return send_msg(msg)

def train_val(path_dataset, args):
    batch_size = args.batch
    label_mode = args.label
    learning_rate = args.lr
    num_epoch = args.epoch
    val_std = args.val_std
    num_worker = args.worker
    log_flag = args.log
    fold = args.fold
    gpu_number = str(args.gpu)
    memo = str(args.memo)
    slack_flag = args.slack
    print_iter = args.iter
    threshold = args.threshold
    model_name = args.model
    light = args.light
    sampler_flag = args.sampler
    time = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if log_flag :
        if not(os.path.isdir('./log')) : os.mkdir('./log')
        os.mkdir('./log/'+time+'_fold'+str(fold)+'_'+memo)
        f1 = open('./log/'+time+'_fold'+str(fold)+'_'+memo+'/train_log.txt', 'w', encoding='utf-8-sig')
        f2 = open('./log/'+time+'_fold'+str(fold)+'_'+memo+'/val_log.txt', 'w', encoding='utf-8-sig')
        f1.writelines(str(args))
        f1.writelines('\n')
        f1.writelines('Fold-{} Dataset from {} \n'.format(fold, (path_dataset +str(seq_len))))
    
    device = torch.device("cuda:"+gpu_number if torch.cuda.is_available() else "cpu")
    
    #################
    
    try :
        if model_name == 'resnet50' : model = Resnet50(batch_size=batch_size, light = light)
        elif model_name == 'resnet34' : model = Resnet34(batch_size=batch_size, light = light)
        elif model_name == 'resnet18' : model = Resnet18(batch_size=batch_size, light = light)
        elif model_name == 'effi' : model = EfficientNet.from_name(model_name = 'efficientnet-b0', in_channels=1, num_classes=1)
    except :
        print('Model Imort Error')
    loss_fn = nn.BCELoss()
    
    optimizer = optim.Adam(
        [param for param in model.parameters() if param.requires_grad], 
        lr = learning_rate, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size = 5, gamma =0.5)
    model = model.to(device, non_blocking=True)

    print('Num of parameters :', sum(p.numel() for p in model.parameters() if p.requires_grad))    

    early_stop_count = 0
    early_stop_std = 10
    loss_min = 10.0
    best_f1 = 0.0 
    goal_f1 = 0.0
    
    train_loader = data_loader(phase='train', path_dataset = path_dataset, seq_len = seq_len, fold=fold, label_mode = label_mode, batch_size = batch_size, sampler_flag = sampler_flag, num_workers= int(cpu_count()*num_worker/100))
    for epoch in range(0, num_epoch) :
        model = model.to(device, non_blocking=True)
        print("Train progress : {}/{} epoch".format(epoch, num_epoch))
        
        count = 0
        loss_e = 0
        prob_epoch = []
        labels_epoch = []
        model.train()
        # Start Train
        st = datetime.datetime.now()
        num_batch = (len(train_loader))
        print('Num of Train batch : ', num_batch)
        for image, label, _, _ in tqdm(train_loader):
            count +=1
            epoch_ = epoch + round(count/num_batch,2)
            image = image.to(device)
            label = label.type(torch.FloatTensor)
            label = label.to(device)
            out = model(image)
            
            out = out.view(-1)
            label = label.view(-1)
            loss = loss_fn(out, label)
            loss = loss.to(device)
            loss_e += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            labels_epoch.extend(label.tolist())
            prob = (out>threshold).long()
            prob_epoch.extend(prob.tolist())

            if count%print_iter == 0 : 
                Acc, Pre, Recall, F1, Spec, log = compute_result(prob_epoch, labels_epoch)
                loss_tmp = loss_e / count
                log1 = 'Epoch : {}/{} \n'.format(epoch_, num_epoch)
                log2 ='Train Loss avg : {}\n'.format(loss_tmp)
                log3 ='Train Result Acc : {}, Precision : {}, Recall : {}, F1 : {}, Specificity : {}\n'.format(Acc, Pre, Recall, F1, Spec)  
                print("-----------------------------------------------")
                print(log1)
                print(log2)
                print(log)
                print(log3)
                if log_flag :
                    f1.write(log1)
                    f1.write(log2)
                    f1.write(log)
                    f1.write(log3)
                    f1.write('---------------------------------\n')
            
            # Perform validation every n minutes
            now = datetime.datetime.now()
            duration = now-st
            if duration > datetime.timedelta(seconds=60 * val_std) or count == num_batch :
                st = datetime.datetime.now()
                # Validation
                count_v = 0
                loss_e_v = 0
                prob_epoch_v = []
                labels_epoch_v = []
                model.eval()
                val_loader = data_loader(phase='val', path_dataset = path_dataset, seq_len = seq_len, fold=fold, label_mode = label_mode, batch_size = batch_size, num_workers= int(cpu_count()*num_worker/100))
                with torch.no_grad()  :
                    print('Progress Validation')
                    for image, label, _, _ in tqdm(val_loader):
                        
                        count_v +=1
                        image = image.to(device)
                        label = label.type(torch.FloatTensor)
                        label = label.to(device)
                        out = model(image)
                        
                        out = out.view(-1)
                        label = label.view(-1)
                        loss = loss_fn(out, label)
                        loss = loss.to(device)
                        loss_e_v += loss

                        labels_epoch_v.extend(label.tolist())
                        
                        prob = (out>threshold).long()
                        prob_epoch_v.extend(prob.tolist())

                    loss_e_v = loss_e_v / count_v

                    Acc, Pre, Recall, F1, Spec, log = compute_result(prob_epoch_v, labels_epoch_v)
                    
                    log1 = 'Epoch {}/{}\n'.format(epoch_, num_epoch)
                    log2 ='Val Loss avg : {}\n'.format(loss_e_v)
                    log3 ='Val Result Acc : {}, Precision : {}, Recall : {}, F1 : {}, Specificity : {}\n'.format(Acc, Pre, Recall, F1, Spec)  
                    print("-----------------------------------------------")
                    print(log1)
                    print(log2)
                    print(log3)
                    if log_flag :
                        f2.write(log1)
                        f2.write(log2)
                        f2.write(log)
                        f2.write(log3)   
                    if slack_flag == True :
                        send_task_stat(memo, log1, log2, log, log3)
                    flag_early = True

                    #check early stop by val loss
                    if loss_e_v < loss_min and log_flag: 
                        flag_early = False
                        print("save model loss")
                        early_stop_count = 0
                        loss_min = loss_e_v

                        save_model(model, optimizer, scheduler, epoch_, time, memo, fold, f2, 'loss', slack_flag)
                        
                    if best_f1 < F1 and log_flag: 
                        flag_early = False
                        print("save model F1")
                        early_stop_count = 0
                        best_f1 = F1

                        save_model(model, optimizer, scheduler, epoch_, time, memo, fold, f2, 'f1', slack_flag)
                        
                    if (Acc > 80 and Pre > 80 and Recall > 83 and Spec > 83 and F1 > 80) and F1 > goal_f1 :
                        flag_early = False
                        early_stop_count = 0
                        print("save model - Goal")
                        goal_f1 = F1

                        save_model(model, optimizer, scheduler, epoch_, time, memo, fold, f2, 'goal', slack_flag)
                        
                    if log_flag : f2.write('---------------------------------\n')

                    if flag_early :
                        early_stop_count+=1

                model = model.to(device, non_blocking=True)
            #End Validation
        if early_stop_count >= early_stop_std : break
        scheduler.step()
    #End Train
    ######################################
            
# def val(model, )
def test(path_dataset_list, args):
    batch_size = args.batch
    label_mode = args.label
    num_worker = args.worker
    fold = args.fold
    gpu_number = str(args.gpu)
    threshold_std = args.threshold
    model_name = args.model
    load_model = args.load
    light = args.light
    print(load_model)
    device = torch.device("cuda:"+gpu_number if torch.cuda.is_available() else "cpu")
    print('device :', device)
    time = datetime.datetime.now()
    time = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    try :
        print("1")
        if model_name == 'resnet50' : model = Resnet50(batch_size=batch_size, light = light)
        elif model_name == 'resnet34' : model = Resnet34(batch_size=batch_size, light = light)
        elif model_name == 'resnet18' : model = Resnet18(batch_size=batch_size, light = light)
        elif model_name == 'effi' : model = EfficientNet.from_name(model_name = 'efficientnet-b0', in_channels=1, num_classes=1)
        print("2")
        state = torch.load(load_model)
        print("3")
        model.load_state_dict(state['model'])
        print("4")
        model = model.to(device, non_blocking=True)
        print("5")
    except Exception as e:
        print('Model Imort Error', e)
    
    loss_fn = nn.BCELoss()

    test_loader = data_loader(phase='test', path_dataset = path_dataset, seq_len = seq_len, fold=fold, label_mode = label_mode, batch_size = batch_size, num_workers= int(cpu_count()*num_worker/100))
    if not(os.path.isdir('./test_result')) : os.mkdir('./test_result')        
    f = open('./test_result/test-'+str(fold)+'_'+str(time)+'-total.csv', 'w', encoding='utf-8-sig', newline='')
    wr = csv.writer(f)
    wr.writerow(['Patient ','Image ', 'Out ', 'Label '])
    f2 = open('./test_result/test-'+str(fold)+'_'+str(time)+'-summary.csv', 'w', encoding='utf-8-sig', newline='')
    wr2 = csv.writer(f2)

    
    print("Test Start")
        
    count = 0
    loss_e = 0
    threshold = []
    prob_epoch = []
    for i in range(0,21):
        threshold.append(round(i*0.05,2))
        prob_epoch.append([])
    print(threshold)
    labels_epoch = []
    lst_patient = []
    dic_pre = {}
    model.eval()
    with torch.no_grad()  :
        for image, label, patient, path in tqdm(test_loader):
            count+=1
            image = image.to(device)
            label = label.type(torch.FloatTensor)
            label = label.to(device)
            out = model(image)

            out = out.view(-1)
            label = label.view(-1)
            loss = loss_fn(out, label)
            loss = loss.to(device)
            loss_e += loss
            
            for p1, p2, o, l in zip(patient, path, out, label) :
                result = ''
                if o.item() >= threshold_std and int(l.item()) == 1 : result = 'TP'
                elif o.item() >= threshold_std and int(l.item()) == 0 : result = 'FP'
                elif o.item() < threshold_std and int(l.item()) == 1 : result = 'FN'
                elif o.item() < threshold_std and int(l.item()) == 0 : result = 'TN'
                log = '{}, {}, {:.4f}, {}, {}\n'.format(p1, p2[-8:], o.item(), int(l.item()), result)
                f.writelines(log)
            labels_epoch.extend(label.tolist())
            for idx, item in enumerate(threshold) :
                prob = (out>item).long()
                prob_epoch[idx].extend(prob.tolist())

            for idx, item in enumerate(patient) :
                if not(patient[idx] in lst_patient) : lst_patient.append(patient[idx])
                if not(patient[idx] in dic_pre) :
                    dic_pre[patient[idx]] = 0
                if out[idx].item() >= threshold[-1] : pre = 1
                else : pre = 0
                dic_pre[patient[idx]] += pre

        wr.writerow(['---------------------------------------------------------------'])
        for idx, item in enumerate(threshold):
            Acc, Pre, Recall, F1, Spec, log = compute_result(prob_epoch[idx], labels_epoch)
            log_ = 'Threshold {} : [Acc : {}, Precision : {}, Recall : {}, F1 : {}, Specificity : {}]'.format(item, Acc, Pre, Recall, F1, Spec)

            wr.writerow([log_])
            wr.writerow([log[:-1]])
            wr.writerow(['---------------------------------------------------------------'])

            wr2.writerow([log_])
            wr2.writerow([log[:-1]])
            wr2.writerow(['---------------------------------------------------------------'])
            print('Threshold :' + str(item) +' '+ log)
            print(log_)

            
        loss_e = loss_e/count


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='python Implementation')
parser.add_argument('--mode', type = str, default ='train')
parser.add_argument('--label', type = str, default ='')
parser.add_argument('--model', type = str, default ='resnet34')
parser.add_argument('--light', type = int, default =2, help = 'x times fewer CNN filters, Recommend :  [1, 2, 4, 8 ... , untill 64]')
parser.add_argument('--gpu', type = int, default = 0)
parser.add_argument('--loss', type = str2bool, default='False')
parser.add_argument('--seq_len', type = int, default = 3)
parser.add_argument('--batch', type = int, default = 128)
parser.add_argument('--fold', type = int, default = 1, help='K-Fold set')
parser.add_argument('--epoch', type = int, default = 1000)
parser.add_argument('--val_std', type = int, default = 60, help = 'Perform validation every n minutes')
parser.add_argument('--lr', type = float, default = 0.0001)
parser.add_argument('--worker', type = int, default = 50, help='CPU Percentage')
parser.add_argument('--sampler', type = str2bool, default ='false')
parser.add_argument('--log', type = str2bool, default ='True')
parser.add_argument('--memo', type = str, default ='')
parser.add_argument('--slack', type = str2bool, default ='True')
parser.add_argument('--iter', type = int, default =100)
parser.add_argument('--threshold', type = float, default =0.5)
parser.add_argument('--load', type = str, default ='./saved_model/2020-12-30 05:24:20_fold2_r34_fold2/5.47_goal.pth')

if __name__ == "__main__":
    args = parser.parse_args()
    seq_len = args.seq_len
    print(args)
    mode = args.mode
    seq_len = args.seq_len
    fold = args.fold
    path_dataset = './Dataset_Normalized_concat'
    print('Fold-{} Dataset from {} '.format(fold, (path_dataset +str(seq_len))))
    if mode =='train' : train_val(path_dataset, args)
    elif mode =='test' : test(path_dataset, args)
