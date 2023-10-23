import os
import cv2
import datetime
from PIL import Image
from multiprocessing import Pool, cpu_count, Lock, Manager
from functools import partial
import json
import csv
def preprocessing_data(seq_len, path_ori,  path_after, path_year, item_y, path_pati, result_list, log_list, item_p) :

    path_pati2 = os.path.join(path_year,item_p)
    if not(os.path.isdir(path_pati2)) : os.mkdir(path_pati2)

    path_data = os.path.join(path_pati,item_p, item_p+'_standard')
    # try:
    lst_data = os.listdir(path_data)
    lst_data = sorted(lst_data)
    result_list.append(path_data)
    print("[{}] : Pre-processing {}'s data ...".format( str(len(result_list)), str(item_p)))
    
    dummy= cv2.imread('./dummy.png', cv2.IMREAD_GRAYSCALE)
    for idx_d, item_d in enumerate(lst_data):
        try:
            path_image3 = os.path.join(path_data, lst_data[idx_d])
            image3 = cv2.imread(path_image3,cv2.IMREAD_GRAYSCALE)
        except IndexError as identifier:
            image3 = dummy

        try:
            path_image4 = os.path.join(path_data, lst_data[idx_d+1])
            image4 = cv2.imread(path_image4,cv2.IMREAD_GRAYSCALE)
        except IndexError as identifier:
            image4 = dummy

        try:
            path_image5 = os.path.join(path_data, lst_data[idx_d+2])
            image5 = cv2.imread(path_image5,cv2.IMREAD_GRAYSCALE)
        except IndexError as identifier:
            image5 = dummy
        try :
            if seq_len == 3 : image = cv2.hconcat([image3, image4, image5])
            elif seq_len ==2 : image = cv2.hconcat([image3, image4])
            elif seq_len == 1 : image = image3
            else : image = None
            ####################
            image = cv2.resize(image, dsize=(int(image.shape[1]/4),int(image.shape[0]/4)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(path_pati2+'/'+item_d, image)
            ####################
        except :
            log_list.append(path_pati2)
            # print(identifier, path_data)

import argparse
parser = argparse.ArgumentParser(description='python Implementation')
parser.add_argument('--label', type = str, default ='')
parser.add_argument('--seq', type = int, default =3)
parser.add_argument('--second', type = int, default = 5)

if __name__ == "__main__":
    args = parser.parse_args()
    mode = args.label
    seq_len = args.seq
    flag_preprocessing = True
    flag_fold = False
    in_second = args.second

    path_after = './Dataset_Normalized_concat' +str(seq_len)
    path_ori = './SNU'

    if flag_preprocessing : 
        processor = cpu_count()
        print(processor,processor/10*9)
        pool = Pool(processes= int(processor/10*9))
        m = Manager()
        result_list = m.list()
        log_list = m.list()
        # f = open('preprocessing_result.csv', 'w', encoding= 'utf-8-sig', newline = '')
        # wr = csv.writer(f)
        if not(os.path.isdir(path_after)): os.mkdir(path_after)

        lst_year = os.listdir(path_ori)
        print(lst_year)
        for idx_y, item_y in enumerate(lst_year):
            if item_y != '2017' : continue
            if os.path.isfile(path_ori+'/'+item_y): 
                print('copy {}..'.format(item_y))
                os.system('cp ./{}/{} {}/{}'.format(path_ori, item_y, path_after, item_y))
                continue
            
            path_year = os.path.join(path_after, item_y)
            if not(os.path.isdir(path_year)) : os.mkdir(path_year)


            path_pati = path_ori+'/'+item_y
            lst_pati = os.listdir(path_pati)
            lst_pati =['A2017-EM-01-0165']
            print(lst_pati)
            func = partial(preprocessing_data, seq_len, path_ori, path_after, path_year, item_y, path_pati, result_list, log_list)
            pool.map(func, lst_pati)
    
        # for idx, item in enumerate(log_list):
        #     wr.writerow([item])
    print(' ----------------Pre-processing is Done ----------------- ')
    print(' ----------------Make CSV file ----------------- ')
    if flag_fold :
        for fold in range(1, 6): #fold 수정
            lst_error = []
            path = path_after
            phase = ['train', 'val', 'test']
            json_path = './Json/Annotation-ver2.json'
            json_file = open(json_path, 'r', encoding='utf-8-sig')
            meta_data = json.load(json_file)

            #from Pre-processed folder
            for idx, item in enumerate(phase):
                f = open('./SNU/'+item+'set-'+str(fold)+'.csv', 'r', encoding='utf-8-sig')
                rdr = csv.reader(f)

                #set csv
                if mode == '' : f1 = open(path+'/'+item+'set_image-'+str(fold)+'.csv', 'w', encoding='utf-8-sig', newline='')
                elif mode == 'apnea' : f1 = open(path+'/'+item+'set_image_'+mode+'-'+str(fold)+'.csv', 'w', encoding='utf-8-sig', newline='')
                wr = csv.writer(f1)
                
                lst_patient = []
                for i in range(len(meta_data['Patient'])):
                    lst_patient.append(meta_data['Patient'][i]['Patient_Number'])
                
                for idx1, item1 in enumerate(rdr):
                    dic_index = lst_patient.index(item1[1])
                    #item1 : ['2015', 'A2015-EM-01-0185', '0.0']
                    #extract event from json
                    start_time = meta_data ['Patient'][dic_index]['Start_time']
                    start_time = datetime.datetime.strptime(start_time, '%Y/%m/%d %I:%M:%S %p')
                    num_event = meta_data['Patient'][dic_index]['Event']
                    analysis_start = int(meta_data['Patient'][dic_index]['Analysis_Start']['Start_Epoch'])
                    lst_event = []
                    for idx2, item2 in enumerate(num_event):
                        event = meta_data['Patient'][dic_index]['Event'][idx2]['Event_Label']

                        #only Apnea, and Hypopnea by flag
                        if mode == '' : check = (event == 'Apnea Central' or event == 'Apnea Mixed' or event == 'Apnea Obstructive' or event == 'Hypopnea')
                        elif mode == 'apnea' : check = (event == 'Apnea Central' or event == 'Apnea Mixed' or event == 'Apnea Obstructive')
                        if check :
                            st = meta_data['Patient'][dic_index]['Event'][idx2]['Start_Time']
                            et = meta_data['Patient'][dic_index]['Event'][idx2]['End_Time']
                            se = int(meta_data['Patient'][dic_index]['Event'][idx2]['Start_Epoch'])
                            ee = int(meta_data['Patient'][dic_index]['Event'][idx2]['End_Epoch'])
                            lst_event.append([event,st,et,se,ee])

                    #check image
                    path1 = os.path.join(path,item1[0],item1[1])
                    try:
                        lst = os.listdir(path1)
                    except FileNotFoundError as identifier:
                        print(path1+"!")
                        continue
                    lst = sorted(lst)

                    #Labeling by Event
                    label = [0]*len(lst)

                    #각 에폭에 이벤트가 n% 이상 포함이면 1로..
                    for idx2, item2 in enumerate(lst_event):
                        try :
                            st = datetime.datetime.strptime(item2[1],'%Y/%m/%d %I:%M:%S %p')
                            et = datetime.datetime.strptime(item2[2], '%Y/%m/%d %I:%M:%S %p')
                        except :
                            print(item1 , '!!!!!')
                        # try : st = datetime.datetime.strptime(item2[1],'%Y-%m-%d %I:%M:%S %p')
                        # except : st = datetime.datetime.strptime(item2[1],'%Y/%m/%d %I:%M:%S %p')
                        # try : et = datetime.datetime.strptime(item2[2], '%Y-%m-%d %I:%M:%S %p')
                        # except : et = datetime.datetime.strptime(item2[2], '%Y/%m/%d %I:%M:%S %p')

                            ### 위에 수정해야함
                            
                        se, ee = item2[3], item2[4]

                        # print(st, '----->', et)
                        tmp = []
                        for i in range(se, ee+1):
                            try :
                                time_in_epoch = 0
                                if i == se :
                                    std = start_time + datetime.timedelta(seconds=(i)*30)
                                    time_in_epoch = (std-st).seconds
                                elif i == ee :
                                    std = start_time + datetime.timedelta(seconds=(i-1)*30)
                                    time_in_epoch = (et-std).seconds
                                else :
                                    time_in_epoch = 30
                                
                                if time_in_epoch > in_second :
                                    label[i-1] = 1
                                    
                                tmp.append([i,time_in_epoch, time_in_epoch > 10, i-1])
                            except : 
                                pass
                        
                    for l1, l2 in zip(lst[analysis_start-1:], label[analysis_start-1:]):
                        path_image = os.path.join(path1,l1)
                        target = l2
                        wr.writerow([path_image, target])     
                    # except :
                    #     lst_error.append(dic_index)
                    #     continue
            print('{} fold Dataset prepared'.format(fold))
            # print(lst_error)