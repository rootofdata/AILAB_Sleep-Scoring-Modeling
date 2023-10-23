import os
import json
import random
import csv
path_image = './SNU'
path_json = './Json/Annotation.json'
def split_dataset(path1='./SNU', path2='./Json/Annotation-ver2.json', ratio = [0.00,0.00,1.0]):
    f_summary = open(path1+'/Dataset_summary.csv', 'w', encoding='utf-8-sig', newline='')
    wr_summary = csv.writer(f_summary)
    
    num_patient, lst_patient_AHI = json_loader(path2)

    lst_patient_AHI = sorted(lst_patient_AHI, key = lambda lst_patient_AHI : lst_patient_AHI[2])

    standard = [0, int(20*ratio[0]), int(20*(ratio[0]+ratio[1])), int(20*(ratio[0]+ratio[1]+ratio[2]))]
    standard2 = ratio

    log = 'Dataset Ratio : [Train : {}%, Val : {}%, Test : {}%]'.format(ratio[0]*100, ratio[1]*100, ratio[2]*100) 
    wr_summary.writerow([log])
    for fold in range(1, 11):
        trainset, valset, testset = [], [], []

        for i in range(0, int(num_patient/20)+1):
            if i == int(num_patient/20) :
                tmp = lst_patient_AHI[i*20:]
                random.shuffle(tmp)
                random.shuffle(tmp)
                v1 =int(len(tmp)*standard2[1])
                v2 =int(len(tmp)*(standard2[1] +standard2[2]))
                valset += tmp[0:v1]
                testset += tmp[v1:v2]
                trainset += tmp[v2:]

            else :
                tmp =lst_patient_AHI[i*20:(i+1)*20]
                random.shuffle(tmp)
                random.shuffle(tmp)
                # print(tmp[standard[0]:standard[1]])
                # print(tmp[standard[1]:standard[2]])
                # print(tmp[standard[2]:standard[3]]           )
                trainset += tmp[standard[0]:standard[1]]
                valset += tmp[standard[1]:standard[2]]
                testset += tmp[standard[2]:standard[3]]            
        dataset = [trainset, valset, testset]
        log = 'Total num of patient : {} -> [train : {}, val : {}, test : {}]'.format(num_patient, len(trainset), len(valset), len(testset))
        print(log)
        wr_summary.writerow([log])

        # check AHI avg for each set
        sum1 = 0
        for idx, item in enumerate(trainset):
            sum1+= item[2]
        sum2 = 0
        for idx, item in enumerate(valset):
            sum2+= item[2]
        sum3 = 0
        for idx, item in enumerate(testset):
            sum3+= item[2]
        
        # a1, a2, a3, a4 = round(sum1/len(trainset),2), round(sum2/len(valset),2), round(sum3/len(testset),2), round((sum1+sum2+sum3)/(len(trainset)+len(valset)+len(testset)),2)
        try :
            a1 = round(sum1/len(trainset),2)
        except : a1=0
        try :
            a2 = round(sum2/len(valset),2)
        except : a2=0
        try :
            a3 = round(sum3/len(testset),2)
        except :
            a3 = 0
        a4 = round((sum1+sum2+sum3)/(len(trainset)+len(valset)+len(testset)),2)


        log = 'AHI AVG : [Total : {}, Trian : {}, Val : {}, Test : {}]'.format(a4, a1, a2, a3)
        wr_summary.writerow([log])
        print(log)

        dic = {0:'trainset', 1:'valset', 2:'testset'}
        for idx, item in enumerate(dic):
            f = open('./SNU/'+dic[item]+'-'+str(fold)+'.csv', 'w', encoding='utf-8-sig')
            wr =csv.writer(f)
            for idx1, item1 in enumerate(dataset[idx]):
                wr.writerow(item1)
        wr_summary.writerow(['----------------------------------------------------------'])

####### 참고해야 할 이벤트 #############
#Central Apnea, Obstructive Apnea, Mixed Apena
#중추성 무호흡   폐쇄성 무호흡       혼합성 무호흡

#Hypopnea 저호흡 + Respiratory Arousal, Desat
######################################

def json_loader(json_path = './Json/Annotation2.json'):
    with open(json_path, 'r', encoding='utf-8-sig') as json_file:
        meta_data = json.load(json_file)
        num_of_patient=int(meta_data['Num_of_Patient'])
        lst_patient = meta_data['Patient']
        lst_patient_AHI = []
        lst_error = []
        for idx, item in enumerate(lst_patient) :
            # print(item['Patient'], (item['Report']['AHI']))
            # print(item['Report']['Patient Serial Number'], (item['Report']['AHI']))
            try :
                lst_patient_AHI.append([item['Report']['Patient Serial Number'][1:5],item['Report']['Patient Serial Number'],float(item['Report']['AHI'])])
            except :
                lst_patient_AHI.append([item['Report']['Patient Serial Number'][1:5],item['Report']['Patient Serial Number'],(0.0)])
                lst_error.append(item['Report']['Patient Serial Number'])
        print('-----------AHI error -----')
        print(lst_error)
        print('-----------AHI error -----')
    return num_of_patient, lst_patient_AHI            

if __name__ == "__main__":
    split_dataset()

