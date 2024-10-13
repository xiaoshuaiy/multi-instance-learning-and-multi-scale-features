# load all subject
# 根据患者初步划分训练集和测试集的人数，后续根据具体数据划分训练集和测试集
import scipy.io as sio
import os.path
import numpy as np
from random import sample

data_path = ''
task = 1

if task == 1:
    data_path = 'data/ADNI/AD_classification'
    task_name = 'AD_classification'
    class_name = ['HC', 'AD']
if task == 2:
    data_path = 'data/ADNI/mci'
    task_name = 'MCI_conversion'
    class_name = ['pmci', 'smci']

train_all_subject = {}
test_all_subject = {}
for name in class_name:
    class_data_path = os.path.join(data_path, name)

    subject_list = []
    for subject in os.listdir(class_data_path):
        # 截取患者id
        subject = subject[3:13]
        if subject not in subject_list:
            subject_list.append(subject)
        else:
            continue

    subject_list = np.array(subject_list)
    # 随机打乱顺序
    permut = np.random.permutation(len(subject_list))
    np.take(subject_list, permut, out=subject_list)
    # 随机抽取五分之一的subject做为测试集
    test_subject = sample(list(subject_list), round(len(subject_list) / 5))
    train_subject = list(set(subject_list).difference(set(test_subject)))
    train_all_subject[name] = train_subject
    test_all_subject[name] = test_subject

os.makedirs('/data/split/{}/'.format(task_name), exist_ok=True)
sio.savemat('/data/split/{}/subejct.mat'.format(task_name),
            {"train_subject": train_all_subject,
             "test_subject": test_all_subject, }
            )
