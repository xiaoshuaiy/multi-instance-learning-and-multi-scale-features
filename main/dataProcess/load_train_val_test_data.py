import os.path
import numpy as np
import scipy.io as sio

def train_val(fold, task):
    if task == 1:
        task_name = 'AD_classification'
    if task == 2:
        task_name = 'MCI_conversion'
    # AD和HC各五分之一为测试集，4/5为train，val
    subject = sio.loadmat(
        'data/split/{}/subejct.mat'.format(task_name),
    )
    trian_val_subject = subject['train_subject']

    train_subject = []
    val_subject = []
    for subject_name in trian_val_subject[0][0]:
        # 从AD和HC中各选出1/5,用来验证
        valid_list = range(len(subject_name) // 5 * fold, len(subject_name) // 5 * (fold + 1))
        train_list = list(set(range(len(subject_name))).difference(set(valid_list)))

        train_subject.extend(list(subject_name[train_list]))
        val_subject.extend(list(subject_name[valid_list]))
    return train_subject, val_subject

# 加载训练和验证数据
def save_load(MRI_data_path, task=1, fold=0):
    if task == 1:
        task_name = 'AD_classification'
        class_name = ['NC', 'AD']
    if task == 2:
        task_name = 'MCI_conversion'
        class_name = ['pmci', 'smci']
    # 划分训练和验证的患者
    train_subject, val_subject = train_val(fold, task)
    train_sample = []
    train_labels = []
    val_sample = []
    val_labels = []
    # 加载对应患者的数据到训练验证数据集中
    for name in class_name:
        data_path = os.path.join(MRI_data_path, name)
        if name == 'AD' or name == 'PMci':
            lable = 1
        else:
            lable = 0
        for data in os.listdir(data_path):
            # 查询该患者id-data[3:13]是否在训练集或验证集中
            if data[3:13] in train_subject:
                sample = os.path.join(name, data)
                train_sample.append(sample)
                train_labels.append(lable)
            if data[3:13] in val_subject:
                sample = os.path.join(name, data)
                val_sample.append(sample)
                val_labels.append(lable)
    train_sample, train_labels, val_sample, val_labels = np.array(train_sample), np.array(train_labels), \
                                                         np.array(val_sample), np.array(val_labels)

    return train_sample, train_labels, val_sample, val_labels

# 加载测试数据
def test_load_subject(MRI_data_path, task=1):
    if task == 1:
        task_name = 'AD_classification'
        class_name = ['NC', 'AD']
    if task == 2:
        task_name = 'MCI_conversion'
        class_name = ['PMci', 'SMci']
    subject = sio.loadmat(
        'data/split/{}/subejct.mat'.format(task_name),
    )
    test_subject = subject['test_subject']
    test_sample = []
    test_labels = []
    for subject_name in test_subject[0][0]:
        for name in class_name:
            data_path = os.path.join(MRI_data_path, name)
            if name == 'AD'or name == 'PMci':
                lable = 1
            else:
                lable = 0
            for data in os.listdir(data_path):
                if data[3:13] in subject_name:
                    sample = os.path.join(name, data)
                    test_sample.append(sample)
                    test_labels.append(lable)
    test_sample, test_labels = np.array(test_sample), np.array(test_labels),
    return test_sample, test_labels

