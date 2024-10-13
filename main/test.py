import torch
import numpy as np
import csv
import scipy.io as sio
from sklearn.metrics import roc_curve, auc
from torch import nn
from torch.autograd import Variable

from main.dataProcess.dataloader import tst_data_flow
from main.dataProcess.load_train_val_test_data import test_load_subject

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# 定义参数
Big_patch_size = 30
Big_patch_num = 80
batch_size = 1

# 数据路径设置
task = 1
if task == 1:
    img_path = "data/mri_data/ad_hc"
    task_name = 'AD_classification'
elif task == 2:
    img_path = "data/mri_data/mci"
    task_name = 'MCI_conversion'

# 加载测试数据
samples_test, labels_test = test_load_subject(img_path, 1)
result = np.zeros(shape=(5, 4))
example_name = 'train_plan0'

# 进行 5 折验证
for i in range(5):
    criterion = nn.CrossEntropyLoss()

    # 加载大块模板中心坐标
    template_cors = sio.loadmat(
        f'Big_block_template_center_size{Big_patch_size}_num{Big_patch_num}.mat')
    template_cors = template_cors['patch_centers']

    # 加载预训练模型
    model = torch.load(
        f'path_to_save_model/best_acc_model.pt')
    model.to(device)

    # 初始化统计变量
    TP, TN = 0, 0
    subject_prob = []
    test_acc, Fristacc, Secondacc = 0., 0., 0.
    pos_num = np.sum(labels_test == 1)
    neg_num = np.sum(labels_test == 0)

    # 准备测试数据
    test_loader = tst_data_flow(img_path, samples_test, labels_test, template_cors, batch_size, Big_patch_size,
                                Big_patch_num)

    # 模型评估
    for i_batch in range(len(samples_test) // batch_size):
        test_data, test_label, test_centers, test_sample_path = next(test_loader)
        test_data, test_label = test_data.to(device), test_label.to(device)
        test_data, test_label = Variable(test_data), Variable(test_label)

        # 计算损失和精度
        loss, _, Y_prob, FristStage_loss, SecondStage_loss, _ = model.calculate_objective(test_data, test_label,
                                                                                          test_centers,
                                                                                          test_sample_path)
        acc, Y_hat, FristStage_acc, SecondStage_acc = model.calculate_classification_acc(test_data, test_label,
                                                                                         test_centers, test_sample_path)

        test_acc += acc
        Fristacc += FristStage_acc
        Secondacc += SecondStage_acc
        subject_prob.append(Y_prob.item())

        if test_label.item() == 1 and Y_hat.item() == 1:
            TP += 1
        if test_label.item() == 0 and Y_hat.item() == 0:
            TN += 1

    # 计算指标
    sen = TP / pos_num
    spe = TN / neg_num
    fpr, tpr, thresholds = roc_curve(labels_test, subject_prob)
    roc_auc = auc(fpr, tpr)

    # 保存结果
    result[i] = [test_acc / len(samples_test), sen, spe, roc_auc]

    # 打印当前折的结果
    print(
        f'Fold {i + 1}: SEN={sen:.4f}, SPE={spe:.4f}, AUC={roc_auc:.4f}, Test Accuracy={test_acc / len(samples_test):.4f}')

# 保存五折的结果到 CSV 文件
csv_file_path = f'/test_ADNI_fold5_size{Big_patch_size}_num{Big_patch_num}.csv'

with open(csv_file_path, 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['fold', 'acc', 'sen', 'spe', 'roc_auc'])
    for i in range(result.shape[0]):
        writer.writerow([f'fold{i + 1}', result[i][0], result[i][1], result[i][2], result[i][3]])

# 打印总结果
print("Final Results for 5 Folds:")
print(result)
