import csv

import torch
import numpy as np
import os
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from main.dataProcess.dataloader import data_flow, tst_data_flow
from main.model.Model import Model
from main.utils.data_loading import load_data
from main.utils.result_saving import save_results
from main.utils.training import train_model
from main.utils.validation import validate_model

task = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 400
max_epochs = 200
learning_rate = 0.00001
batch_size = 1

Big_patch_size = 30
Big_patch_num = 80
Small_patch_size = 15
#
result = np.zeros(shape=(5, 4))
patience = 20
img_path = " "
if task == 1:
    img_path = "/data/mri_data/ad_hc"
    task_name = 'AD_classification'
elif task == 2:
    img_path = "data/mri_data/mci"
    task_name = 'MCI_conversion'

example_name = 'train_plan0'


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

for fold_index in range(5):
    seed_torch(7)
    # Load data
    train_sample, train_labels, val_sample, val_labels, Big_template_cors = load_data(img_path, fold_index, Big_patch_size, Big_patch_num)

    # Initialize model and optimizer
    model = Model(data_dir=img_path, topk=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.8, 0.99), weight_decay=0.0005)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Train model
    train_loader = data_flow(img_path, train_sample, train_labels, Big_template_cors, batch_size, Big_patch_size, Big_patch_num)
    train_loss_list, train_acc_list = train_model(model, train_loader, optimizer, device, nn.CrossEntropyLoss(), epochs, batch_size, scheduler, Big_template_cors, patience,fold_index)

    # Validate model
    val_loader = tst_data_flow(img_path, val_sample, val_labels, Big_template_cors, batch_size, Big_patch_size, Big_patch_num)
    val_loss, val_acc, sen, spe, roc_auc = validate_model(model, val_loader, device, nn.CrossEntropyLoss(), val_labels, np.sum(val_labels == 1), np.sum(val_labels == 0))

    # Save results
    result[fold_index] = [val_acc, sen, spe, roc_auc]
    save_results(example_name, result, train_loss_list, train_acc_list, [val_loss], [val_acc], fold_index)
# 保存五折的结果到 CSV 文件
csv_file_path = f'/train_ADNI_fold5_size{Big_patch_size}_num{Big_patch_num}.csv'

with open(csv_file_path, 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['fold', 'acc', 'sen', 'spe', 'roc_auc'])
    for i in range(result.shape[0]):
        writer.writerow([f'fold{i + 1}', result[i][0], result[i][1], result[i][2], result[i][3]])

# 打印总结果
print("Final Results for 5 Folds:")
print(result)
