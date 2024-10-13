import torch
from sklearn.metrics import roc_curve, auc


def validate_model(model, val_loader, device, criterion, val_labels, pos_num, neg_num):
    val_loss, val_acc = 0., 0.
    TP, TN = 0, 0
    subject_prob = []

    model.eval()
    with torch.no_grad():
        for data, val_label, val_centers, val_sample_path in val_loader:
            data, val_label = data.to(device), val_label.to(device)
            loss, Y_prob, acc, Y_hat = calculate_val_metrics(model, data, val_label, val_centers, val_sample_path)

            val_loss += loss.item()
            val_acc += acc
            subject_prob.append(Y_prob.item())
            if val_label.item() == 1 and Y_hat.item() == 1:
                TP += 1
            if val_label.item() == 0 and Y_hat.item() == 0:
                TN += 1

    sen = TP / pos_num
    spe = TN / neg_num
    fpr, tpr, thresholds = roc_curve(val_labels, subject_prob)
    roc_auc = auc(fpr, tpr)

    return val_loss, val_acc, sen, spe, roc_auc


def calculate_val_metrics(model, data, val_labels, val_centers, val_sample_path):
    loss, _, Y_prob, _, _, _ = model.calculate_objective(data, val_labels, val_centers, val_sample_path)
    acc, Y_hat, _, _ = model.calculate_classification_acc(data, val_labels, val_centers, val_sample_path)
    return loss, Y_prob, acc, Y_hat
