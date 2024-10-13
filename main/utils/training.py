import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_model(model, train_loader, optimizer, device, criterion, epochs, batch_size, scheduler, Big_template_cors,
                patience,fold):
    best_loss = float(1000)
    best_acc = 0
    counter = 0
    train_loss_list, train_acc_list = [], []

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = 0., 0.

        for i_batch, (data, bag_label, centers, sample_path) in enumerate(train_loader):
            data, bag_label = data.to(device), bag_label.to(device)
            optimizer.zero_grad()
            loss, acc = calculate_loss_and_acc(model, data, bag_label, centers, sample_path)

            train_loss += loss.item()
            train_acc += acc
            loss.backward()
            optimizer.step()

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        if train_acc > best_acc:
            best_acc = train_acc
            save_model(model, epoch, '{}_best_acc_model.pt'.format(fold))

        if train_loss < best_loss:
            best_loss = train_loss
            save_model(model, epoch, '{}_best_loss_model.pt'.format(fold))
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Validation loss did not improve for {patience} epochs, stopping early.")
            break
    return train_loss_list, train_acc_list


def calculate_loss_and_acc(model, data, labels, centers, sample_path):
    loss, _, _, _, _, _ = model.calculate_objective(data, labels, centers, sample_path)
    acc, _, _, _ = model.calculate_classification_acc(data, labels, centers, sample_path)
    return loss, acc


def save_model(model, epoch, file_name):
    torch.save(model, f'path_to_save_model/{file_name}')
