import os
import csv
import matplotlib.pyplot as plt


def save_results(example_name, result, train_loss_list, train_acc_list, val_loss_list, val_acc_list, fold_index):
    os.makedirs(f'path_to_results/{example_name}/results/', exist_ok=True)

    train_file = f'path_to_results/{example_name}/results/train_metrics_fold{fold_index}.txt'
    val_file = f'path_to_results/{example_name}/results/val_metrics_fold{fold_index}.txt'

    save_metrics(train_file, train_loss_list, train_acc_list)
    save_metrics(val_file, val_loss_list, val_acc_list)

    save_result_csv(result, example_name, fold_index)

    plot_and_save_graph(train_loss_list, val_loss_list, train_acc_list, val_acc_list, example_name, fold_index)


def save_metrics(file_path, loss_list, acc_list):
    with open(file_path, 'w') as file:
        file.write(f'Loss List: {loss_list}\n')
        file.write(f'Acc List: {acc_list}\n')


def save_result_csv(result, example_name, fold_index):
    csv_file_path = f'path_to_results/{example_name}/results/evaluation_fold{fold_index}.csv'
    with open(csv_file_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Fold', 'Acc', 'Sen', 'Spe', 'ROC_AUC'])
        for i, res in enumerate(result):
            writer.writerow([f'Fold {i}', res[0], res[1], res[2], res[3]])


def plot_and_save_graph(train_acc_list, val_acc_list, train_loss_list, val_loss_list, example_name, fold_index):
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_list, label='Training Accuracy')
    plt.plot(val_acc_list, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.title('Training and Validation Loss')

    img_dir = f'path_to_results/{example_name}/img/'
    os.makedirs(img_dir, exist_ok=True)
    plt_path = os.path.join(img_dir, f"fold_{fold_index}_acc_loss.png")
    plt.savefig(plt_path)
    plt.close()
