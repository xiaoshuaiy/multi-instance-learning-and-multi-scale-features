import scipy.io as sio

from main.dataProcess.load_train_val_test_data import save_load


def load_data(img_path, fold_index, Big_patch_size, Big_patch_num):
    train_sample, train_labels, val_sample, val_labels = save_load(img_path, 1, fold_index)
    # 加载体素块中心 - utils.get_centers
    Big_template_cors = sio.loadmat(
        'Big_block_template_center_size{}_num{}.mat'.format(Big_patch_size, Big_patch_num))
    Big_template_cors = Big_template_cors['patch_centers']

    return train_sample, train_labels, val_sample, val_labels, Big_template_cors
