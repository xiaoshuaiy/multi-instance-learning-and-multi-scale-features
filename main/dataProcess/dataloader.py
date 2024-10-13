import SimpleITK as sitk
import numpy as np
import torch
import os
from scipy.stats import entropy

def entr(data):
    # 将 img_patch 展平为一维数组
    flat_img_patch = data.flatten()
    # 计算熵信息
    entropy_value = entropy(flat_img_patch)
    return entropy_value

def IsAllZero(data):
    is_all_zero = np.all(data == 0)  # 判断数据是否全为0
    return is_all_zero


def data_flow(img_path, sample_name, sample_labels, center_cors,
              batch_size, patch_size, num_patches, shuffle_flag=True):
    margin = int(np.floor((patch_size - 1) / 2.0))
    input_shape = (1, patch_size, patch_size, patch_size)
    output_shape = (batch_size, 1)

    # indices = np.random.permutation(center_cors.shape[0])
    # # 打乱中心坐标
    # center_cors = center_cors[indices]

    while True:
        if shuffle_flag:
            sample_name = np.array(sample_name)
            sample_labels = np.array(sample_labels)
            permut = np.random.permutation(len(sample_name))
            np.take(sample_name, permut, out=sample_name)
            np.take(sample_labels, permut, out=sample_labels)
            sample_name = sample_name.tolist()
            sample_labels = sample_labels.tolist()
        img_path = img_path
        i_batch = 0

        inputs = []
        for i_input in range(num_patches):
            inputs.append(np.zeros(input_shape, dtype='float32'))
        outputs = np.ones(output_shape, dtype=np.int64)
        for i_iter in range(len(sample_name)):
            centers = []
            img_dir = os.path.join(img_path, sample_name[i_iter].strip())
            I = sitk.ReadImage(img_dir)
            img = np.array(sitk.GetArrayFromImage(I))
            img = np.pad(img, pad_width=((2, 2), (2, 3), (2, 2)), mode='constant', constant_values=(0, 0))
            for idx,center in enumerate(center_cors):
                x_cor, y_cor, z_cor = center
                img_patch = img[x_cor - margin: x_cor + margin + 1,
                            y_cor - margin: y_cor + margin + 1,
                            z_cor - margin: z_cor + margin + 1]
                # print(np.array(img_patch).shape)
                # if np.array(img_patch).shape ==(15,15,15) and IsAllZero(np.array(img_patch)) == False \
                #                                 and entr(np.array(img_patch))>1.0:
                #     if random.random()>0.5:
                #          img_patch = gen_data(np.array(img_patch))
                #     inputs[idx][0,:,:,:] = img_patch
                #     centers.append(center)
                if np.array(img_patch).shape ==(30, 30, 30)and IsAllZero(np.array(img_patch))== False:
                    if entr(np.array(img_patch))>2.0:
                         # img_patch = gen_data(np.array(img_patch))
                         img_patch = np.array(img_patch)
                         inputs[idx][0,:,:,:] = img_patch
                         centers.append(center)
                else:
                    continue
            outputs[i_batch, :] = sample_labels[i_iter] * outputs[i_batch, :]
            i_batch += 1
            if i_batch == batch_size:
                inputs = np.expand_dims(np.array(inputs[:len(centers)]),axis=0)
                yield (torch.from_numpy(inputs), torch.from_numpy(np.array(outputs)),centers,sample_name[i_iter].strip())
                inputs = []
                for i_input in range(num_patches):
                    inputs.append(np.zeros(input_shape, dtype='float32'))
                outputs = np.ones(output_shape, dtype=np.int64)
                i_batch = 0


def tst_data_flow(img_path, sample_name, sample_labels, center_cors,
              batch_size, patch_size, num_patches, shuffle_flag=False):
    margin = int(np.floor((patch_size - 1) / 2.0))
    input_shape = (1, patch_size, patch_size, patch_size)
    output_shape = (batch_size, 1)

    while True:
        if shuffle_flag:
            sample_name = np.array(sample_name)
            sample_labels = np.array(sample_labels)
            permut = np.random.permutation(len(sample_name))
            np.take(sample_name, permut, out=sample_name)
            np.take(sample_labels, permut, out=sample_labels)
            sample_name = sample_name.tolist()
            sample_labels = sample_labels.tolist()
        img_path = img_path
        i_batch = 0
        inputs = []
        for i_input in range(num_patches):
            inputs.append(np.zeros(input_shape, dtype='float32'))
        outputs = np.ones(output_shape, dtype=np.int64)
        for i_iter in range(len(sample_name)):
            # inputs =[]
            centers = []
            img_dir = os.path.join(img_path, sample_name[i_iter].strip())
            # print(img_dir)
            I = sitk.ReadImage(img_dir)
            img = np.array(sitk.GetArrayFromImage(I))
            img = np.pad(img, pad_width=((2, 2), (2, 3), (2, 2)), mode='constant', constant_values=(0, 0))

            for idx,center in enumerate(center_cors):
                x_cor, y_cor, z_cor = center
                img_patch = img[x_cor - margin: x_cor + margin + 1,
                            y_cor - margin: y_cor + margin + 1,
                            z_cor - margin: z_cor + margin + 1]
                # if np.array(img_patch).shape == (15, 15, 15) and IsAllZero(np.array(img_patch)) == False\
                #         and entr(np.array(img_patch)) > 1.0:
                #     inputs[idx][0, :, :, :] = img_patch
                #     centers.append(center)
                if np.array(img_patch).shape == (30, 30, 30) and IsAllZero(np.array(img_patch))== False :
                    if entr(np.array(img_patch)) >2.0:
                        img_patch = np.array(img_patch)
                        inputs[idx][0, :, :, :] = img_patch
                        centers.append(center)
                else:
                    continue
            outputs[i_batch, :] = sample_labels[i_iter] * outputs[i_batch, :]
            i_batch += 1
            if i_batch == batch_size:
                inputs = np.expand_dims(np.array(inputs[:len(centers)]),axis=0)
                yield (
                torch.from_numpy(inputs), torch.from_numpy(np.array(outputs)), centers, sample_name[i_iter].strip())
                inputs = []
                for i_input in range(num_patches):
                    inputs.append(np.zeros(input_shape, dtype='float32'))
                outputs = np.ones(output_shape, dtype=np.int64)
                i_batch = 0


