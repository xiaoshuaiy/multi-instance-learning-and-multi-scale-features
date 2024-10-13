import torch.nn as nn
import numpy as np
import torch
import nibabel as nib
import os

from main.model.FristMIL import FristMIL
from main.model.SencondMultiScale import MultiScale


class Model(nn.Module):
    def __init__(self,data_dir,topk = None):
        super(Model, self).__init__()

        self.Frist_Stage = FristMIL()
        self.Second_Stage =  MultiScale()
        self.topk = topk
        self.data_dir = data_dir
        self.Small_patch_size =15
        self.Big_patch_size = 30
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.Frist_classifier = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.Second_classifier = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def get_topk_ids(self, A,topk, h):
        if len(A.shape) == 1:
            A = A.view(1, -1)
        # 获取注意力分数最大的前k个特征值
        top_p_ids = torch.topk(A, topk)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        # # 获取注意力分数最小的前k个特征值
        # top_n_ids = torch.topk(-A, topk, dim=1)[1][-1]
        # top_n = torch.index_select(h, dim=0, index=top_n_ids)

        topk_inst_feature = torch.cat([top_p], dim=0)
        top_k_ids = torch.cat([top_p_ids], dim=0)
        return top_k_ids, topk_inst_feature

    def get_topk_centers(self,centers, ids):
        topk_centers = []
        for idx in ids:
            idx = idx.item()
            topk_centers.append(centers[idx])
        return topk_centers

    def get_small_centers(self,Big_centers):
        Small_block_centers = []
        Big_half_dis = self.Big_patch_size // 2
        for centers in Big_centers:
            for l in range(centers[0] - Big_half_dis, centers[0] + Big_half_dis, self.Small_patch_size):
                for w in range(centers[1] - Big_half_dis, centers[1] + Big_half_dis, self.Small_patch_size):
                    for h in range(centers[2] - Big_half_dis, centers[2] + Big_half_dis, self.Small_patch_size):
                        Small_half_dis = (self.Small_patch_size - 1) // 2
                        # print(center)
                        center_l = l + Small_half_dis  # 区域在 l 维度的中心点索引
                        center_w = w + Small_half_dis  # 区域在 w 维度的中心点索引
                        center_h = h + Small_half_dis  # 区域在 h 维度的中心点索引
                        center_point = (center_l, center_w, center_h)
                        Small_block_centers.append(center_point)
        return list(Small_block_centers)

    def loader(self, sample_name, new_centers,device):
        margin = int(np.floor((self.Small_patch_size ) / 2.0))
        input_shape = (1, self.Small_patch_size , self.Small_patch_size , self.Small_patch_size)

        inputs = []
        for i_input in range(len(new_centers)):
            inputs.append(np.zeros(input_shape, dtype='float32'))

        img_dir = os.path.join(self.data_dir, sample_name.strip())
        nii = nib.load(img_dir)
        # get_fdata()：获取NIfTI格式的数据数组
        img = nii.get_fdata()
        img = np.pad(img, pad_width=((2, 2), (2, 3), (2, 2)), mode='constant', constant_values=(0, 0))
        img = torch.from_numpy(img).float()
        img = img.to(device)
        inputs = torch.from_numpy(np.array(inputs)).float().to(device)
        for idx, center in enumerate(new_centers):
            x_cor, y_cor, z_cor = center
            img_patch = img[x_cor - margin: x_cor + margin + 1,
                        y_cor - margin: y_cor + margin + 1,
                        z_cor - margin: z_cor + margin + 1]
            # img_patch = np.array(img_patch)
            inputs[idx][0, :, :, :] = img_patch
        inputs = inputs.unsqueeze(dim=0)
        return inputs

    def forward(self,x,y,centers,sample_name):
        M, A_sorce,res_dict1,total_inst_loss,Ffeatrues,centers= self.Frist_Stage(x,y,centers,instance_eval=True)
        # 获取A_sorce top_K 的index
        FristStage_prob = self.Frist_classifier(M)
        FristStage_hat = torch.ge(FristStage_prob, 0.5).float()

        topk = min(self.topk, len(centers))
        ids,Ftopk_featrues= self.get_topk_ids(A_sorce,topk,Ffeatrues)
        # 获取topk中心点
        topk_centers = self.get_topk_centers(centers, ids)
        # 获取新的中心点
        new_centers = self.get_small_centers(topk_centers)
        # 加载新的数据
        Second_x = self.loader(sample_name,new_centers,x.device).to(x.device)

        Second_M ,res_dict2= self.Second_Stage(Second_x,Ftopk_featrues)

        SecondStage_prob = self.Second_classifier(Second_M)
        SecondStage_hat = torch.ge(SecondStage_prob, 0.5).float()

        featrues = torch.cat((M,Second_M),dim=1)
        Y_prob = self.classifier(featrues)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat,FristStage_prob,FristStage_hat,SecondStage_prob,SecondStage_hat,total_inst_loss

    def calculate_classification_acc(self, X, Y,centers,sample_name):
        _, Y_hat, _, FristStage_hat, _,SecondStage_hat,_= self.forward(X,Y, centers,sample_name)
        acc = Y_hat.eq(Y).cpu().float().mean().item()
        FristStage_acc = FristStage_hat.eq(Y).cpu().float().mean().item()
        SecondStage_acc = SecondStage_hat.eq(Y).cpu().float().mean().item()
        return acc, Y_hat, FristStage_acc,SecondStage_acc

    def calculate_objective(self, X, Y,centers,sample_name):
        Y_prob, _, FristStage_prob, _,SecondStage_prob,_,inst_loss = self.forward(X,Y,centers,sample_name)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        FristStage_prob = torch.clamp(FristStage_prob, min=1e-5, max=1. - 1e-5)
        SecondStage_prob = torch.clamp(SecondStage_prob, min=1e-5, max=1. - 1e-5)

        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob)) # negative log bernoulli
        FristStage_loss = -1. * (Y * torch.log(FristStage_prob ) + (1. - Y) * torch.log(1. - FristStage_prob ))
        SecondStage_loss = -1. * (Y * torch.log( SecondStage_prob) + (1. - Y) * torch.log(1. -  SecondStage_prob))

        loss = 0.1*inst_loss+ 0.9*(0.5*neg_log_likelihood+(0.5*0.5*(FristStage_loss + SecondStage_loss)))
        return loss,neg_log_likelihood, Y_prob,FristStage_loss,SecondStage_loss,inst_loss




