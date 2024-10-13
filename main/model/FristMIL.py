import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from main.model.MVNet import MVNet

class TransLayer(nn.Module):
    def __init__(self,  dim=512):
        super().__init__()
        # Multi-Head Self-Attention Layer
        self.self_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=8)
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(1,0,2)
        x = self.layer_norm(x)
        attn_output,  weight = self.self_attention(x, x, x)
        x = x + attn_output
        return x, weight
#
class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        # # (b, 640, 512)
        self.proj = nn.Conv3d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv3d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv3d(dim, dim,3, 1, 3 // 2, groups=dim)


    def forward(self, x, H, W,L):
        B, _, C = x.shape
        feat_token = x #hf
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W,L)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

class TransAttention(nn.Module):
    def __init__(self):
        super(TransAttention, self).__init__()
        self.pos_layer = PPEG(dim=512)
        # self._fc1 = nn.Sequential(nn.Linear(1280, 512),nn.BatchNorm1d(512), nn.ReLU(inplace=True))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        # X shape (N,512)
        h = x.unsqueeze(dim=0)
        # ---->pad
        H = h.shape[1]
        _H, _W, _L = int(np.ceil(np.cbrt(H))), int(np.ceil(np.cbrt(H))), int(np.ceil(np.cbrt(H)))
        add_length = _H * _W * _L - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]
        # ---->PPEG
        h = self.pos_layer(h, _H, _W,_L)  # [B, N, 512]
        # ---->Translayer x1
        h,weight = self.layer1(h)  # [B, N, 512]
        h = h.permute(1, 0, 2).squeeze(dim=0)
        h = self.norm(h)
        return h,weight,add_length


class FristMIL(nn.Module):
    def __init__(self):
        super(FristMIL, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1

        self.n_classes = 2
        instance_classifiers = [nn.Linear(512, 2)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = 5
        self.neg_k_sample = 5
        self.instance_loss_fn = nn.CrossEntropyLoss()
        self.subtyping = False
        self.bag_weight = 0.8

        self.MVnet = MVNet()
        self.softmax = nn.Softmax()

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(4, 4, 4), stride=1, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(2, stride=2),
            nn.Dropout(0.2),

            nn.Conv3d(64, 128, kernel_size=(4, 4, 4), stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(2, stride=2),
            nn.Dropout(0.2),

            nn.Conv3d(128, 256, kernel_size=(4, 4, 4), stride=1, padding=0),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
                nn.Conv3d(256,256, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(256 * 2 * 2 * 2, self.L),
            nn.BatchNorm1d(self.L),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.attention = TransAttention()

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )
        self._fc1 = nn.Sequential(nn.Linear(1280, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True))

    def relocate(self):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        # 获取注意力分数最大的前k个特征值
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        all_targets = torch.cat([p_targets], dim=0)
        all_instances = torch.cat([top_p], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def inst_eval_neg(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        # 获取注意力分数最大的前neg_k个特征值
        # length = A.shape[1]
        # top_p_ids = torch.topk(A, length)[1][-1][length//4:length//4+ self.neg_k_sample]
        top_p_ids = torch.topk(A, self.neg_k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        n_targets = self.create_negative_targets(self.neg_k_sample, device)
        all_targets = torch.cat([n_targets], dim=0)
        all_instances = torch.cat([top_p], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def forward(self,x,label=None,centers=None, instance_eval=False, return_features=False, attention_only=False,no_twostage=False):
        x = x.squeeze(dim=0)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 256 * 2 * 2 * 2)
        H= self.feature_extractor_part2(H)
        H2D = self.MVnet(x)
        features = torch.cat((H, H2D), dim=1)
        features = self._fc1(features)
        cls_H,SA_weight,add_length= self.attention(features,centers)  # NxK
        features = torch.cat((features,features[:add_length,:]),dim=0)
        # centers.extend(centers[:add_length])
        A_V = self.attention_V(cls_H)  # NxD
        A_U = self.attention_U(cls_H)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, cls_H)  # KxL
        if instance_eval:
            total_inst_loss = 0.0
            insts_acc = 0
            for i in range(len(self.instance_classifiers)):
                inst_label = label.item()
                classifier = self.instance_classifiers[i]
                #类内，针对真实标签计算loss
                if inst_label == 1: #正样本:
                    # print('inst lable',inst_label)
                    instance_loss, preds, targets= self.inst_eval(A, features, classifier)
                    inst_acc = (preds == targets).sum().item()
                else: #负样本
                    # print('inst lable', inst_label)
                    instance_loss, preds, targets= self.inst_eval_neg(A,features, classifier)
                    inst_acc = (preds == targets).sum().item()
                insts_acc += inst_acc
                total_inst_loss += instance_loss
            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        if instance_eval:
            results_dict = {'instance_loss':total_inst_loss, 'inst_acc':insts_acc,
                            'all_inst_num':self.k_sample,}
        else:
            results_dict = {'instance_loss': 0, 'inst_acc': 0,'all_inst_num':0}
        if no_twostage == True:
            Y_prob = self.classifier(M)
            Y_hat = torch.ge(Y_prob, 0.5).float()
            results_dict.update({'Y_prob':Y_prob,'Y_hat':Y_hat})
        if return_features:
            results_dict.update({'features': features,'att_contribute':A})
        length = features.shape[0]
        return  M,A[:,:length-add_length], results_dict,total_inst_loss,cls_H[:length-add_length],centers