import torch.nn as nn
import torch.nn.functional as F
import torch
class MultiScale(nn.Module):
    def __init__(self):
        super(MultiScale, self).__init__()

        self.L = 512
        self.D = 128
        self.K = 1

        self.n_classes = 2

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(4, 4, 4), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(2, stride=2),
            nn.Dropout(0.2),

            nn.Conv3d(64, 128, kernel_size=(4, 4, 4), stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(2, stride=2),
            nn.Dropout(0.2),

            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(128 * 1 * 1 * 1, self.L),
            nn.BatchNorm1d(self.L),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

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
        self._fc =  nn.Sequential(
            nn.Linear(1024, 512),
            nn.Sigmoid()
        )
    # Ftopk_feature 第一阶段高权重体素块权重
    def forward(self,x,Ftopk_feature= None,help_prob = False):
            x = x.squeeze(0)
            H = self.feature_extractor_part1(x)
            H = H.view(-1, 128 * 1 * 1 * 1)
            H = self.feature_extractor_part2(H)  # NxL
            # H, SA_weight = self.attention(H)  # NxK
            # 将 Ftopk_feature 的每一个复制成8个，一个大块8个小块
            expanded_Ftopk_feature = torch.cat(
                [Ftopk_feature[i:i + 1, :].repeat(8, 1) for i in range(Ftopk_feature.shape[0])], dim=0)
            # 将 expanded_Ftopk_feature 和 H 拼接
            H = torch.cat([expanded_Ftopk_feature, H], dim=1)
            H = self._fc(H)
            A_V = self.attention_V(H)  # NxD
            A_U = self.attention_U(H)  # NxD
            A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N
            M = torch.mm(A, H)  # KxL
            results_dict = {}
            if help_prob:
                Y_prob = self.classifier(M)
                Y_hat = torch.ge(Y_prob, 0.5).float()
                results_dict.update({'Y_prob': Y_prob, 'Y_hat': Y_hat})
            return  M,results_dict