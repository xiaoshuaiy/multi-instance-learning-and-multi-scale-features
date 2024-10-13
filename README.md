运行步骤
1. 运行dataloader/subject_split 划分患者，获得训练、测试患者集合
2. 获取中心点坐标位置，运行utils/get_center.py
3. 运行train.py 训练
4. 运行test.py 测试

所需环境：
python 3.7
torch 1.7.1
torchvision 0.8.2
SimpleITK 1.2.4
numpy 1.19.5
os 2.7.18
scipy.io 1.5.4


