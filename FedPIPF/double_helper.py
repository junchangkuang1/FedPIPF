# %%

import os
import copy

import numpy as np
import torch
import torch.nn as nn
import math
from torchvision import transforms
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from unet import UNet
from dice_loss import dice_coeff

import matplotlib.pyplot as plt
from IPython.display import clear_output
import re
############################
# Helper func
############################
from helper import *
#################################
TRAIN_RATIO = 0.8
RS = 30448  # random state
N_CHANNELS, N_CLASSES = 1, 1
bilinear = True
BATCH_SIZE, EPOCHS = 16,250
# BATCH_SIZE, EPOCHS = 16, 100
img_size = 1024
CROP_SIZE = (1024, 1024)
#########################################
data_path = r'/root/autodl-tmp/project/kjc/data'
# data_path = r'G:\thirdWork_datasets\second'
# data_path = r'/root/autodl-tmp'
# CLIENTS = ['XY','SX','GD','JM']
CLIENTS = ['BIDMC','I2CVB','RUNMC','UCL','HK','BMC']
TOTAL_CLIENTS = len(CLIENTS)
# 假设路径为 './checkpoints/medsam_vit.pth'

device = torch.device('cuda:0')
LR, WD, TH = 1e-5, 1e-5, 0.9
import torch
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
# 指定模型类型为 vit_b
model_type = "vit_b"

# 你的 MedSAM 权重文件路径（请修改为你实际保存的路径）
checkpoint_path = "/root/autodl-tmp/project/kjc/medsam_vit_b.pth"

# 加载模型
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam.to(device)
def load_sam_model(checkpoint_path):
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)  # 或者具体模型名称如 "vit_h"

    return sam
# 初始化预测器（用于后续图像推理）
predictor = SamPredictor(sam)
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

import torch
import torch.nn as nn
import torch.nn.functional as F

class MedSAMWrapper(nn.Module):
    def __init__(self, sam_checkpoint_path, device, img_size=(256, 256)):
        super().__init__()
        self.device = device
        self.sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint_path).to(device)
        self.image_encoder = self.sam.image_encoder
        self.prompt_encoder = self.sam.prompt_encoder

        self.image_size = img_size

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 2, 128, 3, padding=1),  # 原特征+prompt特征
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, image, point_coords=None, point_labels=None):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        B, C, H, W = image.shape

        if C == 1:
            image = image.repeat(1, 3, 1, 1)

        image = F.interpolate(image, size=self.image_size, mode='bilinear', align_corners=False)

        image_embedding = self.image_encoder(image)  # [B, 256, h, w]

        # ========== 处理点标注 ==========
        if point_coords is not None and point_labels is not None:
            # 确保 point_coords 形状为 [B, N_points, 2]
            if point_coords.dim() == 2:
                # 如果是 [B, 2]，加一个点维度变成 [B, 1, 2]
                point_coords = point_coords.unsqueeze(1)
            elif point_coords.dim() == 3:
                # 如果是 [B, 1, 2]，不变
                pass
            else:
                raise ValueError(f"Unexpected point_coords shape: {point_coords.shape}")

            sparse_embeddings, _ = self.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None
            )  # sparse_embeddings: [B, N_points, 256]

            # 取点坐标均值，做成与image_embedding相同空间维度的额外通道
            prompt_feat = point_coords.clone()  # [B, N_points, 2]
            prompt_feat_mean = prompt_feat.mean(dim=1)  # [B, 2]

            # 确认 prompt_feat_mean 的元素数量等于 B*2，才能 reshape
            assert prompt_feat_mean.numel() == B * 2, \
                f"prompt_feat_mean numel {prompt_feat_mean.numel()} != B*2 ({B*2})"

            prompt_feat = prompt_feat_mean.view(B, 2, 1, 1).expand(-1, -1,
                                                                   image_embedding.shape[2],
                                                                   image_embedding.shape[3])

        else:
            # 若无点引导，使用全零prompt
            prompt_feat = torch.zeros((B, 2, image_embedding.shape[2], image_embedding.shape[3]),
                                      device=image.device)

        # ========== 拼接并解码 ==========
        features = torch.cat([image_embedding, prompt_feat], dim=1)  # [B, 256+2, h, w]
        mask_pred = self.decoder(features)  # [B, 1, h, w]
        mask_pred = F.interpolate(mask_pred, size=(H, W), mode='bilinear', align_corners=False)

        return mask_pred.squeeze(1)  # [B, H, W]




lung_dataset = dict()
for client in CLIENTS:
    lung_dataset[client + '_train'] = BasicDataset(data_path, split=client, train=True,
                                                   transforms=transforms.Compose(
                                                       [RandomGenerator(output_size=(224,224), train=True)]))
    if client != 'GX':
        lung_dataset[client + '_test'] = BasicDataset(data_path, split=client, train=False,
                                                      transforms=transforms.Compose(
                                                          [RandomGenerator(output_size=(224,224), train=False)]))



# %% md

## Initialize the weights

# %%

TOTAL_DATA = []
for client in CLIENTS:
    if client != 'Interobs' and client != 'Lung1':
        print(len(lung_dataset[client + '_train']))
        TOTAL_DATA.append(len(lung_dataset[client + '_train']))

DATA_AMOUNT = sum(TOTAL_DATA)
WEIGHTS = [t / DATA_AMOUNT for t in TOTAL_DATA]

ORI_WEIGHTS = copy.deepcopy(WEIGHTS)

score = [0, 0, 0, 0,0,0]
dice = [0, 0, 0, 0,0,0]

# %% md

# storage file

# %%

training_clients, testing_clients = dict(), dict()
for client in CLIENTS:
    if client + '_train' in lung_dataset:
        training_clients[client] = lung_dataset[client + '_train']
    else:
        print(f"Warning: No training data for client {client}")
    if client + '_test' in lung_dataset:
        testing_clients[client] = lung_dataset[client + '_test']
    else:
        print(f"Warning: No testing data for client {client}")

acc_train, acc_valid, loss_train, loss_test = dict(), dict(), \
                                              dict(), dict()
loss_test = dict()
alpha_acc = []

nets, optimizers = dict(), dict()

acc_train1 = []
loss_train1 = []
# %%

sam_checkpoint_path = "/root/autodl-tmp/project/kjc/medsam_vit_b.pth"

nets = {client: {'medsam': None, 'unet': None} for client in CLIENTS}

# 2. 初始化 global 模型，确保格式一样
nets['global'] = {'medsam': None, 'unet': None}
nets['global']['unet'] =UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(device)
ema_net = nets['global']['unet']  # 举例，把global的unet作为ema模型


optimizers = {client: {'medsam': None, 'unet': None} for client in CLIENTS}




for client in CLIENTS:
    if nets[client]['medsam'] is None:
        nets[client]['medsam'] = MedSAMWrapper(sam_checkpoint_path=sam_checkpoint_path, device=device, img_size=(1024, 1024)).to(device)
    if nets[client]['unet'] is None:
        nets[client]['unet'] =UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(device)

    optimizers[client]['medsam'] = optim.Adam(nets[client]['medsam'].parameters(), lr=LR, weight_decay=WD)
    optimizers[client]['unet'] = optim.Adam(nets[client]['unet'].parameters(), lr=LR, weight_decay=WD)



# 初始化训练和验证列表
for client in CLIENTS:
    acc_train[client], acc_valid[client] = [], []
    loss_train[client], loss_test[client] = [], []


for client in CLIENTS:
    if client == 'Lung1' or client == 'Interobs':
        print(client)
        print(len(lung_dataset[client]))


WEIGHTS_POSTWARMUP = [0.3,0.3,0.325,0.025,0.025,0.025]  # put more weight to client with strong supervision
# WARMUP_EPOCH = 100
WARMUP_EPOCH =1
CLIENTS_SUPERVISION = ['labeled', 'labeled','labeled','unlabeled','unlabeled','unlabeled']
# CLIENTS_SUPERVISION = ['labeled', 'labeled']
# %% md

### First 150 epochs warmup by training locally on labeled clients

# %%

best_avg_acc, best_epoch_avg = 0, 0
index = []
iter_nums = 0

USE_UNLABELED_CLIENT = False
loss = []

best_metrics_warmup = {
    'epoch': -1,
    'acc': 0.0,
    'jc': 0.0,
    'assd': float('inf'),
    'hd95': float('inf')
}
best_metrics_post = {
    'epoch': -1,
    'acc': 0.0,
    'jc': 0.0,
    'assd': float('inf'),
    'hd95': float('inf')
}

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusionModule(nn.Module):
    def __init__(self):
        super(AttentionFusionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred1, pred2):
        # pred1, pred2: [B, H, W] or [B, 1, H, W]
        if pred1.dim() == 3:
            pred1 = pred1.unsqueeze(1)
        if pred2.dim() == 3:
            pred2 = pred2.unsqueeze(1)

        combined = torch.cat([pred1, pred2], dim=1)  # [B, 2, H, W]
        attention_map = self.sigmoid(self.conv1(combined))  # [B, 1, H, W]
        fused = attention_map * pred1 + (1 - attention_map) * pred2
        return fused.squeeze(1)  # 返回 [B, H, W]

fusion_module = AttentionFusionModule().to(device)


for epoch in range(EPOCHS):
    epoch_train_acc, epoch_test_acc,epoch_test_hd95,epoch_test_assd,epoch_test_jc = [],[],[],[],[]
    epoch_loss = []

    print('epoch {} :'.format(epoch))
    if epoch == WARMUP_EPOCH:
        WEIGHTS = WEIGHTS_POSTWARMUP
        USE_UNLABELED_CLIENT = True

    index.append(epoch)

    #################### copy fed model ###################

    copy_fed(CLIENTS, nets, model_type='unet', fed_name='global')

    #### conduct training #####
    for client, supervision_t in zip(CLIENTS, CLIENTS_SUPERVISION):
        if supervision_t == 'unlabeled':
            if not USE_UNLABELED_CLIENT:
                acc_train[client].append(0)
                loss_train[client].append(0)
                continue

        if client != 'Interobs' and client != 'Lung1':
            print(f"Current client: {client}")

            acc_, loss_ = train_model_sam(
                epoch,
                training_clients[client],
                optimizers[client]['medsam'],
                device,
                nets[client]['medsam'],
                acc=acc_train[client],
                loss=loss_train[client],
                learning_rate=LR,
                iter_num=iter_nums
            )
            acc_unet, loss_unet = train_model_unet(
                epoch,
                training_clients[client],
                optimizers[client]['unet'],
                device,
                nets[client]['unet'],
                ema_model=ema_net,
                net_medsam=nets[client]['medsam'],  # 新增传入medsam模型
                fusion_module=fusion_module,  # 新增传入融合模块
                acc=acc_train[client],
                loss=loss_train[client],
                supervision_type=supervision_t,
                learning_rate=LR,
                iter_num=iter_nums
            )
    total_loss = loss_unet + loss_
    epoch_loss.append(total_loss)
    epoch_train_acc.append(acc_unet)
    epoch_loss.append(loss_)
    epoch_train_acc.append(acc_)
    loss_train1.append(sum(epoch_loss)/len(epoch_loss))
    acc_train1.append(sum(epoch_train_acc)/len(epoch_train_acc))

    nets_unet = {client: nets[client]['unet'] for client in CLIENTS}
    nets_unet['global'] = nets['global']['unet']
    iter_nums += 1
    aggr_fed(CLIENTS, WEIGHTS, nets_unet, fed_name='global')

    ################### test ################################
    avg_acc = 0.0
    for order, (client, supervision_t) in enumerate(zip(CLIENTS, CLIENTS_SUPERVISION)):
        # testloader, net, device, acc = None, loss = None
        if client == 'GX':
            continue
        # acc_test,jc,assd,hd95 = test_sam(testing_clients[client], nets['global'], device, acc_valid[client], loss_test[client])
        acc_test, jc, assd, hd95 = test_unet(
            testing_clients[client], nets_unet['global'], device, acc_valid[client], loss_test[client]
        )

        epoch_test_jc.append(jc)
        epoch_test_assd.append(assd)
        epoch_test_hd95.append(hd95)
        epoch_test_acc.append(acc_test)
        avg_acc += acc_valid[client][-1]
        # if not USE_UNLABELED_CLIENT:
        if supervision_t == "labeled":
            score[order] = acc_valid[client][-1]
        # else:
        dice[order] = acc_valid[client][-1]
    ######################################################
    ####### dynamic weighting #########
    ###################################
    print('test score')
    print("acc is :", epoch_test_acc)
    print("jc is :", epoch_test_jc)
    print("assd is :", epoch_test_assd)
    print("hd95 is :", epoch_test_hd95)
    # 当前 epoch 的平均指标
    avg_acc = np.mean(epoch_test_acc)
    avg_jc = np.mean(epoch_test_jc)
    avg_assd = np.mean(epoch_test_assd)
    avg_hd95 = np.mean(epoch_test_hd95)

    # 更新最佳性能
    if epoch < WARMUP_EPOCH:
        if avg_acc > best_metrics_warmup['acc']:
            best_metrics_warmup.update({
                'epoch': epoch,
                'acc': avg_acc,
                'jc': avg_jc,
                'assd': avg_assd,
                'hd95': avg_hd95
            })
    else:
        if avg_acc > best_metrics_post['acc']:
            best_metrics_post.update({
                'epoch': epoch,
                'acc': avg_acc,
                'jc': avg_jc,
                'assd': avg_assd,
                'hd95': avg_hd95
            })

    # 打印最佳结果
    print(f"\n[Warmup Best @ Epoch {best_metrics_warmup['epoch']}]: "
          f"Acc: {best_metrics_warmup['acc']:.4f}, "
          f"JC: {best_metrics_warmup['jc']:.4f}, "
          f"ASSD: {best_metrics_warmup['assd']:.4f}, "
          f"HD95: {best_metrics_warmup['hd95']:.4f}")

    print(f"[Post-Warmup Best @ Epoch {best_metrics_post['epoch']}]: "
          f"Acc: {best_metrics_post['acc']:.4f}, "
          f"JC: {best_metrics_post['jc']:.4f}, "
          f"ASSD: {best_metrics_post['assd']:.4f}, "
          f"HD95: {best_metrics_post['hd95']:.4f}\n")
    WEIGHTS_DATA = copy.deepcopy(ORI_WEIGHTS)
    denominator = sum(score)
    score = [s / denominator for s in score]
    for order, _ in enumerate(WEIGHTS_DATA):
        WEIGHTS_DATA[order] = WEIGHTS_DATA[order] * score[order]

    ### normalize #####################
    denominator = sum(WEIGHTS_DATA)
    WEIGHTS_DATA = [w / denominator for w in WEIGHTS_DATA]

    if USE_UNLABELED_CLIENT:
        for order, supervision_t in enumerate(CLIENTS_SUPERVISION):
            if supervision_t == "labeled":
                WEIGHTS[order] = copy.deepcopy(WEIGHTS_DATA[order] * 0.95)
    else:
        WEIGHTS = copy.deepcopy(WEIGHTS_DATA)

    print("weight is::::", WEIGHTS)
    # WEIGHTS_DATA = copy.deepcopy(ORI_WEIGHTS)
    # denominator = sum(score)
    # score = [s / denominator for s in score]
    # for order, _ in enumerate(WEIGHTS_DATA):
    #     WEIGHTS_DATA[order] = WEIGHTS_DATA[order] * score[order]
    #
    # ### normalize #####################
    # denominator = sum(WEIGHTS_DATA)
    # WEIGHTS_DATA = [w / denominator for w in WEIGHTS_DATA]
    #
    # if USE_UNLABELED_CLIENT:
    #     for order, supervision_t in enumerate(CLIENTS_SUPERVISION):
    #         if supervision_t == "labeled":
    #             WEIGHTS[order] = copy.deepcopy(WEIGHTS_DATA[order] * 0.95)
    # else:
    #     WEIGHTS = copy.deepcopy(WEIGHTS_DATA)
    #
    # print("weight is::::", WEIGHTS)
    # w = []
    # s = []
    # w.append(WEIGHTS)
    # s.append(score)
    #
    # avg_acc = avg_acc / TOTAL_CLIENTS
    # save_model_4(r'C:\Users\Admin\Desktop\ourmodel',avg_acc, epoch, nets, ema_net)
    # if epoch == 0 or epoch >200:
    #     save_model_4(r'C:\Users\Admin\Desktop\onstep\second\FedDUS',sum(epoch_test_acc)/TOTAL_CLIENTS, epoch, nets)
    #     # save_model_4(r'/root/autodl-tmp/FedDUS',sum(epoch_test_acc)/TOTAL_CLIENTS, epoch, nets)
    # ############################################################
    # if avg_acc > best_avg_acc:
    #     best_avg_acc = avg_acc
    #     best_epoch = epoch
    #     save_model_4(PTH, epoch, nets, ema_net)
    # save_mode_path = "F:\pythonProject\FedDUS\model/epoch/"
    # torch.save(nets['global'].state_dict(), save_mode_path + 'epoch_' + str(epoch) + '.pth')
    # torch.save(ema_net.state_dict(), save_mode_path + 'emaepoch_' + str(epoch) + '.pth')
# with open(r'/root/autodl-tmp/FedDUS/loss_train.txt', 'a') as f:
#     f.writelines(str(loss_train1))
# # with open(r'/root/autodl-tmp/FedDUS/onstep\acc_train.txt', 'a') as f1:
# #     f1.writelines(str(acc_train1))
# with open(r'C:\Users\Admin\Desktop\onstep\second/FedDUS/dus_loss_train.txt', 'a') as f:
#     f.writelines(str(loss_train1))
# with open(r'C:\Users\Admin\Desktop\onstep\second\FedDUS/dus_acc_train.txt', 'a') as f1:
#     f1.writelines(str(acc_train1))
    # with open(r"F:\pythonProject\FedDUS\model\loss.txt",'w') as f:
    #     f.writelines(loss)
    ################################
    # plot #########################
    ################################
    # np.save(PTH + '/outcome/acc_train', acc_train)
    # np.save(PTH + '/outcome/acc_test', acc_valid)
    # np.save(PTH + '/outcome/loss_train', loss_train)
    # np.save(PTH + '/outcome/weight', w)
    # np.save(PTH + '/outcome/score', s)
    # clear_output(wait=True)
