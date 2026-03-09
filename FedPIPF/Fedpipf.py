# %%
import os
import copy
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
import torch
import torch.nn as nn
import math
from torchvision import transforms
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
# import torch
# torch.cuda.empty_cache()
import torch
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
BATCH_SIZE, EPOCHS = 8,150
# BATCH_SIZE, EPOCHS = 16, 100
img_size = 224
CROP_SIZE = (224, 224)
#########################################
# data_path = r'/root/autodl-tmp/project/jpg'

data_path = r'/root/autodl-tmp/project/tg3k'
CLIENTS = ['center_0','center_1','center_2','center_3','center_4']


# data_path = r'G:\thirdWork_datasets\second'
# data_path = r'/root/autodl-tmp'
# CLIENTS = ['BIDMC','I2CVB','RUNMC','UCL','HK','BMC']
# CLIENTS = ['center_0','center_0','center_0','center_0','center_0','BMC']

# CLIENTS = ['GD','SX','JM','XY']
TOTAL_CLIENTS = len(CLIENTS)
# 假设路径为 './checkpoints/medsam_vit.pth'
# print(1111)
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
checkpoint_path = "/root/autodl-tmp/project/medsam_vit_b.pth"
def get_feature_embedding(self, x):
    feat = self.encoder(x)  # 假设 encoder 是输出 [B, C, H, W]
    pooled = torch.mean(feat, dim=[2, 3])  # [B, C]
    return pooled  # ✅ 不转 numpy

# get_client_embedding 函数
def get_client_embedding(model, dataloader, device):
    model.eval()
    embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['img'].unsqueeze(0).to(device)
            output = model.get_feature_embedding(images)  # 返回 numpy
            embeddings.append(output)

    return np.mean(embeddings, axis=0).reshape(-1)  # 或 .squeeze()


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
            # 将 point_coords 和 point_labels reshape 成 [B, N_points, 2] 和 [B, N_points]
            if point_coords.dim() == 2:
                assert B == 1, f"当 point_coords 是 2D 时，batch size 应为 1，但当前是 {B}"
                point_coords = point_coords.unsqueeze(0)  # [1, N_points, 2]
                point_labels = point_labels.unsqueeze(0)  # [1, N_points]
            elif point_coords.dim() == 3:
                assert point_coords.shape[0] == B, \
                    f"point_coords batch ({point_coords.shape[0]}) 与图像 batch ({B}) 不匹配"
            else:
                raise ValueError(f"Unexpected point_coords shape: {point_coords.shape}")

            sparse_embeddings, _ = self.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None
            )  # sparse_embeddings: [B, N_points, 256]

            # 取点坐标均值，做成与 image_embedding 相同空间维度的额外通道
            prompt_feat_mean = point_coords.float().mean(dim=1)  # [B, 2]

            prompt_feat = prompt_feat_mean.view(B, 2, 1, 1).expand(
                -1, -1,
                image_embedding.shape[2],
                image_embedding.shape[3]
            )

        else:
            # 若无点引导，使用全零 prompt
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

score = [0, 0, 0,0,0]
dice = [0, 0, 0,0,0]

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

sam_checkpoint_path = "/root/autodl-tmp/project/medsam_vit_b.pth"

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



WEIGHTS_POSTWARMUP = [0.2,0.25,0.25,0.025,0.025]  # put more weight to client with strong supervision
# WARMUP_EPOCH = 100
WARMUP_EPOCH =50
CLIENTS_SUPERVISION = ['unlabeled', 'labeled','unlabeled','labeled','labeled']
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

from peft import LoraConfig, get_peft_model, TaskType

# 推荐配置
import torch
from peft import get_peft_model, LoraConfig, TaskType

# 假设你已经有了加载好的 MedSAM 模型实例 sam
# sam = ...

# 1. 配置 LoRA，只针对 mask_decoder 的关键模块做微调
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        # 结合你之前的结构，mask_decoder中常用的qkv, proj, lin1, lin2等全连接层
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "lin1",
        "lin2",
    ],
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)

# 2. 将 LoRA 应用到 mask_decoder 子模块
# 先单独拿出 mask_decoder，然后用 PEFT 包装
sam.mask_decoder = get_peft_model(sam.mask_decoder, lora_config)

# 3. 冻结整个模型其他参数（除了 LoRA 引入的参数）
for name, param in sam.named_parameters():
    if "mask_decoder" not in name:
        param.requires_grad = False

# 4. LoRA 参数默认是 requires_grad=True，训练时只会更新这些
# 你可以检查一下可训练参数：
trainable_params = [p for p in sam.parameters() if p.requires_grad]
print(f"Trainable params count: {sum(p.numel() for p in trainable_params)}")

# 5. 定义优化器，只优化可训练参数
optimizer = torch.optim.Adam(trainable_params, lr=1e-4)

# 之后按正常流程训练sam

# LoRA 应用到 image_encoder + prompt_encoder

#
# for client in CLIENTS:
#
#     nets[client]['medsam'] = get_peft_model(nets[client]['medsam'], peft_config)
#     nets[client]['medsam'].to(device)

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
            acc_, loss_ = train_model_sam_lora(
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
            print("全局模型已经保存")
            save_mode_path = "/root/autodl-tmp/project/tg3k/"
            torch.save(nets['global']['unet'].state_dict(), save_mode_path + 'epoch_' + str(epoch) + '.pth')

    # 打印最佳结果
    print(f"\n[Warmup Best @ Epoch {best_metrics_warmup['epoch']}]: "
          f"Acc: {best_metrics_warmup['acc']:.4f}, "
          f"JC: {best_metrics_warmup['jc']:.4f}, "
          f"ASSD: {best_metrics_warmup['assd']:.4f}, "
          f"HD95: {best_metrics_warmup['hd95']:.4f}")
    print("2025-10-25")
    print(f"[Post-Warmup Best @ Epoch {best_metrics_post['epoch']}]: "
          f"Acc: {best_metrics_post['acc']:.4f}, "
          f"JC: {best_metrics_post['jc']:.4f}, "
          f"ASSD: {best_metrics_post['assd']:.4f}, "
          f"HD95: {best_metrics_post['hd95']:.4f}\n")
    # 1. 先基于客户端训练数据量计算基础权重
    # ==== Step 0: 提前获取嵌入维度（仅在第一次运行时执行一次）====
    # ==== Step 0: 获取嵌入维度（只需一次）====
    # Step 0: 获取嵌入维度（只需一次）
    if 'embedding_dim' not in globals():
        for client in CLIENTS:
            if CLIENTS_SUPERVISION[CLIENTS.index(client)] == "labeled":
                emb = get_client_embedding(nets[client]['unet'], training_clients[client], device)
                embedding_dim = emb.shape[0]
                break
    # Step 1: 样本量权重
    WEIGHTS_DATA = copy.deepcopy(ORI_WEIGHTS)

    # Step 2: 获取所有嵌入（无标签客户端前150轮填0）
    CLIENTS_FEATURE_EMBEDDING = []
    for idx, client in enumerate(CLIENTS):
        if epoch < WARMUP_EPOCH and CLIENTS_SUPERVISION[idx] != "labeled":
            CLIENTS_FEATURE_EMBEDDING.append(np.zeros(embedding_dim))
        else:
            # 错误的写法：
            # emb = get_client_embedding(nets[client], training_clients[client], device)

            # 正确写法（以unet为例）：
            emb = get_client_embedding(nets[client]['unet'], training_clients[client], device)
            CLIENTS_FEATURE_EMBEDDING.append(emb)
    # Step 3: 计算对齐得分（中心对齐）
    CENTER_EMBEDDING = np.mean(CLIENTS_FEATURE_EMBEDDING, axis=0)
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
    ALIGN_SCORES = [cosine_similarity(e, CENTER_EMBEDDING) for e in CLIENTS_FEATURE_EMBEDDING]
    ALIGN_SCORES = [s / (sum(ALIGN_SCORES) + 1e-8) for s in ALIGN_SCORES]
    # Step 4: 计算多样性（表示与所有其他客户端距离的均值）
    def diversity_score(i, all_embeddings):
        others = [all_embeddings[j] for j in range(len(all_embeddings)) if j != i]
        dists = [np.linalg.norm(all_embeddings[i] - emb) for emb in others]
        return np.mean(dists)
    DIVERSITY_SCORES = []
    for i in range(len(CLIENTS)):
        if epoch < WARMUP_EPOCH and CLIENTS_SUPERVISION[i] != "labeled":
            DIVERSITY_SCORES.append(0.0)
        else:
            DIVERSITY_SCORES.append(diversity_score(i, CLIENTS_FEATURE_EMBEDDING))

    DIVERSITY_SCORES = [s / (sum(DIVERSITY_SCORES) + 1e-8) for s in DIVERSITY_SCORES]

    # Step 5: 融合（FedDisAlign）
    ALPHA, BETA, GAMMA =0.5,0.5, 0.5  # 数据量 / 多样性 / 对齐度

    FEDDISALIGN_WEIGHTS = []
    for i in range(len(CLIENTS)):
        if epoch < WARMUP_EPOCH and CLIENTS_SUPERVISION[i] != "labeled":
            client_weight = 0.0
        else:
            client_weight = (
                    ALPHA * WEIGHTS_DATA[i] +
                    BETA * DIVERSITY_SCORES[i] +
                    GAMMA * ALIGN_SCORES[i]
            )
        FEDDISALIGN_WEIGHTS.append(client_weight)

    # Step 6: 归一化
    WEIGHTS = [w / (sum(FEDDISALIGN_WEIGHTS) + 1e-8) for w in FEDDISALIGN_WEIGHTS]

    print("FedDisAlign Weights:", WEIGHTS)

    # 所有客户端数量
    # TOTAL_CLIENTS = len(CLIENTS_SUPERVISION)
    #
    # if epoch <= WARMUP_EPOCH:
    #     # 只训练有标签客户端
    #     labeled_indices = [i for i, t in enumerate(CLIENTS_SUPERVISION) if t == "labeled"]
    #     num_labeled = len(labeled_indices)
    #
    #     WEIGHTS = [0.0] * TOTAL_CLIENTS
    #     for i in labeled_indices:
    #         WEIGHTS[i] = 1.0 / num_labeled  # 均值分配
    # else:
    #     # 所有客户端都训练，均值权重
    #     WEIGHTS = [1.0 / TOTAL_CLIENTS] * TOTAL_CLIENTS
    # print(f"Epoch {epoch} -> Weights: {WEIGHTS}")

    # w = []
    # s = []
    # w.append(WEIGHTS)
    # s.append(score)
    #
    # avg_acc = avg_acc / TOTAL_CLIENTS
    # save_model_4(r'C:\Users\Admin\Desktop\ourmodel',avg_acc, epoch, nets, ema_net)
    # if epoch == 0 or epoch >100:
    #     # save_model_4(r'C:\Users\Admin\Desktop\onstep\second\FedDUS',sum(epoch_test_acc)/TOTAL_CLIENTS, epoch, nets)
    #     save_model_4(r'/root/autodl-tmp/project/FedMedsam_LoRA',sum(epoch_test_acc)/TOTAL_CLIENTS, epoch, nets)
    # # ############################################################
    # # if avg_acc > best_avg_acc:
    #     best_avg_acc = avg_acc
    #     best_epoch = epoch
    #     save_mode_path = "/root/autodl-tmp/project/kjc/FedMedsam_LoRA"
        # torch.save(nets['global']['unet'].state_dict(), save_mode_path + 'epoch_' + str(epoch) + '.pth')
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
