import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import random
from PIL import Image
import cv2
import torch.nn as nn
from unet import UNet
from scipy import ndimage
from scipy.ndimage import zoom
from torchvision import transforms
from scipy import ndimage
from dice_loss import dice_coeff
sigmoid = nn.Sigmoid()
TRAIN_RATIO = 0.8
RS = 30448  # random state
N_CHANNELS, N_CLASSES = 1, 1
bilinear = True
BATCH_SIZE, EPOCHS = 16, 200
# BATCH_SIZE, EPOCHS = 16, 100
img_size = 224
CROP_SIZE = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = r'F:\pythonProject\FedDUS\public_datasettxt_tvt'
# data_path = r"G:\secondWork_datasets\nolidc_npz"
class BasicDataset(Dataset):
    def __init__(self, base_dir: str, split, train = False, transforms = None):
        print(split)
        self.transform = transforms  # using transform in torch!
        self.split = split
        self.image_list = []
        self._base_dir = base_dir
        self.train = train
        if train:
            with open(self._base_dir+'/{}_train.txt'.format(split), 'r') as f:
                self.image_list = f.readlines()
        else:
            with open(self._base_dir+'/{}_test.txt'.format(split), 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]
        print("{} has total {} samples".format(split,len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        # path = os.path.join(r"G:\secondWork_datasets", self.split , image_name)
        path = os.path.join(r"F:\pythonProject\FedDUS\public_dataset", self.split, image_name)
        # path = os.path.join(r"G:\secondWork_datasets\nolidc_npz", self.split, image_name)
        img = np.load(path)['img']
        mask = np.load(path)['mask']

        sample = {'img': img, 'mask': mask,'filename':path.split('\\')[-1]}
        if self.transform:
            sample = self.transform(sample)
        return sample

class RandomGenerator(object):
    def __init__(self, output_size, train = False):
        self.output_size = output_size
        self.train = train

    def __call__(self, sample):
        img,mask,filename = sample['img'],sample['mask'],sample['filename']
        x, y = img.shape
        # img[img < -1000] = -1000
        # img[img > 150] = 150
        # img = (img-np.min(img))/(np.max(img)-np.min(img))
        if x != self.output_size[0] or y != self.output_size[1]:
            img = zoom(img, (self.output_size[0] / x, self.output_size[1] / y), order = 0)  # why not 3?
            mask = zoom(mask, (self.output_size[0] / x, self.output_size[1] / y), order = 0)
        mask[mask >= 1] = 1
        flag = 0
        if np.sum(mask)==0:
            flag = 0
        elif np.sum(mask)>1:
            distance_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)

            # 获取最大距离
            max_distance = np.max(distance_transform)

            # 直径是最大距离的两倍
            diameter = max_distance * 2
            flag = round(diameter)

        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(mask.astype(np.float32))

        sample = {'img': img, 'mask': mask, 'filename':filename,'type':flag}
        return sample


def dis(input, target):
    """Dice coeff for batches"""
    intersection = torch.sum(input * target)
    dice = (2. * intersection+0.00001) / (torch.sum(input) + torch.sum(target)+0.00001)
    return dice
CLIENTS = ['JM']
# CLIENTS = ['Client2','Client3','Client1','Client4']
lung_dataset = dict()
testing_clients = dict()
model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)
for client in CLIENTS:
        lung_dataset[client + '_test'] = BasicDataset(data_path, split=client, train=False,
                                                      transforms=transforms.Compose(
                                                          [RandomGenerator(output_size=CROP_SIZE, train=False)]))

for client in CLIENTS:
    testing_clients[client] = DataLoader(lung_dataset[client + '_test'], batch_size=1, shuffle=False, num_workers=0)

model_path = r'C:\Users\Admin\Desktop\ourmodel\first\FedMix-main\net_0.7956533262914844.pth'  # 替换为实际路径
if os.path.exists(model_path):
    model = torch.load(model_path)
    print("模型已成功加载。")
else:
    print("模型文件未找到，检查路径。")
from scipy.spatial.distance import cdist
running_loss = 0.0
all_list = []
for batch in testing_clients['JM']:  # 替换 'SX' 为需要的客户端
    inputs, labels = batch['img'].to(device), batch['mask'].to(device)

    filenames = batch['filename'][0]
    outputs = sigmoid(model(inputs)).squeeze()
    mask_pred = (outputs > 0.5).float()
    if torch.sum(labels) == 0 or torch.sum(mask_pred) == 0:
        continue
        # loss_dice = 1
        # all.append(loss_dice)
    # if torch.sum(mask_pred) == 0:
    #     print(batch['type'])
    #     continue
    if batch['type'] <4:
        continue
    else:
        loss_dice = dis(mask_pred.type(torch.float), labels.squeeze())
        print(batch['type'],loss_dice)
        if loss_dice>0.1:
            all_list.append(loss_dice.item())
    # output_image = mask_pred.cpu().detach().numpy()
    # output_image = zoom(output_image, (512 / 224, 512 / 224), order=0)
    # output_filename = os.path.join(r"C:\Users\Admin\Desktop\ourmodel\first\pred\FedDUS\SX", r'{}.jpg'.format(filenames.split('.')[0]))
    # image = Image.fromarray(output_image * 255)  # 乘以 255 将 0 和 1 映射到 0 和 255 之间
    # image = image.convert('L')
    # image.save(output_filename)

    # loss_dice = dice_coeff(mask_pred, labels.squeeze())
    # all.append(loss_dice.item())
print(all_list)
print(sum(all_list)/len(all_list))



