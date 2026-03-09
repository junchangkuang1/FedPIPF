import os
import tqdm
import numpy as np
from glob import glob
import SimpleITK as sitk
images_path = glob(r"G:\secondWork_datasets\first\nii\JM\imagesTr\*")
for img_p in tqdm.tqdm(images_path):
    label_path = img_p.replace('_0000','_segmentation').replace('imagesTr','labelsTr')
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_p))
    label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
    # img[img < -1000] = 1000
    # img[img > 150] = 150
    if np.max(label) == 0:
        print(img_p)
        continue
    area = np.where(label==1)[0]
    min_a = np.min(area) - 15
    max_a = np.max(area) + 15

    if max_a > img.shape[0]:
        max_a = img.shape[0]
    if min_a < 0:
        min_a = 0
    for slice in range(min_a,max_a):
        img_slice = img[slice]
        label_slice = label[slice]
        if not os.path.exists(r'G:\secondWork_datasets\JM'):
            os.makedirs(r'G:\secondWork_datasets\JM')
        # print(r'G:\secondWork_datasets\lidc_npz\Client1\{}_{}.npz'.format(img_p.split('\\')[-1].split('.')[0],slice))
        np.savez(r'G:\secondWork_datasets\JM\{}_{}.npz'.format(img_p.split('\\')[-1].split('.')[0],slice), img=img_slice, mask=label_slice)

