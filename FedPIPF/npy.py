import os
import glob
import cv2
import numpy as np
pth = r"F:\pythonProject\FedDUS\public_dataset\SX\*.npz"

files = glob.glob(pth)

for f in files:
    data = np.load(f)
    image = data['img']
    mask = data['mask']
    image_slice = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mask_slice = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(r'C:\Users\Admin\Desktop\ourmodel\first\pred\true\image\{}.jpg'.format(f.split('\\')[-1].split('.')[0]),
                image_slice)
    cv2.imwrite(r'C:\Users\Admin\Desktop\ourmodel\first\pred\true\mask\{}.jpg'.format(f.split('\\')[-1].split('.')[0]),
                mask_slice)