import numpy as np
from PIL import Image
import glob
paths = glob.glob(r"C:\Users\Admin\Desktop\onstep\choose\image\*.jpg")
# 假设原始图像为image，并且裁剪后的mask
for p in paths:
    image_path = p
    mask_path = p.replace('image','radio')

    # 读取原始图像
    original_image = np.array(Image.open(image_path).convert('L'))  # 原始图像转为灰度

    # 读取裁剪后的mask
    mask = np.array(Image.open(mask_path).convert('L'))  # 已经裁剪的mask区域

    # 获取原始图像尺寸
    original_height, original_width = original_image.shape

    # 创建一个与原始图像相同大小的全零（黑色）图像
    restored_mask = np.zeros((original_height, original_width), dtype=np.uint8)

    # 将裁剪后的mask放回原图相应的位置
    restored_mask[64:448, 64:448] = mask  # 恢复裁剪的部分

    restored_mask_image = Image.fromarray(restored_mask)
    # restored_mask_image.show()


    restored_mask_image.save(p.replace('image','radio512'))
