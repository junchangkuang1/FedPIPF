import torch
import torch.nn as nn
import numpy as np
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

class MedSAMWrapper(nn.Module):
    def __init__(self, model_type='vit_b', checkpoint_path=None, image_size=256):
        super().__init__()
        assert checkpoint_path is not None, "Checkpoint path must be specified."
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.transform = ResizeLongestSide(image_size)
        self.image_size = image_size

        # 默认冻结 encoder，节省资源
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, mask_gt):
        """
        image: [B, 3, H, W] tensor (float, 0~1), B 必须为 1（MedSAM 原生不支持 batch）
        mask_gt: [B, H, W] tensor，值为 0/1
        """
        assert image.shape[0] == 1, "MedSAM 暂时仅支持 batch size=1"

        # 转换尺寸、获取嵌入
        image_np = image[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        image_np = (image_np * 255).astype(np.uint8)
        orig_size = image_np.shape[:2]

        # Resize & embed
        input_image = self.transform.apply_image(image_np)
        input_tensor = torch.as_tensor(input_image.transpose(2, 0, 1)).float().to(image.device) / 255.
        input_tensor = input_tensor.unsqueeze(0)  # [1, 3, H, W]

        image_embedding = self.model.image_encoder(input_tensor)

        # 使用 dummy 中心点作为 prompt
        h, w = orig_size
        input_point = torch.tensor([[[w // 2, h // 2]]], dtype=torch.float, device=image.device)
        input_label = torch.tensor([[1]], dtype=torch.int, device=image.device)

        # prompt 编码
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=(input_point, input_label),
            boxes=None,
            masks=None,
        )

        low_res_masks, _ = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upsample
        upscaled_logits = torch.nn.functional.interpolate(
            low_res_masks,
            size=orig_size,
            mode='bilinear',
            align_corners=False,
        )

        return upscaled_logits  # shape: [1, 1, H, W]
