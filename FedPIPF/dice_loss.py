import torch
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        # print(self.inter, self.union, t,torch.sum(input),torch.sum(target))
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


import torch

def dice_coeff(input: torch.Tensor, target: torch.Tensor, epsilon=1e-6):
    """
    Dice coefficient for binary segmentation.
    input:  [B, H, W]  sigmoid 后的预测值
    target: [B, H, W]  GT mask，应该是 0/1
    """

    # ✅ 保护输入范围，防止 CUDA 报错
    input = torch.clamp(input, 0, 1)
    target = torch.clamp(target, 0, 1)

    # 转成 float
    input = input.float()
    target = target.float()

    # 展开
    input_flat = input.contiguous().view(input.shape[0], -1)
    target_flat = target.contiguous().view(target.shape[0], -1)

    intersection = (input_flat * target_flat).sum(dim=1)
    union = input_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice.mean()

