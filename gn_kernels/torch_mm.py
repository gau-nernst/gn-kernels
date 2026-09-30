import torch.nn.functional as F
from torch import Tensor


def mxfp8_mm(x: Tensor, x_sf: Tensor, w: Tensor, w_sf: Tensor):
    scale = F.ScalingType.BlockWise1x32
    swizzle = F.SwizzleType.SWIZZLE_32_4_4
    return F.scaled_mm(x, w.T, x_sf, scale, w_sf, scale, swizzle, swizzle)


def nvfp4_mm(x: Tensor, x_sf: Tensor, x_scale: Tensor, w: Tensor, w_sf: Tensor, w_scale: Tensor) -> Tensor:
    recipe = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
    swizzle = [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]
    return F.scaled_mm(x, w.T, [x_sf, x_scale], recipe, [w_sf, w_scale], recipe, swizzle, swizzle)
