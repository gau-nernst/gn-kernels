import torch
import triton
import triton.language as tl
from torch import Tensor

# drop this workaround once PyTorch 2.8 is released
FP4_DTYPE = getattr(torch, "float4_e2m1fn_x2", torch.uint8)

DTYPE_AMAX_LUT = {
    FP4_DTYPE: 6.0,
    torch.float8_e4m3fn: 448.0,
    torch.float8_e5m2: 57_344.0,
}

DTYPE_POW2_AMAX_LUT = {
    FP4_DTYPE: 4.0,
    torch.float8_e4m3fn: 256.0,
    torch.float8_e5m2: 32_768.0,
}


# https://github.com/NVIDIA/cutlass/blob/v3.9.2/media/docs/cpp/blackwell_functionality.md#scale-factor-layouts
def pack_block_scales_nv(scales: Tensor):
    M, N = scales.shape
    assert M % 128 == 0 and N % 4 == 0  # don't support padding for now
    out = scales.reshape(M // 128, 128, N // 4, 4).transpose(1, 2)  # [num_M_tiles, num_N_tiles, 128, 4]
    out = out.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return out.flatten()


# https://docs.nvidia.com/cuda/cublas/index.html#d-block-quantization
def absmax_to_mx_scales_nv(absmax: Tensor, dtype: torch.dtype):
    assert absmax.dtype == torch.float32
    dtype_amax = DTYPE_AMAX_LUT[dtype]
    scales = absmax / dtype_amax
    bits = scales.view(torch.int32)
    exponent = bits >> 23  # x > 0, so don't need to mask sign bit
    mantissa = bits & 0x7F_FFFF
    return torch.where(
        (((exponent > 0) & (exponent < 254) & (mantissa > 0)) | ((exponent == 0) & (mantissa > 0x40_0000))),
        exponent + 1,
        exponent,
    )


# https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
def absmax_to_mx_scales_ocp(absmax: Tensor, dtype: torch.dtype):
    assert absmax.dtype == torch.float32
    # return (absmax.view(torch.int32) >> 23).clip(2) - 2
    dtype_pow2_amax = DTYPE_POW2_AMAX_LUT[dtype]
    return ((absmax.view(torch.int32) & 0x7F80_0000).view(torch.float32) / dtype_pow2_amax).view(torch.int32) >> 23


def fp32_to_fp4e2m1x2(x: Tensor):
    assert x.dtype == torch.float32

    bits_f32 = x.view(torch.int32)
    sign = (bits_f32 >> 31) & 0x1

    x_abs = x.abs()
    nosign = torch.where(x_abs <= 5.0, 0b0110, 0b0111).int()
    nosign = torch.where(x_abs < 3.5, 0b0101, nosign)
    nosign = torch.where(x_abs <= 2.5, 0b0100, nosign)
    nosign = torch.where(x_abs < 1.75, 0b0011, nosign)
    nosign = torch.where(x_abs <= 1.25, 0b0010, nosign)
    nosign = torch.where(x_abs < 0.75, 0b0001, nosign)
    nosign = torch.where(x_abs <= 0.25, 0b0000, nosign)

    f4_e2m1 = (sign << 3) | nosign

    # pack to 32-bit register
    f4_e2m1x2 = (
        f4_e2m1[..., ::8]
        | (f4_e2m1[..., 1::8] << 4)
        | (f4_e2m1[..., 2::8] << 8)
        | (f4_e2m1[..., 3::8] << 12)
        | (f4_e2m1[..., 4::8] << 16)
        | (f4_e2m1[..., 5::8] << 20)
        | (f4_e2m1[..., 6::8] << 24)
        | (f4_e2m1[..., 7::8] << 28)
    )
    return f4_e2m1x2.view(FP4_DTYPE)


def quantize_mx(x: Tensor, dtype: torch.dtype, compute_scale_method: str = "ocp"):
    assert dtype in (torch.float8_e4m3fn, torch.float8_e5m2, FP4_DTYPE)
    x_blocks_f32 = x.float().unflatten(-1, (-1, 32))  # [..., N/32, 32]
    blocks_amax = x_blocks_f32.abs().amax(dim=-1)  # [..., N/32]

    if compute_scale_method == "ocp":
        block_scales_bits = absmax_to_mx_scales_ocp(blocks_amax, dtype)
    elif compute_scale_method == "nv":
        block_scales_bits = absmax_to_mx_scales_nv(blocks_amax, dtype)
    else:
        raise ValueError(f"Unsupported {compute_scale_method=}")
    scales = block_scales_bits.to(torch.int8).view(torch.float8_e8m0fnu)

    # TODO: division by bit manipulation?
    dtype_amax = DTYPE_AMAX_LUT[dtype]
    x_blocks_f32 = x_blocks_f32 / (block_scales_bits.unsqueeze(-1) << 23).view(torch.float32).clip(1e-12)
    x_blocks_f32 = x_blocks_f32.clip(-dtype_amax, dtype_amax)

    if dtype == FP4_DTYPE:
        xq = fp32_to_fp4e2m1x2(x_blocks_f32)
    else:
        xq = x_blocks_f32.to(dtype)
    xq = xq.view(*x.shape[:-1], -1)

    return xq, scales


def dequantize_mx(xq: Tensor, scales: Tensor):
    assert scales.dtype == torch.float8_e8m0fnu

    if xq.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return (xq.float().reshape(-1, 32) * scales.float().reshape(-1, 1)).view(xq.shape)

    elif xq.dtype == FP4_DTYPE:
        FP4E2M1_LUT = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            device=xq.device,
            dtype=torch.float32,
        )

        # unpack
        xq_i32 = xq.view(torch.int32)
        xq_unpacked = torch.stack(
            [
                xq_i32 & 0xF,
                (xq_i32 >> 4) & 0xF,
                (xq_i32 >> 8) & 0xF,
                (xq_i32 >> 12) & 0xF,
                (xq_i32 >> 16) & 0xF,
                (xq_i32 >> 20) & 0xF,
                (xq_i32 >> 24) & 0xF,
                (xq_i32 >> 28) & 0xF,
            ],
            dim=-1,
        )
        x = FP4E2M1_LUT[xq_unpacked]

        scales_f32 = (scales.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
        x_scaled = x.unflatten(-1, (xq.shape[-1] // 16, 32)) * scales_f32.unsqueeze(-1)
        return x_scaled.view(*xq.shape[:-1], -1)


# https://docs.nvidia.com/cuda/cublas/index.html#d-block-quantization
def quantize_nvfp4(x: Tensor, tensor_scale: Tensor | None = None):
    x_blocks_f32 = x.float().unflatten(-1, (-1, 16))  # [..., N/16, 16]

    q_dtype = FP4_DTYPE
    s_dtype = torch.float8_e4m3fn
    q_dtype_amax = DTYPE_AMAX_LUT[q_dtype]
    s_dtype_amax = DTYPE_AMAX_LUT[s_dtype]

    # tensor_scale can be provided (e.g. for activations)
    # or calculated on-the-fly (e.g. for weights)
    if tensor_scale is None:
        tensor_scale = x_blocks_f32.abs().amax() / (q_dtype_amax * s_dtype_amax)

    blocks_amax = x_blocks_f32.abs().amax(dim=-1)  # [M, N/16]
    scales_f32 = blocks_amax / (q_dtype_amax * tensor_scale).clip(1e-12)
    scales = scales_f32.clip(-s_dtype_amax, s_dtype_amax).to(s_dtype)

    x_blocks_f32 = x_blocks_f32 / (tensor_scale * scales.float()).unsqueeze(-1).clip(1e-12)
    xq = fp32_to_fp4e2m1x2(x_blocks_f32).view(x.shape[0], -1)

    return xq, scales, tensor_scale


@triton.jit
def quantize_nvfp4_triton_kernel(x_ptr, tensor_scale_ptr, q_ptr, s_ptr, stride_xm, stride_xn, N):
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)

    offs_m = pid_m * 128 + tl.arange(0, 128)[:, None]
    offs_n = pid_n * 64 + tl.arange(0, 64)[None, :]
    x = tl.load(x_ptr + offs_m * stride_xm + offs_n * stride_xn)  # [128, 64]
    x_blocks = x.to(tl.float32).reshape(128, 4, 16)  # [128, 4, 16]

    tensor_scale = tl.load(tensor_scale_ptr)

    block_amax = tl.max(x_blocks.abs(), axis=2)  # [128, 4]
    scales_f32 = block_amax / tl.maximum(6.0 * tensor_scale, 1e-12)
    scales_f32 = tl.minimum(tl.maximum(scales_f32, -448.0), 448.0)
    scales = scales_f32.to(tl.float8e4nv)

    # NVIDIA layout
    packed_scales = scales.reshape(4, 32, 4).permute(1, 0, 2).reshape(32, 16)
    offs_m = tl.arange(0, 32)[:, None]
    offs_n = tl.arange(0, 16)[None, :]
    tl.store(s_ptr + (pid_m * tl.num_programs(0) + pid_n) * (32 * 16) + offs_m * 16 + offs_n, packed_scales)

    x_blocks = x_blocks / tl.maximum(scales.to(tl.float32)[:, :, None] * tensor_scale, 1e-12)
    x_fp4x2 = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b8 byte0, byte1, byte2, byte3;
        cvt.rn.satfinite.e2m1x2.f32 byte0, $5, $1;
        cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $2;
        cvt.rn.satfinite.e2m1x2.f32 byte2, $7, $3;
        cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $4;
        mov.b32 $0, {byte0, byte1, byte2, byte3};
        }
        """,
        constraints=(
            "=r,"  # output, $0
            "r,r,r,r,"  # lo nibble, $1-$4
            "r,r,r,r"  # hi nibble, $5-$8
        ),
        args=x_blocks.reshape(128, 32, 2).split(),
        dtype=tl.int8,
        is_pure=True,
        pack=4,
    )  # (128, 32)
    offs_m = pid_m * 128 + tl.arange(0, 128)[:, None]
    offs_n = pid_n * 32 + tl.arange(0, 32)[None, :]
    tl.store(q_ptr + offs_m * (N // 2) + offs_n, x_fp4x2)


def quantize_nvfp4_triton(x: Tensor, tensor_scale: Tensor):
    M, N = x.shape
    assert M % 128 == 0 and N % 64 == 0
    xq = x.new_empty(M, N // 2, dtype=torch.int8)
    scales = x.new_empty(M, N // 16, dtype=torch.float8_e4m3fn)

    grid = (N // 64, M // 128)
    quantize_nvfp4_triton_kernel[grid](x, tensor_scale, xq, scales, x.stride(0), x.stride(1), N)

    return xq.view(FP4_DTYPE), scales
