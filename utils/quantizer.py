import torch
import torch.nn as nn

CLIPMIN = 1e-5


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits=8,
        symmetric=False,
        group_size=None,
    ):
        super().__init__()

        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1

    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            x = x.reshape(-1, self.group_size)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax =  x.amax(reduce_shape, keepdim=True)
        if self.symmetric:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = abs_max / (2 ** (self.n_bits - 1) - 1)
            scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2 ** (self.n_bits - 1) - 1) * torch.ones_like(scale)
        else:
            range = xmax - xmin
            scale = range / (2 ** self.n_bits - 1)
            scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (scale)
        zero_point = zero_point.clamp(min=-1e4, max=1e4).round()

        return scale, zero_point
    
    def fake_quant(self, x, scale, zero_point):        
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        # quantize
        x_int = round_ste(x / scale)
        if zero_point is not None:
            x_int = x_int.add(zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        # de-quantize
        x_dequant = x_int
        if zero_point is not None:
            x_dequant = x_dequant.sub(zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        
        return x_dequant
    
    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16:
            return x

        scale, zero_point = self.per_token_dynamic_calibration(x)  
        x_dequant = self.fake_quant(x, scale, zero_point)

        return x_dequant
