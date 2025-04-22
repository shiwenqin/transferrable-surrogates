from math import floor, log, sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from positional_encodings.torch_encodings import (
    PositionalEncodingPermute1D,
    PositionalEncodingPermute2D,
    Summer,
)
from torch.fft import fftn, ifftn


def pair(x):
    if isinstance(x, tuple) or isinstance(x, list) and len(x) == 2:
        return x
    elif isinstance(x, int):
        return (x, x)


class Lambda(nn.Module):
    """Module that applies a lambda function to the input."""

    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

    def __repr__(self):
        return f"Lambda({self.lambd})"


nn.Lambda = Lambda


class SequentialModule(nn.Module):
    """
    Module that applies a sequence of functions to the input.

    Attributes
    ----------
    first_fn : nn.Module
        The first function that is applied to the input
    second_fn : nn.Module
        The second function that is applied to the output of the first function
    """

    def __init__(self, first_fn, second_fn):
        super(SequentialModule, self).__init__()
        self.first_fn = first_fn
        self.second_fn = second_fn

    def forward(self, x):
        out = self.first_fn(x)
        out = self.second_fn(out)
        return out


class SequentialModule4(nn.Module):
    """
    Module that applies a sequence of functions to the input.

    Attributes
    ----------
    fns : nn.Module
        The function that is applied to the input four times
    """

    def __init__(self, fns):
        super(SequentialModule4, self).__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        for fn in self.fns:
            x = fn(x)
        return x


class SequentialModule8(nn.Module):
    """
    Module that applies a sequence of functions to the input.

    Attributes
    ----------
    fns : nn.Module
        The function that is applied to the input eight times
    """

    def __init__(self, fns):
        super(SequentialModule8, self).__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        for fn in self.fns:
            x = fn(x)
        return x


class BranchingModule(nn.Module):
    """
    Module that branches the input and then aggregates the outputs.

    Attributes
    ----------
    branching_fn : nn.Module
        The function that branches the input (only supports a branching factor of 2 right now)
    inner_fn : nn.Module
        The list of functions that is applied to each branch, respectively (each branch is a separate module)
    aggregation_fn : nn.Module
        The function that aggregates the outputs of the inner functions (only supports a branching factor of 2 right now)
    """

    def __init__(self, branching_fn, inner_fn, aggregation_fn):
        super(BranchingModule, self).__init__()
        self.branching_fn = branching_fn
        self.inner_fn = nn.ModuleList(inner_fn)
        self.aggregation_fn = aggregation_fn

    def forward(self, x):
        branching_outs = list(self.branching_fn(x))
        # print("BranchingModule branching_outs", [b.shape for b in branching_outs])
        inner_outs = []
        for i in range(len(branching_outs)):
            inner_out = self.inner_fn[i](branching_outs[i])
            inner_outs.append(inner_out)
        # print("BranchingModule inner_outs", [i.shape for i in inner_outs])
        aggregation_out = self.aggregation_fn(inner_outs)
        # print("BranchingModule aggregation_out", aggregation_out.shape)
        return aggregation_out


class RoutingModule(nn.Module):
    """
    Module that applies a sequence of functions to the input.

    Attributes
    ----------
    prerouting_fn : nn.Module
        The function that is applied before the computation function, rearranges/permutes the input
    computation_fn : nn.Module
        The module that processes the output of the prerouting function
    postrouting_fn : nn.Module
        The function that is applied after the computation function, rearranges/permutes the output
    """

    def __init__(
        self,
        prerouting_fn,
        inner_fn,
        postrouting_fn,
    ):
        super(RoutingModule, self).__init__()
        self.prerouting_fn = prerouting_fn
        self.inner_fn = inner_fn
        self.postrouting_fn = postrouting_fn
        if hasattr(self.prerouting_fn, "fold_output_shape"):
            self.postrouting_fn.output_shape = (
                self.prerouting_fn.fold_output_shape
            )

    def forward(self, x):
        # make sure postrouting functions can undo any prerouting changes
        # so e.g. col2im can reshape things based on the original shape before im2col
        # out = self.prerouting_fn(x)
        # out = self.inner_fn(out)
        # out = self.postrouting_fn(out)
        # return out
        return self.postrouting_fn(self.inner_fn(self.prerouting_fn(x)))


class ComputationModule(nn.Module):
    """
    Module that applies a sequence of functions to the input.

    Attributes
    ----------
    computation_fn : nn.Module
        The function that is applied to the input, e.g. a linear layer or a normalization layer
    """

    def __init__(
        self,
        computation_fn,
    ):
        super(ComputationModule, self).__init__()
        self.computation_fn = computation_fn

    def forward(self, x):
        out = self.computation_fn(x)
        return out


class CloneTensor(nn.Module):
    """Clone a tensor a given number of times."""

    def __init__(self, num_clones, **kwargs):
        super(CloneTensor, self).__init__()
        self.n = num_clones

    def forward(self, x):
        return (torch.clone(x) for _ in range(self.n))

    def __repr__(self):
        return f"CloneTensor(n={self.n})"


class GroupDim(nn.Module):
    """Group a tensor along a given dimension. E.g split a tensor along the second dimension into 2 groups of equal size."""

    def __init__(self, splits, dim, dim_total, **kwargs):  # dim in [1, 2, 3]
        super(GroupDim, self).__init__()
        self.sections = [dim_total // splits] * splits
        self.n = splits
        self.dim = dim

    def forward(self, x):
        return torch.split(x, self.sections, dim=self.dim)

    def __repr__(self):
        return f"GroupDim(splits={self.n}, dim={self.dim})"


# im2col style unfolding of input image
class Im2Col(nn.Module):
    """
    Rearrange the dimensions of a tensor to form a matrix.
    This is the inverse of the Col2Im class.
    It converts from the "im" mode to the "col" mode.
    """

    def __init__(
        self,
        input_shape,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        **kwargs,
    ):
        super(Im2Col, self).__init__()
        batch, channels, height, width = input_shape
        self.kernel_size, self.stride, self.padding, self.dilation = (
            pair(kernel_size),
            pair(stride),
            pair(padding),
            pair(dilation),
        )
        # self.prearrange = Rearrange(
        #     "batch (groups in_channels_divided_by_groups) width_num_patches width_patches -> (batch groups) in_channels_divided_by_groups width_num_patches width_patches",
        #     groups=groups,
        # )
        # equivalent using permute and reshape operations
        self.prearrange = lambda x: x
        # self.postarrange = Rearrange(
        #     "(batch groups) kernel_squared_times_in_channels_divided_by_groups patch_size -> batch (groups patch_size) kernel_squared_times_in_channels_divided_by_groups",
        #     groups=groups,
        # )
        # equivalent using permute and reshape operations
        self.postarrange = lambda x: x.permute(0, 2, 1)
        self.unfold = nn.Unfold(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        self.fold_output_shape = (
            floor(
                (
                    (
                        height
                        + 2 * self.padding[0]
                        - self.dilation[0] * (self.kernel_size[0] - 1)
                        - 1
                    )
                    / self.stride[0]
                )
                + 1
            ),
            floor(
                (
                    (
                        width
                        + 2 * self.padding[1]
                        - self.dilation[1] * (self.kernel_size[1] - 1)
                        - 1
                    )
                    / self.stride[1]
                )
                + 1
            ),
        )
        # print("Im2Col input_shape", input_shape)
        # print("Im2Col fold_output_shape", self.fold_output_shape)

    def forward(self, x):
        # print("im2col parameters", self.kernel_size, self.stride, self.padding, self.dilation)
        # print("im2col input_shape", x.shape)
        # x = self.prearrange(x)
        # print("im2col prearrange", x.shape)
        # x = self.unfold(x)
        # print("im2col unfold", x.shape)
        # x = self.postarrange(x)
        # print("im2col postarrange", x.shape)
        # return x
        return self.postarrange(self.unfold(self.prearrange(x)))

    def __repr__(self):
        return f"Im2Col(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


# im2col style unfolding of input image
class Im2Col1d(nn.Module):
    """
    Rearrange the dimensions of a tensor to form a matrix.
    This is the inverse of the Col2Im class.
    It converts from the "im" mode to the "col" mode.
    """

    def __init__(
        self,
        input_shape,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        **kwargs,
    ):
        super(Im2Col1d, self).__init__()
        batch, channels, dim = input_shape
        self.kernel_size, self.stride, self.padding, self.dilation = kernel_size, stride, padding, dilation
        # equivalent using permute and reshape operations
        self.prearrange = lambda x: x
        # equivalent using permute and reshape operations
        self.postarrange = lambda x: x.permute(0, 2, 1)
        self.unfold = nn.Unfold(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        self.fold_output_shape = (
            floor(
                (
                    (
                        height
                        + 2 * self.padding[0]
                        - self.dilation[0] * (self.kernel_size[0] - 1)
                        - 1
                    )
                    / self.stride[0]
                )
                + 1
            ),
            floor(
                (
                    (
                        width
                        + 2 * self.padding[1]
                        - self.dilation[1] * (self.kernel_size[1] - 1)
                        - 1
                    )
                    / self.stride[1]
                )
                + 1
            ),
        )
        # print("Im2Col input_shape", input_shape)
        # print("Im2Col fold_output_shape", self.fold_output_shape)

    def forward(self, x):
        # print("im2col parameters", self.kernel_size, self.stride, self.padding, self.dilation)
        # print("im2col input_shape", x.shape)
        # x = self.prearrange(x)
        # print("im2col prearrange", x.shape)
        # x = self.unfold(x)
        # print("im2col unfold", x.shape)
        # x = self.postarrange(x)
        # print("im2col postarrange", x.shape)
        # return x
        return self.postarrange(self.unfold(self.prearrange(x)))

    def __repr__(self):
        return f"Im2Col(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation})"


class Col2Im(nn.Module):
    """
    Rearrange the dimensions of a tensor to form an image.
    This is the inverse of the Im2Col class.
    It converts form the "col" mode to the "im" mode.
    """

    def __init__(self, **kwargs):
        super(Col2Im, self).__init__()
        # self.rearrange = lambda x: Rearrange(
        #     "batch (groups h w) c -> batch (groups c) h w",
        #     h=self.output_shape[0],
        #     w=self.output_shape[1],
        # )(x)
        # equivalent using permute and reshape
        self.rearrange = lambda x: x.permute(0, 2, 1).reshape(x.shape[0], -1, self.output_shape[0], self.output_shape[1])

    def forward(self, x):
        # print("col2im output_shape", self.output_shape)
        # print("col2im input_shape", x.shape)
        x = self.rearrange(x)
        # print("col2im rearrange", x.shape)
        return x

    def __repr__(self):
        if hasattr(self, "output_shape"):
            return f"Col2Im(output_shape={self.output_shape})"
        else:
            return f"Col2Im()"


class Col2Im2(nn.Module):
    """
    Rearrange the dimensions of a tensor to form an image.
    This is the inverse of the Im2Col class.
    It converts form the "col" mode to the "im" mode.
    """

    def __init__(
        self, kernel_size, stride=1, padding=0, dilation=1, groups=1, **kwargs
    ):
        super(Col2Im2, self).__init__()
        self.kernel_size, self.stride, self.padding, self.dilation = (
            pair(kernel_size),
            pair(stride),
            pair(padding),
            pair(dilation),
        )
        self.rearrange = lambda x: Rearrange(
            "batch (groups h w) c -> batch (groups c) h w",
            h=self.output_shape[0],
            w=self.output_shape[1],
        )(x)

    def forward(self, x):
        x = self.rearrange(x)
        return x

    def __repr__(self):
        try:
            return f"Col2Im(output_shape={self.output_shape}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation})"
        except AttributeError:
            return f"Col2Im(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation})"


# col2im style unfolding of input image
class Col2Im3(nn.Module):
    """
    Rearrange the dimensions of a tensor to form an image.
    This is the inverse of the Im2Col class.
    It converts form the "col" mode to the "im" mode.
    """

    def __init__(
        self, kernel_size, stride=1, padding=0, dilation=1, groups=1, **kwargs
    ):
        super(Col2Im3, self).__init__()
        self.kernel_size, self.stride, self.padding, self.dilation = (
            pair(kernel_size),
            pair(stride),
            pair(padding),
            pair(dilation),
        )
        self.prearrange = Rearrange(
            "batch (groups patch_size) kernel_squared_times_in_channels_divided_by_groups -> (batch groups) kernel_squared_times_in_channels_divided_by_groups patch_size",
            groups=groups,
        )
        self.postarrange = Rearrange(
            "(batch groups) in_channels_divided_by_groups width_num_patches width_patches -> batch (groups in_channels_divided_by_groups) width_num_patches width_patches",
            groups=groups,
        )
        self.fold = lambda x: nn.Fold(
            output_size=self.output_shape,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )(x)

    def forward(self, x):
        # ("col2im parameters", self.output_shape, self.kernel_size, self.stride, self.padding, self.dilation)
        # print("col2im input_shape", x.shape)
        x = self.prearrange(x)
        # print("col2im prearrange", x.shape)
        x = self.fold(x)
        # print("col2im fold", x.shape)
        x = self.postarrange(x)
        # print("col2im postarrange", x.shape)
        return x


class Permute(nn.Module):
    """Permute the dimensions of a tensor."""

    def __init__(self, dims, **kwargs):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)

    def __repr__(self):
        return f"Permute(dims={self.dims})"


class DotProduct(nn.Module):
    """Dot product of two tensors with optional scaling."""

    def __init__(self, scaled=False, **kwargs):
        super(DotProduct, self).__init__()
        self.scaled = scaled

    def forward(self, tensors):
        a, b = tensors
        scale_factor = 1.0 / sqrt(a.size(-1)) if self.scaled else 1.0
        if (a.dim() == 2) and (b.dim() == 2):
            a, b = a.unsqueeze(1), b.unsqueeze(-1)
            return (a @ b * scale_factor).squeeze(-1)
        else:
            return a @ b * scale_factor

    def __repr__(self):
        return f"DotProduct(scaled={self.scaled})"


class AddTensors(nn.Module):
    """Add tensors together."""

    def __init__(self, **kwargs):
        super(AddTensors, self).__init__()

    def forward(self, tensors):
        out = torch.stack(tensors)
        out = out.sum(0)
        return out


class CatTensors(nn.Module):
    """Concatenate tensors along a specified dimension."""

    def __init__(self, dim, **kwargs):
        super(CatTensors, self).__init__()
        self.dim = dim

    def forward(self, tensors):
        # print("CatTensors input_shape", [t.shape for t in tensors])
        out= torch.cat(tensors, dim=self.dim)
        # print("CatTensors output_shape", out.shape)
        return out

    def __repr__(self):
        return f"CatTensors(dim={self.dim})"


class BroadcastTensors(nn.Module):
    """
    Merge two tensors in the most general way to a common output shape.
    This is useful for broadcasting two tensors of different shapes into a single output.
    This works with any two tensors that share a common prefix in their shapes and have 3 or 4 dimensions.
    """

    def __init__(self, mode="add", **kwargs):
        super(BroadcastTensors, self).__init__()
        self.mode = mode

    def forward(self, tensors):
        """
        Align and merge two tensors using the specified mode: 'add', 'cat', or 'matmul'.
        
        Parameters:
        - tensors: tuple of two tensors (a, b) to be merged.
        
        Returns:
        - Merged tensor based on the mode.
        """
        a, b = tensors
        assert a.dim() in [3, 4] and b.dim() in [3, 4], "Only 3D and 4D tensors are supported"

        # Align dimensions by unsqueezing
        while a.dim() < b.dim():
            a = a.unsqueeze(1)
        while b.dim() < a.dim():
            b = b.unsqueeze(1)

        # if any dimensions match but are out of order, permute the dimensions of b to match a
        # print(f"Original shapes: {a.shape}, {b.shape}")
        a_dims, b_dims = a.dim(), b.dim()
        for i in range(1, a_dims):
            for j in range(1, b_dims):
                if a.shape[i] == b.shape[j] and a.shape[i] != b.shape[i] and a.shape[j] != b.shape[j]:
                    # swap i and j of b
                    dims = list(range(b.dim()))
                    dims[i], dims[j] = dims[j], dims[i]
                    b = b.permute(dims)
        # print(f"Permuted shapes: {a.shape}, {b.shape}")

        # if dimensions match, do nothing
        # if a dimension is 1, expand it to match the other tensor
        # if dimensions are different, reduce the dimensionality of the first tensor and then expand
        a_shape, b_shape = list(a.shape), list(b.shape)
        for i in range(1, min(a_dims, b_dims) + 1):
            if a_shape[-i] == b_shape[-i]: # match found, do nothing
                continue
            elif a_shape[-i] == 1: # expand a to match b
                a_shape[-i] = b_shape[-i]
                a = a.expand(a_shape)
            elif b_shape[-i] == 1:# expand b to match a
                b_shape[-i] = a_shape[-i]
                b = b.expand(b_shape)
            else: # reduce the dimensionality of a and then expand to match b
                b = b.mean(dim=-i, keepdim=True)
                b_shape[-i] = a_shape[-i]
                b = b.expand(b_shape)
        if self.mode == "add":
            return a + b
        elif self.mode == "cat":
            return torch.cat([a, b], dim=-1)
        elif self.mode == "matmul":
            # swap the final two dims
            b = b.permute(list(range(b.dim() - 2)) + [-1, -2])
            return torch.matmul(a, b)

    def __repr__(self):
        return f"BroadcastTensors(mode={self.mode})"


class EinLinear(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        """Linear layer with input and output dimensions."""
        super(EinLinear, self).__init__()
        self.fn = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # print("EinLinear input_shape", x.shape)
        # print("EinLinear weight_shape", self.fn.weight.shape)
        out = self.fn(x)
        # print("EinLinear output_shape", out.shape)
        return out


class FFTLayer(nn.Module):
    """
    A class that uses the pytorch FFT function to compute the FFT of an image
    The input is an image in the [B, C, H, W] or [B, C, L] format
    The output is a tensor containing the real and imaginary parts of the FFT
    """

    def __init__(self, **kwargs):
        super(FFTLayer, self).__init__()

    def forward(self, x):
        if x.dim() == 4:
            fft = fftn(x, dim=[2, 3])
        if x.dim() == 3:
            fft = fftn(x, dim=[2])
        out = torch.cat([fft.real, fft.imag], dim=-1)
        return out


class IFFTLayer(nn.Module):
    """
    A class that uses the pytorch IFFT function to compute the IFFT of an image
    The input is the real and imaginary parts of the FFT
    The output is the reconstructed image
    """

    def __init__(self, **kwargs):
        super(IFFTLayer, self).__init__()

    def forward(self, x):
        fft = torch.complex(
            x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
        )
        if x.dim() == 4:
            out = ifftn(fft, dim=[2, 3])
        if x.dim() == 3:
            out = ifftn(fft, dim=[2])
        return out


class EinNorm(nn.Module):
    """LayerNorm for 3D and BatchNorm for 4D."""

    def __init__(self, input_shape, **kwargs):
        super(EinNorm, self).__init__()
        if len(input_shape) == 2:
            self.fn = nn.LayerNorm(input_shape[-1])
        elif len(input_shape) == 3:
            self.fn = nn.BatchNorm1d(input_shape[1])
        elif len(input_shape) == 4:
            self.fn = nn.BatchNorm2d(input_shape[1])
        else:
            raise NotImplementedError(
                "Only shapes of (B, C, H, W) and (B, C, L) implemented."
            )

    def forward(self, x):
        return self.fn(x)


class PositionalEncoding(nn.Module):
    """Positional Encoding"""

    def __init__(self, input_shape):
        super().__init__()
        if len(input_shape) == 2:
            self.fn = Summer(
                PositionalEncodingPermute1D(channels=input_shape[1])
            )
            print("TESTING 1D POS-ENC!")
        elif len(input_shape) == 3:
            self.fn = Summer(
                PositionalEncodingPermute1D(channels=input_shape[1])
            )
        elif len(input_shape) == 4:
            self.fn = Summer(
                PositionalEncodingPermute2D(channels=input_shape[1])
            )

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]`` or ``[batch_size, channels, height, width]``
        """
        return self.fn(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable Positional Encoding"""

    def __init__(self, input_shape, **kwargs):
        super(LearnablePositionalEncoding, self).__init__()
        self.input_shape = input_shape
        if len(input_shape) == 3:
            self.fn = nn.Parameter(
                torch.randn(1, input_shape[1], input_shape[2])
            )
        elif len(input_shape) == 4:
            self.fn = nn.Parameter(
                torch.randn(1, input_shape[1], input_shape[2], input_shape[3])
            )
        else:
            raise NotImplementedError(
                "Only shapes of (B, C, H, W) and (B, C, L) implemented."
            )

    def forward(self, x):
        # print("LearnablePositionalEncoding input_shape", x.shape)
        return x + self.fn


########################################################################################
# Modules for hNASBench201
########################################################################################

class Sequential2(nn.Module):
    """
    A sequential module that takes in 2 inputs.
    """

    def __init__(self, first_fn, second_fn):
        super(Sequential2, self).__init__()
        self.first_fn = first_fn
        self.second_fn = second_fn

    def forward(self, x):
        out = self.first_fn(x)
        out = self.second_fn(out)
        return out

class Sequential3(nn.Module):
    """
    A sequential module that takes in 3 inputs.
    """

    def __init__(self, first_fn, second_fn, third_fn):
        super(Sequential3, self).__init__()
        self.first_fn = first_fn
        self.second_fn = second_fn
        self.third_fn = third_fn

    def forward(self, x):
        out = self.first_fn(x)
        out = self.second_fn(out)
        out = self.third_fn(out)
        return out


class Sequential4(nn.Module):
    """
    A sequential module that takes in 4 inputs.
    """

    def __init__(self, first_fn, second_fn, third_fn, fourth_fn):
        super(Sequential4, self).__init__()
        self.first_fn = first_fn
        self.second_fn = second_fn
        self.third_fn = third_fn
        self.fourth_fn = fourth_fn

    def forward(self, x):
        out = self.first_fn(x)
        out = self.second_fn(out)
        out = self.third_fn(out)
        out = self.fourth_fn(out)
        return out


class Residual2(nn.Module):
    """
    A residual module that takes in 2 inputs.
    """

    def __init__(self, first_fn, residual_fn, second_fn):
        super(Residual2, self).__init__()
        self.first_fn = first_fn
        self.residual_fn = residual_fn
        self.second_fn = second_fn

    def forward(self, x):
        out = self.first_fn(x)
        out = self.second_fn(out)
        residual = self.residual_fn(x)
        out = out + residual
        return out


class Residual3(nn.Module):
    """
    A residual module that takes in 3 inputs.
    """

    def __init__(self, first_fn, second_fn, residual_fn, third_fn):
        super(Residual3, self).__init__()
        self.first_fn = first_fn
        self.second_fn = second_fn
        self.residual_fn = residual_fn
        self.third_fn = third_fn

    def forward(self, x):
        out = self.first_fn(x)
        out = self.second_fn(out)
        out = self.third_fn(out)
        residual = self.residual_fn(x)
        out = out + residual
        return out


class Cell(nn.Module):
    """
    A NasBench201 cell module that has 6 input functions.
    """

    def __init__(self, a, b, c, d, e, f):
        super(Cell, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def forward(self, x):
        a_out = self.a(x)
        b_out = self.b(x)
        c_out = self.c(a_out)
        d_out = self.d(x)
        e_out = self.e(a_out)
        f_out = self.f(b_out + c_out)
        out = d_out + e_out + f_out
        return out


class Diamond2(nn.Module):
    """
    A diamond module that takes in 4 function inputs.
    """

    def __init__(self, a, b, c, d):
        super(Diamond2, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def forward(self, x):
        a_out = self.a(x)
        b_out = self.b(x)
        c_out = self.c(a_out)
        d_out = self.d(b_out)
        out = c_out + d_out
        return out


class Diamond3(nn.Module):
    """
    A diamond module that takes in 6 function inputs.
    """

    def __init__(self, a, b, c, d, e, f):
        super(Diamond3, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def forward(self, x):
        a_out = self.a(x)
        b_out = self.b(x)
        c_out = self.c(a_out)
        d_out = self.d(b_out)
        e_out = self.e(c_out)
        f_out = self.f(d_out)
        out = e_out + f_out
        return out


########################################################################################
# Functions for hNASBench301
########################################################################################

def sequential4_D1_D1_D0_D0(**kwargs):
    return Sequential4(**kwargs)


def sequential3_D1_D1_D0(**kwargs):
    return Sequential3(**kwargs)


def sequential3_D0_D1_D1(**kwargs):
    return Sequential3(**kwargs)


def sequential3_C_C_D(**kwargs):
    return Sequential3(**kwargs)


def sequential3_C_C_CL(**kwargs):
    return Sequential3(**kwargs)


def sequential2_CL_DOWN(**kwargs):
    return Sequential2(**kwargs)


def sequential2_CL_CL(**kwargs):
    return Sequential2(**kwargs)


def sequential3_ACT_CONV_NORM(**kwargs):
    return Sequential3(**kwargs)


def residual3_C_C_D_D(**kwargs):
    return Residual3(**kwargs)


def residual3_C_C_CL_CL(**kwargs):
    return Residual3(**kwargs)


def residual2_CL_DOWN_DOWN(**kwargs):
    return Residual2(**kwargs)


def residual2_CL_CL_CL(**kwargs):
    return Residual2(**kwargs)


def diamond3_C_C_C_C_D_D(**kwargs):
    return Diamond3(**kwargs)


def diamond3_C_C_C_C_CL_CL(**kwargs):
    return Diamond3(**kwargs)


def diamond2_CL_CL_DOWN_DOWN(**kwargs):
    return Diamond2(**kwargs)


def diamond2_CL_CL_CL_CL(**kwargs):
    return Diamond2(**kwargs)


def cell_OP_OP_OP_OP_OP_OP(**kwargs):
    return Cell(**kwargs)


class DownConv(nn.Module):
    """
    A downsampling convolutional layer.
    """

    def __init__(self, input_shape):
        super(DownConv, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[1],
            out_channels=input_shape[1] * 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=input_shape[1] * 2,
            out_channels=input_shape[1] * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(
            in_channels=input_shape[1],
            out_channels=input_shape[1] * 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        # separate out the two branches into multiple lines
        # if any dimension is odd, one branch will round down and the other will round up
        # so we need to pad the input to make sure the dimensions are even
        if x.shape[2] % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1), mode="constant", value=0)
        if x.shape[3] % 2 == 1:
            x = F.pad(x, (0, 1, 0, 0), mode="constant", value=0)
        out1 = self.conv2(self.conv1(x))
        out2 = self.conv3(self.avgpool(x))
        return out1 + out2


def down(**kwargs):
    return DownConv(input_shape=kwargs["input_shape"])


def zero(**kwargs):
    return nn.Lambda(lambda x: torch.zeros_like(x).to(x.device))


def identity(**kwargs):
    return nn.Identity()


def avg_pool(**kwargs):
    return nn.AvgPool2d(kernel_size=3, stride=1, padding=1)


def relu(**kwargs):
    return nn.ReLU()


def hardswish(**kwargs):
    return nn.Hardswish()


def mish(**kwargs):
    return nn.Mish()


def conv1x1(**kwargs):
    return nn.Conv2d(
        in_channels=kwargs["input_shape"][1],
        out_channels=kwargs["input_shape"][1],
        kernel_size=1,
        stride=1,
        padding=0,
    )


def conv3x3(**kwargs):
    return nn.Conv2d(
        in_channels=kwargs["input_shape"][1],
        out_channels=kwargs["input_shape"][1],
        kernel_size=3,
        stride=1,
        padding=1,
    )


def dconv3x3(**kwargs):
    return nn.Conv2d(
        in_channels=kwargs["input_shape"][1],
        out_channels=kwargs["input_shape"][1],
        kernel_size=3,
        stride=1,
        padding=1,
        groups=kwargs["input_shape"][1],
    )


def batchnorm(**kwargs):
    return nn.BatchNorm2d(kwargs["input_shape"][1])


def instancenorm(**kwargs):
    return nn.InstanceNorm2d(kwargs["input_shape"][1])


def layernorm(**kwargs):
    return nn.LayerNorm(kwargs["input_shape"][1:])

########################################################################################
# End of hNASBench301
########################################################################################


# the functions which make up the terminals of the search space
def sequential_module(**kwargs):
    return SequentialModule(**kwargs)


def branching_module(**kwargs):
    return BranchingModule(**kwargs)


def routing_module(**kwargs):
    return RoutingModule(**kwargs)


def computation_module(**kwargs):
    return ComputationModule(**kwargs)


def clone_tensor2(**kwargs):
    return CloneTensor(num_clones=2, **kwargs)


def clone_tensor4(**kwargs):
    return CloneTensor(num_clones=4, **kwargs)


def clone_tensor8(**kwargs):
    return CloneTensor(num_clones=8, **kwargs)


def group_dim2s1d(**kwargs):
    return GroupDim(
        splits=2, dim=1, dim_total=kwargs["input_shape"][1], **kwargs
    )


def group_dim2s2d(**kwargs):
    return GroupDim(
        splits=2, dim=2, dim_total=kwargs["input_shape"][2], **kwargs
    )


def group_dim2s3d(**kwargs):
    return GroupDim(
        splits=2, dim=3, dim_total=kwargs["input_shape"][3], **kwargs
    )


def group_dim4s1d(**kwargs):
    return GroupDim(
        splits=4, dim=1, dim_total=kwargs["input_shape"][1], **kwargs
    )


def group_dim4s2d(**kwargs):
    return GroupDim(
        splits=4, dim=2, dim_total=kwargs["input_shape"][2], **kwargs
    )


def group_dim4s3d(**kwargs):
    return GroupDim(
        splits=4, dim=3, dim_total=kwargs["input_shape"][3], **kwargs
    )


def group_dim8s1d(**kwargs):
    return GroupDim(
        splits=8, dim=1, dim_total=kwargs["input_shape"][1], **kwargs
    )


def group_dim8s2d(**kwargs):
    return GroupDim(
        splits=8, dim=2, dim_total=kwargs["input_shape"][2], **kwargs
    )


def group_dim8s3d(**kwargs):
    return GroupDim(
        splits=8, dim=3, dim_total=kwargs["input_shape"][3], **kwargs
    )


def im2col1k1s0p(**kwargs):
    return Im2Col(kernel_size=1, stride=1, padding=0, **kwargs)


def im2col1k2s0p(**kwargs):
    return Im2Col(kernel_size=1, stride=2, padding=0, **kwargs)


def im2col3k1s1p(**kwargs):
    return Im2Col(kernel_size=3, stride=1, padding=1, **kwargs)


def im2col3k2s1p(**kwargs):
    return Im2Col(kernel_size=3, stride=2, padding=1, **kwargs)


def im2col5k1s2p(**kwargs):
    return Im2Col(kernel_size=5, stride=1, padding=2, **kwargs)


def im2col7k1s3p(**kwargs):
    return Im2Col(kernel_size=7, stride=1, padding=3, **kwargs)


def im2col7k2s3p(**kwargs):
    return Im2Col(kernel_size=7, stride=2, padding=3, **kwargs)


def im2col4k4s0p(**kwargs):
    return Im2Col(kernel_size=4, stride=4, padding=0, **kwargs)


def im2col8k8s0p(**kwargs):
    return Im2Col(kernel_size=8, stride=8, padding=0, **kwargs)


def im2col16k16s0p(**kwargs):
    return Im2Col(kernel_size=16, stride=16, padding=0, **kwargs)


def col2im(**kwargs):
    return Col2Im(**kwargs)


def col2im1k1s0p(**kwargs):
    return Col2Im(kernel_size=1, stride=1, padding=0, **kwargs)


def col2im1k2s0p(**kwargs):
    return Col2Im(kernel_size=1, stride=2, padding=0, **kwargs)


def col2im3k1s1p(**kwargs):
    return Col2Im(kernel_size=3, stride=1, padding=1, **kwargs)


def col2im3k2s1p(**kwargs):
    return Col2Im(kernel_size=3, stride=2, padding=1, **kwargs)


def col2im5k1s2p(**kwargs):
    return Col2Im(kernel_size=5, stride=1, padding=2, **kwargs)


def col2im7k1s3p(**kwargs):
    return Col2Im(kernel_size=7, stride=1, padding=3, **kwargs)


def col2im7k2s3p(**kwargs):
    return Col2Im(kernel_size=7, stride=2, padding=3, **kwargs)


def col2im4k4s0p(**kwargs):
    return Col2Im(kernel_size=4, stride=4, padding=0, **kwargs)


def col2im8k8s0p(**kwargs):
    return Col2Im(kernel_size=8, stride=8, padding=0, **kwargs)


def col2im16k16s0p(**kwargs):
    return Col2Im(kernel_size=16, stride=16, padding=0, **kwargs)


def permute132(**kwargs):
    return Permute(dims=(0, 1, 3, 2), **kwargs)


def permute213(**kwargs):
    return Permute(dims=(0, 2, 1, 3), **kwargs)


def permute231(**kwargs):
    return Permute(dims=(0, 2, 3, 1), **kwargs)


def permute312(**kwargs):
    return Permute(dims=(0, 3, 1, 2), **kwargs)


def permute321(**kwargs):
    return Permute(dims=(0, 3, 2, 1), **kwargs)


def permute21(**kwargs):
    return Permute(dims=(0, 2, 1), **kwargs)


def linear16(**kwargs):
    return EinLinear(in_dim=kwargs["input_shape"][-1], out_dim=16, **kwargs)


def linear32(**kwargs):
    return EinLinear(in_dim=kwargs["input_shape"][-1], out_dim=32, **kwargs)


def linear64(**kwargs):
    return EinLinear(in_dim=kwargs["input_shape"][-1], out_dim=64, **kwargs)


def linear128(**kwargs):
    return EinLinear(in_dim=kwargs["input_shape"][-1], out_dim=128, **kwargs)


def linear256(**kwargs):
    return EinLinear(in_dim=kwargs["input_shape"][-1], out_dim=256, **kwargs)


def linear512(**kwargs):
    return EinLinear(in_dim=kwargs["input_shape"][-1], out_dim=512, **kwargs)


def linear1024(**kwargs):
    return EinLinear(in_dim=kwargs["input_shape"][-1], out_dim=1024, **kwargs)


def linear2048(**kwargs):
    return EinLinear(in_dim=kwargs["input_shape"][-1], out_dim=2048, **kwargs)


def linear_same(**kwargs):
    return EinLinear(
        in_dim=kwargs["input_shape"][-1],
        out_dim=kwargs["input_shape"][-1],
        **kwargs,
    )


def linear_half(**kwargs):
    return EinLinear(
        in_dim=kwargs["input_shape"][-1],
        out_dim=kwargs["input_shape"][-1] // 2,
        **kwargs,
    )


def linear_double(**kwargs):
    return EinLinear(
        in_dim=kwargs["input_shape"][-1],
        out_dim=kwargs["input_shape"][-1] * 2,
        **kwargs,
    )


def linear_4th(**kwargs):
    return EinLinear(
        in_dim=kwargs["input_shape"][-1],
        out_dim=kwargs["input_shape"][-1] // 4,
        **kwargs,
    )


def linear_x4(**kwargs):
    return EinLinear(
        in_dim=kwargs["input_shape"][-1],
        out_dim=kwargs["input_shape"][-1] * 4,
        **kwargs,
    )


def linear_8th(**kwargs):
    return EinLinear(
        in_dim=kwargs["input_shape"][-1],
        out_dim=kwargs["input_shape"][-1] // 8,
        **kwargs,
    )


def linear_x8(**kwargs):
    return EinLinear(
        in_dim=kwargs["input_shape"][-1],
        out_dim=kwargs["input_shape"][-1] * 8,
        **kwargs,
    )


def conv1d1k1s0p32d(**kwargs):
    return nn.Conv1d(
        in_channels=kwargs["input_shape"][1],
        out_channels=32,
        kernel_size=1,
        stride=1,
        padding=0,
    )


def conv1d1k1s0p64d(**kwargs):
    return nn.Conv1d(
        in_channels=kwargs["input_shape"][1],
        out_channels=64,
        kernel_size=1,
        stride=1,
        padding=0,
    )


def conv1d1k1s0p128d(**kwargs):
    return nn.Conv1d(
        in_channels=kwargs["input_shape"][1],
        out_channels=128,
        kernel_size=1,
        stride=1,
        padding=0,
    )


def conv1d1k1s0p256d(**kwargs):
    return nn.Conv1d(
        in_channels=kwargs["input_shape"][1],
        out_channels=256,
        kernel_size=1,
        stride=1,
        padding=0,
    )


def conv1d3k1s1p32d(**kwargs):
    return nn.Conv1d(
        in_channels=kwargs["input_shape"][1],
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=1,
    )


def conv1d3k1s1p64d(**kwargs):
    return nn.Conv1d(
        in_channels=kwargs["input_shape"][1],
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
    )


def conv1d3k1s1p128d(**kwargs):
    return nn.Conv1d(
        in_channels=kwargs["input_shape"][1],
        out_channels=128,
        kernel_size=3,
        stride=1,
        padding=1,
    )


def conv1d3k1s1p256d(**kwargs):
    return nn.Conv1d(
        in_channels=kwargs["input_shape"][1],
        out_channels=256,
        kernel_size=3,
        stride=1,
        padding=1,
    )


def conv1d5k1s2p32d(**kwargs):
    return nn.Conv1d(
        in_channels=kwargs["input_shape"][1],
        out_channels=32,
        kernel_size=5,
        stride=1,
        padding=2,
    )


def conv1d5k1s2p64d(**kwargs):
    return nn.Conv1d(
        in_channels=kwargs["input_shape"][1],
        out_channels=64,
        kernel_size=5,
        stride=1,
        padding=2,
    )


def conv1d5k1s2p128d(**kwargs):
    return nn.Conv1d(
        in_channels=kwargs["input_shape"][1],
        out_channels=128,
        kernel_size=5,
        stride=1,
        padding=2,
    )


def conv1d5k1s2p256d(**kwargs):
    return nn.Conv1d(
        in_channels=kwargs["input_shape"][1],
        out_channels=256,
        kernel_size=5,
        stride=1,
        padding=2,
    )


def conv1d8k1s3p32d(**kwargs):
    return nn.Sequential(
        nn.Lambda(lambda x: F.pad(x, (0, 1))),
        nn.Conv1d(
            in_channels=kwargs["input_shape"][1],
            out_channels=32,
            kernel_size=8,
            stride=1,
            padding=3,
        )
    )


def conv1d8k1s3p64d(**kwargs):
    return nn.Sequential(
        nn.Lambda(lambda x: F.pad(x, (0, 1))),
        nn.Conv1d(
            in_channels=kwargs["input_shape"][1],
            out_channels=64,
            kernel_size=8,
            stride=1,
            padding=3,
        )
    )


def conv1d8k1s3p128d(**kwargs):
    return nn.Sequential(
        nn.Lambda(lambda x: F.pad(x, (0, 1))),
        nn.Conv1d(
            in_channels=kwargs["input_shape"][1],
            out_channels=128,
            kernel_size=8,
            stride=1,
            padding=3,
        )
    )


def conv1d8k1s3p256d(**kwargs):
    return nn.Sequential(
        nn.Lambda(lambda x: F.pad(x, (0, 1))),
        nn.Conv1d(
            in_channels=kwargs["input_shape"][1],
            out_channels=256,
            kernel_size=8,
            stride=1,
            padding=3,
        )
    )


def norm(**kwargs):
    return EinNorm(**kwargs)


def leakyrelu(**kwargs):
    return nn.LeakyReLU()


def softmax(**kwargs):
    return nn.Softmax(dim=-1)


def positional_encoding(**kwargs):
    return PositionalEncoding(input_shape=kwargs["input_shape"])


def learnable_positional_encoding(**kwargs):
    return LearnablePositionalEncoding(input_shape=kwargs["input_shape"])


def identity(**kwargs):
    return nn.Identity()


def dot_product(**kwargs):
    return DotProduct(scaled=False, **kwargs)


def scaled_dot_product(**kwargs):
    return DotProduct(scaled=True, **kwargs)


def add_tensors(**kwargs):
    return AddTensors(**kwargs)


def cat_tensors1d2t(**kwargs):
    return CatTensors(dim=1, **kwargs)


def cat_tensors2d2t(**kwargs):
    return CatTensors(dim=2, **kwargs)


def cat_tensors3d2t(**kwargs):
    return CatTensors(dim=3, **kwargs)


def cat_tensors1d4t(**kwargs):
    return CatTensors(dim=1, **kwargs)


def cat_tensors2d4t(**kwargs):
    return CatTensors(dim=2, **kwargs)


def cat_tensors3d4t(**kwargs):
    return CatTensors(dim=3, **kwargs)


def cat_tensors1d8t(**kwargs):
    return CatTensors(dim=1, **kwargs)


def cat_tensors2d8t(**kwargs):
    return CatTensors(dim=2, **kwargs)


def cat_tensors3d8t(**kwargs):
    return CatTensors(dim=3, **kwargs)


def maxpool3k2s1p(**kwargs):
    return nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


def adaptiveavgpool(**kwargs):
    return nn.AdaptiveAvgPool2d((1, 1))


if __name__ == "__main__":
    input_shape = (1, 16, 64)
    x = torch.randn(input_shape)
    col2im = col2im1k1s0p()
    col2im.output_shape = int(sqrt(input_shape[-2])), int(
        sqrt(input_shape[-2])
    )
    out = col2im(x)
    print(out.shape)
    im2col = im2col1k1s0p(**{"input_shape": out.shape})
    x = im2col(out)
    print(x.shape)
