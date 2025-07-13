import warnings
from math import gcd, log2

import torch.nn as nn

from .utils.channel_count import scn
from .utils.layer_init import conv_init, dense_init


class ConvBnAct(nn.Module):
    """
    Combines convolution, batch normalization and activation into a single module.

    Args:
        activation_fn: Activation function to use after batch norm. Defaults to Mish.
        **conv_args: Keyword arguments passed to nn.Conv2d constructor.
    """

    def __init__(self, activation_fn=nn.Mish, **conv_args) -> None:
        super(ConvBnAct, self).__init__()

        if "bias" not in conv_args:
            conv_args["bias"] = False

        if "activation" in conv_args:
            raise ValueError(
                "Activation function should be passed as a separate argument, You should use: activation_fn"
            )

        self.conv = nn.Conv2d(**conv_args)
        self.bn = nn.BatchNorm2d(conv_args["out_channels"])
        self.activation = activation_fn()

        self.init_weights()

    def init_weights(self):
        conv_init(self.conv)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class DepthwiseConv(nn.Module):
    """
    Combines depthwise convolution, batch normalization and activation into a single module.

    Args:
        activation_fn: Activation function to use after batch norm. Defaults to Mish.
        **conv_args: Keyword arguments passed to nn.Conv2d constructor.
    """

    def __init__(self, activation_fn=nn.Mish, **conv_args):
        super(DepthwiseConv, self).__init__()

        conv_args["groups"] = gcd(conv_args["in_channels"], conv_args["out_channels"])
        if conv_args["groups"] == 1:
            warnings.warn(
                "DepthwiseConv groups == 1, Could not find a large dividable group size!"
            )

        self.depthwise = ConvBnAct(activation_fn=activation_fn, **conv_args)

    def forward(self, x):
        return self.depthwise(x)


class SEUnit(nn.Module):
    """
    Squeeze-and-Excitation Unit.

    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int): Reduction ratio for the hidden dimension.
        act1 (nn.Module): Activation function for the first fully connected layer.
        act2 (nn.Module): Activation function for the second fully connected layer.
    """

    def __init__(
        self,
        in_channels,
        reduction_ratio=4,
        act1=nn.PReLU,
        act2=nn.Sigmoid,
    ):
        super(SEUnit, self).__init__()

        hidden_dim = scn(in_channels / reduction_ratio, divisible=8)

        if hidden_dim < 1:
            raise ValueError("in_channels too small")

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True)
        if act1 is nn.PReLU:
            self.act1 = act1(hidden_dim)
        else:
            self.act1 = act1()

        self.fc2 = nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=True)
        self.act2 = act2()

        self.init_weights()

    def init_weights(self):
        dense_init(self.fc1)
        dense_init(self.fc2)

    def forward(self, x):
        se_weight = self.avg_pool(x)

        se_weight = self.fc1(se_weight)
        se_weight = self.act1(se_weight)

        se_weight = self.fc2(se_weight)
        se_weight = self.act2(se_weight)

        return x * se_weight


class ECAUnit(nn.Module):
    """
    Efficient Channel Attention (ECA) Unit.

    Args:
        in_channels (int): Number of channels in the input feature map.
        gamma (int, optional): Scaling factor for kernel size calculation. Default: 2.
        b (int, optional): Offset for kernel size calculation. Default: 1.

    """

    def __init__(self, in_channels, gamma=2, b=1):
        super(ECAUnit, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        t = int((log2(in_channels) + b) / gamma)
        kernel_size = t if t % 2 else t + 1

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(1, 2)
        y = self.conv(y)
        y = y.transpose(1, 2).unsqueeze(-1)

        y = self.sigmoid(y)

        return x * y.expand_as(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution with optional channel attention.

    Args:
        activation_pw_fn (nn.Module, optional): Activation for pointwise convolution.
            Default: nn.PReLU.
        activation_dw_fn (nn.Module, optional): Activation for depthwise convolution.
            Default: nn.Mish.
        channel_attention (str): Type of channel attention to apply. Options:
            - "none": No attention
            - "eca": Efficient Channel Attention
            - "se": Squeeze-and-Excitation
            Default: "eca".
        channel_attention_kwargs (dict): Additional kwargs for attention module.
        **conv_args: Arguments passed to underlying convolution layers (e.g., in_channels,
            out_channels, kernel_size, etc.).

    """

    def __init__(
        self,
        activation_pw_fn=nn.PReLU,
        activation_dw_fn=nn.Mish,
        channel_attention="eca",
        channel_attention_kwargs={},
        **conv_args,
    ) -> None:
        super(DepthwiseSeparableConv, self).__init__()

        if channel_attention not in ["none", "eca", "se"]:
            raise ValueError(f"Invalid channel attention type: {channel_attention}")

        self.use_attention = channel_attention in ["eca", "se"]

        match channel_attention:
            case "eca":
                self.attention = ECAUnit(
                    conv_args["in_channels"], **channel_attention_kwargs
                )
            case "se":
                self.attention = SEUnit(
                    conv_args["in_channels"], **channel_attention_kwargs
                )

        self.pointwise = ConvBnAct(
            activation_fn=activation_pw_fn,  # type: ignore
            in_channels=conv_args["in_channels"],
            out_channels=conv_args["out_channels"],
            kernel_size=1,
            padding="valid",
        )

        conv_args["out_channels"] = conv_args[
            "in_channels"
        ]  # The depthwise conv wont do any channel change
        self.depthwise = DepthwiseConv(activation_fn=activation_dw_fn, **conv_args)

    def forward(self, x):
        x = self.depthwise(x)
        if self.use_attention:
            x = self.attention(x)
        x = self.pointwise(x)
        return x
