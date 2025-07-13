# Libs >>>
import torch.nn.init as torch_init


# Main Funcs >>>
def dense_init(
    dense_layer,
    xavier_gain: float = 1.0,
) -> None:
    """
    Initialize dense (fully connected) layer weights using Xavier initialization.
    Optimized for fully connected layers in neural networks.

    Args:
        dense_layer: The dense layer to initialize
        xavier_gain (float): Scaling factor for Xavier initialization
    """
    torch_init.xavier_normal_(dense_layer.weight, gain=xavier_gain)


def conv_init(
    conv_layer,
    fan_mode: str = "fan_out",
    activation: str = "relu",
) -> None:
    """
    Initialize convolutional layer weights using Kaiming initialization.
    Optimized for convolutional layers with ReLU activation.

    Args:
        conv_layer: The convolutional layer to initialize
        fan_mode (str): Fan mode ('fan_in' or 'fan_out')
        activation (str): Nonlinearity after the layer
    """
    torch_init.kaiming_normal_(
        conv_layer.weight, mode=fan_mode, nonlinearity=activation
    )
