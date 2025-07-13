# Main >>>
def scn(input_nc: float, divisible: int = 16, mode: str = "round") -> int:
    """
    Standardizes the number of channels to be divisible by a given number.

    Args:
        input_nc (float): Input number of channels
        divisible (int): Number to make input_nc divisible by
        mode (str): Rounding mode - "round", "floor", or "ceil"

    Returns:
        int: Standardized number of channels
    """
    if mode == "round":
        nc = round(input_nc / divisible) * divisible
    elif mode == "floor":
        nc = int(input_nc // divisible**2)
    elif mode == "ceil":
        nc = int((input_nc + divisible - 1) // divisible**2)
    else:
        raise ValueError("Invalid mode")

    return nc
