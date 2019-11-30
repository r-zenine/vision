import torch
from torch import nn, Tensor


def decode_png(input):
    # type: (Tensor) -> Tensor
    """
    Decodes a PNG image into a 3 dimensional RGB Tensor.
    The values of the output tensor are uint8 between 0 and 255.

    Arguments:
        input (Tensor[1]): a one dimensional int8 tensor containing
    the raw bytes of the PNG image.

    Returns:
        output (Tensor[image_width, image_height, 3])
    """
    if not isinstance(raw_data, torch.Tensor) or len(raw_data) == 0:
        raise ValueError("Expected a non empty 1-dimensional tensor.")
    if not raw_data.dtype == torch.uint8:
        raise ValueError("Expected a torch.uint8 tensor.")
    output = torch.ops.torchvision.decode_png(raw_data)
    return output


class DecodePNG(nn.Module):
    """
    See ps_roi_pool
    """
    def __init__(self):
        super(DecodePNG, self).__init__()

    def forward(self, input):
        return decode_png(input)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += ')'
        return tmpstr
