import torch
import os.path
from torch import nn, Tensor


def read_png_from_file(path):
    # type: (str) -> Tensor
    """
    Read a PNG image into a 3 dimensional RGB Tensor.
    The values of the output tensor are uint8 between 0 and 255.

    Arguments:
        path (str): The path of the string to read.

    Returns:
        output (Tensor[image_width, image_height, 3])
    """
    if not os.path.isfile(path):
        raise ValueError("File does not exist.")
    output = torch.ops.torchvision.read_png_from_file(path)
    return output


class ReadPNGFromFile(nn.Module):
    """
    See ps_roi_pool
    """
    def __init__(self):
        super(ReadPNGFromFile, self).__init__()

    def forward(self, input):
        return read_png_from_file(input)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += ')'
        return tmpstr
