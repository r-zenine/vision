import os
import unittest

import torch
from PIL import Image
from torchvision.ops import decode_png, read_png_from_file
import numpy as np

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fakedata", "imagefolder")
def get_png_images(directory): 
    assert os.path.isdir(directory)
    for root, dir, files in os.walk(directory):
        for fl in files: 
            _, ext = os.path.splitext(fl)
            if ext == ".png":
                yield os.path.join(root, fl)


class ReadPngTester(unittest.TestCase): 
    def test_read_png_from_file(self):
        for img_path in get_png_images(IMAGE_DIR):
            img_pil = torch.from_numpy(np.array(Image.open(img_path)))
            img_lpng = read_png_from_file(img_path)
            self.assertTrue(torch.all(img_lpng == img_pil))
        
    def test_read_png_from_file(self):
        for img_path in get_png_images(IMAGE_DIR):
            img_pil = torch.from_numpy(np.array(Image.open(img_path)))
            img_lpng = decode_png(torch.from_numpy(np.fromfile(img_path, dtype=np.uint8)))
            self.assertTrue(torch.all(img_lpng == img_pil))
    

if __name__ == '__main__':
    unittest.main()
