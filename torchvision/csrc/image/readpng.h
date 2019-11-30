#ifndef READPNG_H
#define READPNG_H

#include <torch/torch.h>
#include <string>

namespace image {
  namespace png {
  torch::Tensor decode_png(const torch::Tensor& data);
  torch::Tensor read_png_from_file(const std::string& path);
  } // namespace png
} // namespace image

#endif // READPNG_H
