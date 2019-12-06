#include "readpng.h"

#include <torch/torch.h>
#include <png.h>
#include <setjmp.h>
#include <string>

namespace image {
namespace png {

torch::Tensor decode_png(const torch::Tensor& data) {
  auto png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  TORCH_CHECK(png_ptr, "libpng read structure allocation failed!")
  auto info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_read_struct(&png_ptr, nullptr, nullptr);
    // Seems redundant with the if statement. done here to avoid leaking memory.
    TORCH_CHECK(info_ptr, "libpng info structure allocation failed!")
  }

  auto datap = data.accessor<unsigned char, 1>().data();
  
  if (setjmp(png_jmpbuf(png_ptr)) != 0) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    auto errptr = static_cast<png_const_charp>(png_get_error_ptr(png_ptr));
    TORCH_CHECK(false, "Internal error.");
  }
  auto is_png = !png_sig_cmp(datap, 0, 8);
  TORCH_CHECK(is_png, "Content is not png!")

  struct Reader {
    png_const_bytep ptr;
  } reader;
  reader.ptr = png_const_bytep(datap) + 8;

  auto read_callback =
      [](png_structp png_ptr, png_bytep output, png_size_t bytes) {
        auto reader = static_cast<Reader*>(png_get_io_ptr(png_ptr));
        std::copy(reader->ptr, reader->ptr + bytes, output);
        reader->ptr += bytes;
      };

  png_set_sig_bytes(png_ptr, 8);
  png_set_read_fn(png_ptr, &reader, read_callback);
  png_read_info(png_ptr, info_ptr);

  png_uint_32 width, height;
  int bit_depth, color_type;
  auto retval = png_get_IHDR(
      png_ptr,
      info_ptr,
      &width,
      &height,
      &bit_depth,
      &color_type,
      nullptr,
      nullptr,
      nullptr);

  if (retval != 1 ) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    TORCH_CHECK(retval == 1, "Could read image metadata from content.")
  }
  if ( color_type != PNG_COLOR_TYPE_RGB ) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    TORCH_CHECK(color_type == PNG_COLOR_TYPE_RGB, "Non RGB images are not supported.")
  }

  auto tensor =
      torch::empty({int64_t(height), int64_t(width), int64_t(3)}, torch::kU8);
  auto ptr = tensor.accessor<uint8_t, 3>().data();
  auto bytes = png_get_rowbytes(png_ptr, info_ptr);
  for (decltype(height) i = 0; i < height; ++i) {
    png_read_row(png_ptr, ptr, nullptr);
    ptr += bytes;
  }
  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
  return tensor;
}

torch::Tensor read_png_from_file(const std::string& path) {
  auto options =
      torch::TensorOptions().dtype(torch::kU8);

  std::ifstream file(path, std::ios::binary | std::ios::ate);
  TORCH_CHECK(file.is_open(), "File does not exist!.");
  std::streamsize size = file.tellg();
  TORCH_CHECK(size > 0, "File is empty!.");
  file.close();

  auto data = torch::from_file(path, false, size, options).contiguous();
  return decode_png(data);
}
} // namespace png
} // namespace image
