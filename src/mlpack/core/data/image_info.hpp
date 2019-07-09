/**
 * @file image_info.hpp
 * @author Mehul Kumar Nirala
 *
 * An image information holder.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DATA_IMAGE_INFO_HPP
#define MLPACK_CORE_DATA_IMAGE_INFO_HPP

#ifdef HAS_STB // Compile this only if stb is present.

#include <mlpack/prereqs.hpp>

#include "extension.hpp"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace mlpack {
namespace data {

/**
 * Checks if the given image filename is supported.
 *
 * @param filename Name of the image file.
 * @return Boolean value indicating success if it is an image.
 */
inline bool ImageFormatSupported(const std::string& fileName,
                                 bool save = false);

class ImageInfo
{
 public:
  /**
   * Instantiate the ImageInfo object with the given image width, height,
   * channels parameter.
   *
   * @param width Image width.
   * @param height Image height.
   * @param channels number of channels in the image.
   */
  ImageInfo(const size_t width = 0,
            const size_t height = 0,
            const size_t channels = 3,
            const size_t quality = 90);

  //! Get the image height.
  const size_t& Height() const { return height; }
  //! Modify the image height.
  size_t& Height() { return height; }

  //! Get the image width.
  const size_t& Width() const { return width; }
  //! Modify the image width.
  size_t& Width() { return width; }

  //! Get the image channels.
  const size_t& Channels() const { return channels; }
  //! Modify the image channels.
  size_t& Channels() { return channels; }

  //! Get the image quality.
  const size_t& Quality() const { return quality; }
  //! Modify the image quality.
  size_t& Quality() { return quality; }

 private:
  // To store the image height.
  size_t height;

  // To store the image width;
  size_t width;

  // To store the number of channels in the image.
  size_t channels;

  // Compression of the image if saved as jpg (0 - 100).
  size_t quality;

  // Image format.
  std::string format;

  // Flip the image vertical upon loading/saving.
  bool flipVertical;
};

} // namespace data
} // namespace mlpack

// Include implementation of Image.
#include "image_info_impl.hpp"

#endif // HAS_STB.

#endif
