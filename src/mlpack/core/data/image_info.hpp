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

#include <mlpack/core.hpp>
#include <mlpack/core/util/log.hpp>
#include <mlpack/prereqs.hpp>

#include "extension.hpp"

namespace mlpack {
namespace data {

/**
 * Checks if the given image filename is supported.
 *
 * @param filename Name of the image file.
 * @return Boolean value indicating success if it is an image.
 */
bool ImageFormatSupported(const std::string& fileName, bool save = false);

class ImageInfo
{
 public:
  /**
   * ImageInfo default constructor.
   */
  ImageInfo();

  /**
   * Instantiate the ImageInfo object with the image width, height, channels.
   *
   * @param width Image width.
   * @param height Image height.
   * @param channels number of channels in the image.
   */
  ImageInfo(const size_t width, const size_t height, const size_t channels);

  /**
   * ImageInfo default destructor.
   */
  ~ImageInfo();

  // To store the image height.
  size_t height;

  // To store the image width;
  size_t width;

  // To store the number of channels in the image.
  size_t channels;

  // Compression of the image if saved as jpg (0-100).
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
