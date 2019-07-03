/**
 * @file image_info_impl.hpp
 * @author Mehul Kumar Nirala
 *
 * An image information holder implementation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DATA_IMAGE_INFO_IMPL_HPP
#define MLPACK_CORE_DATA_IMAGE_INFO_IMPL_HPP

#ifdef HAS_STB // Compile this only if stb is present.

// In case it hasn't been included yet.
#include "image_info.hpp"

namespace mlpack {
namespace data {

const std::vector<std::string> loadFileTypes({"jpg", "png", "tga",
    "bmp", "psd", "gif", "hdr", "pic", "pnm"});

const std::vector<std::string> saveFileTypes({"jpg", "png", "tga",
    "bmp", "hdr"});

bool ImageFormatSupported(const std::string& fileName, bool save)
{
  if (save)
  {
    // Iterate over all supported file types that can be saved.
    for (auto extension : saveFileTypes)
      if (extension == Extension(fileName))
        return true;
    return false;
  }
  else
  {
    // Iterate over all supported file types that can be loaded.
    for (auto extension : loadFileTypes)
      if (extension == Extension(fileName))
        return true;
    return false;
  }
}

ImageInfo::ImageInfo():
              width(0),
              height(0),
              channels(3),
              quality(90),
              format(""),
              flipVertical(true)
{
  // Do nothing.
}

ImageInfo::ImageInfo(const size_t width,
                     const size_t height,
                     const size_t channels):
                     width(width),
                     height(height),
                     channels(channels),
                     quality(90),
                     format(""),
                     flipVertical(true)
{
  // Do nothing.
}

ImageInfo::~ImageInfo()
{
  // Do nothing.
}

} // namespace data
} // namespace mlpack

#endif // HAS_STB.

#endif
