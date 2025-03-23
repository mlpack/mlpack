/**
 * @file core/data/load_image.hpp
 * @author Mehul Kumar Nirala
 *
 * Implementation of image loading functionality via STB.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DATA_LOAD_IMAGE_HPP
#define MLPACK_CORE_DATA_LOAD_IMAGE_HPP

#include <mlpack/core/stb/stb.hpp>

#include "image_info.hpp"

namespace mlpack {
namespace data {

/**
 * Image load/save interfaces.
 */

/**
 * Load the image file into the given matrix.
 *
 * @param filename Name of the image file.
 * @param matrix Matrix to load the image into.
 * @param info An object of ImageInfo class.
 * @param fatal If an error should be reported as fatal (default false).
 * @return Boolean value indicating success or failure of load.
 */
template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal = false);

/**
 * Load the image file into the given matrix.
 *
 * @param files A vector consisting of filenames.
 * @param matrix Matrix to save the image from.
 * @param info An object of ImageInfo class.
 * @param fatal If an error should be reported as fatal (default false).
 * @return Boolean value indicating success or failure of load.
 */
template<typename eT>
bool Load(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal = false);

// Implementation found in load_image.hpp.
inline bool LoadImage(const std::string& filename,
                      arma::Mat<unsigned char>& matrix,
                      ImageInfo& info,
                      const bool fatal = false);

} // namespace data
} // namespace mlpack

// Include implementation of Load() for images.
#include "load_image_impl.hpp"

#endif
