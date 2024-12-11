/**
 * @file core/data/save_image.hpp
 * @author Ryan Curtin
 *
 * Implementation of save functionality.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_SAVE_IMAGE_HPP
#define MLPACK_CORE_DATA_SAVE_IMAGE_HPP

#include <mlpack/core/stb/stb.hpp>

#include "image_info.hpp"

namespace mlpack {
namespace data {

/**
 * Save the image file from the given matrix.
 *
 * @param filename Name of the image file.
 * @param matrix Matrix to save the image from.
 * @param info An object of ImageInfo class.
 * @param fatal If an error should be reported as fatal (default false).
 * @return Boolean value indicating success or failure of load.
 */
template<typename eT>
bool Save(const std::string& filename,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal = false);

/**
 * Save the image file from the given matrix.
 *
 * @param files A vector consisting of filenames.
 * @param matrix Matrix to save the image from.
 * @param info An object of ImageInfo class.
 * @param fatal If an error should be reported as fatal (default false).
 * @return Boolean value indicating success or failure of load.
 */
template<typename eT>
bool Save(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          ImageInfo& info,
          const bool fatal = false);

/**
 * Helper function to save files.  Implementation in save_image.hpp.
 */
inline bool SaveImage(const std::string& filename,
                      arma::Mat<unsigned char>& image,
                      ImageInfo& info,
                      const bool fatal = false);

} //namespace data
} //namespace mlpack

// Include implementation of Save() for images.
#include "save_image_impl.hpp"

#endif
