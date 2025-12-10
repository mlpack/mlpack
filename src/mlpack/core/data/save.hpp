/**
 * @file core/data/save.hpp
 * @author Ryan Curtin
 *
 * Save an Armadillo matrix to file.  This is necessary because Armadillo does
 * not transpose matrices upon saving, and it allows us to give better error
 * output.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_SAVE_HPP
#define MLPACK_CORE_DATA_SAVE_HPP

#include <mlpack/prereqs.hpp>

#include "image_options.hpp"
#include "text_options.hpp"
#include "detect_file_type.hpp"
#include "save_deprecated.hpp"
#include "save_numeric.hpp"
#include "save_model.hpp"
#include "save_image.hpp"

namespace mlpack {
namespace data /** Functions to load and save matrices. */ {

/**
 * This function defines a unified data saving interface for the library.
 * Using this function it will be possible to save matrices, models, and
 * images.
 *
 * To specify what you would like to save, please use the DataOptionsType.
 *
 * @param filename Name of file to load.
 * @param matrix Matrix to load contents of file into.
 * @param opts DataOptions to be passed to the function
 * @return Boolean value indicating success or failure of Save.
 */
template<typename MatType, typename DataOptionsType>
bool Save(const std::string& filename,
          const MatType& matrix,
          DataOptionsType& opts,
          const typename std::enable_if_t<
              IsDataOptions<DataOptionsType>::value>* = 0);

template<typename MatType, typename DataOptionsType>
bool Save(const std::string& filename,
          const MatType& matrix,
          const DataOptionsType& opts,
          const typename std::enable_if_t<
              IsDataOptions<DataOptionsType>::value>* = 0);

template<typename eT>
bool Save(const std::vector<std::string>& files,
          const arma::Mat<eT>& matrix,
          ImageOptions& opts);

// Image saving API for multiple files.
template<typename eT>
bool Save(const std::vector<std::string>& files,
          const arma::Mat<eT>& matrix,
          ImageOptions& opts)

{
  return SaveImage(files, matrix, opts);
}

} // namespace data
} // namespace mlpack

// Include implementation.
#include "save_impl.hpp"

#endif
