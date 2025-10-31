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

#include "image_options.hpp"

namespace mlpack {
namespace data {

/**
 * Image load/save interfaces.
 */

//
// Old Image loading interface, to be removed in mlpack 5.0.0
//

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
          ImageInfo& opts,
          const bool fatal)
{
  // Use the new implementation.
  opts.Fatal() = fatal;
  opts.Format() = FileType::ImageType;
  std::vector<std::string> files;
  files.push_back(filename);
  return LoadImage(files, matrix, opts);
}

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
          ImageInfo& opts,
          const bool fatal)
{
  // Use the new implementation.
  opts.Fatal() = fatal;
  opts.Format() = FileType::ImageType;
  return LoadImage(files, matrix, opts);
}

template<typename eT, typename DataOptionsType>
bool Load(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          const DataOptionsType& opts,
          const typename std::enable_if_t<
              IsDataOptions<DataOptionsType>::value>* = 0);
/**
 * This function loads a set of several dataset files into one matrix.
 * This is usually the case if the dataset is collected on several occasions
 * and not agglomerated into one file, or if the dataset has been partitioned
 * into multiple files.
 *
 * Note, the load will fail if the number of dimension (data points) in all
 * files is not equal, or if the dataset does not have the same filetype. For
 * example, the load will fail one file is CSV and the other is binary.
 *
 * The user needs to specify all the filenames in one std::vector before using
 * this function.
 *
 * @param filenames Names of files to load.
 * @param matrix Matrix to load contents of files into.
 * @param opts DataOptions to be passed to the function
 * @return Boolean value indicating success or failure of load.
 */
/**
 * Load a set of image files into the given matrix.
 *
 * @param files A vector consisting of filenames.
 * @param matrix Matrix to save the image from.
 * @param opts An object of ImageOptions class.
 * @return Boolean value indicating success or failure of load.
 */
template<typename eT, typename DataOptionsType>
bool Load(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          DataOptionsType& opts,
          const typename std::enable_if_t<
              IsDataOptions<DataOptionsType>::value>* = 0);

} // namespace data
} // namespace mlpack

// Include implementation of Load() for images.
#include "load_image_impl.hpp"

#endif
