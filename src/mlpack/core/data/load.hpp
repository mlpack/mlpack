/**
 * @file core/data/load.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * mlpack Load function interface from a file.
 *
 * This Load interface allows to load numeric / image / models from disk into
 * an Armadillo matrix or mlpack object.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_HPP
#define MLPACK_CORE_DATA_LOAD_HPP

#include <mlpack/prereqs.hpp>

#include "image_options.hpp"
#include "text_options.hpp"
#include "detect_file_type.hpp"
#include "load_deprecated.hpp"
#include "load_arff.hpp"
#include "load_numeric.hpp"
#include "load_image.hpp"
#include "load_model.hpp"

namespace mlpack {
namespace data /** Functions to load and save matrices and models. */ {

/**
 * Loads a matrix from file, guessing the filetype from the extension.  This
 * will load with the options specified in `opts`.
 *
 * @param filename Name of file to load.
 * @param matrix Matrix to load contents of file into.
 * @param opts DataOptions to be passed to the function
 * @return Boolean value indicating success or failure of load.
 */
template<typename MatType, typename DataOptionsType>
bool Load(const std::string& filename,
          MatType& matrix,
          DataOptionsType& opts,
          const typename std::enable_if_t<
              IsDataOptions<DataOptionsType>::value>* = 0);

/**
 * Loads a matrix from file, guessing the filetype from the extension.  This
 * will load with the options specified in `opts`.
 *
 * @param filename Name of file to load.
 * @param matrix Matrix to load contents of file into.
 * @param opts Non-modifiable DataOptions to be passed to the function
 * @return Boolean value indicating success or failure of load.
 */
template<typename MatType, typename DataOptionsType>
bool Load(const std::string& filename,
          MatType& matrix,
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
template<typename eT, typename DataOptionsType>
bool Load(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          DataOptionsType& opts,
          const typename std::enable_if_t<
              IsDataOptions<DataOptionsType>::value>* = 0);

} // namespace data
} // namespace mlpack

// Include implementation of Load() for matrix.
#include "load_impl.hpp"

#endif
