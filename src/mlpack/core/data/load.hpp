/**
 * @file core/data/load.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * Load an Armadillo matrix from file.  This is necessary because Armadillo does
 * not transpose matrices on input, and it allows us to give better error
 * output.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_HPP
#define MLPACK_CORE_DATA_LOAD_HPP

#include <mlpack/prereqs.hpp>
#include <string>

#include "data_options.hpp"
#include "format.hpp"
#include "image_info.hpp"
#include "load_numeric.hpp"
#include "load_categorical.hpp"
#include "load_arff.hpp"
#include "load_image.hpp"
#include "utilities.hpp"

namespace mlpack {
namespace data /** Functions to load and save matrices and models. */ {

/**
 * Loads a matrix from file, guessing the filetype from the extension.  This
 * will transpose the matrix at load time (unless the transpose parameter is set
 * to false).
 *
 * @param filename Name of file to load.
 * @param matrix Matrix to load contents of file into.
 * @param opts DataOptions to be passed to the function
 * @return Boolean value indicating success or failure of load.
 */
template<typename MatType, typename DataOptionsType>
bool Load(const std::string& filename,
          MatType& matrix,
          DataOptionsType& opts);

// TODO: clean this up---this overload is necessary for when a user didn't make
// an actual DataOptionsType object.  We should check here that they didn't
// specify that they want to keep headers or anything like this and throw a
// warning.
template<typename MatType, typename DataOptionsType>
bool Load(const std::string& filename,
          MatType& matrix,
          const DataOptionsType& opts);

/**
 * This function a set of several dataset files into one matrix.
 * This is usually the case if the dataset is collected on several occasions
 * and not agglomerated into one file.
 *
 * Note, the number of columns in all files must be equal, and the dataset
 * needs to be of the same natures. Please do not load different datasets using
 * the following function.
 *
 * The user needs to specify all the filesname in one std::vector before using
 * this function.
 *
 * @param filename Names of files to load.
 * @param matrix Matrix to load contents of files into.
 * @param opts DataOptions to be passed to the function
 * @return Boolean value indicating success or failure of load.
 */
template<typename MatType, typename DataOptionsType>
bool Load(const std::vector<std::string>& filesname,
          MatType& matrix,
          DataOptionsType& opts);

/**
 * This function a set of several dataset files into one matrix.
 * This is usually the case if the dataset is collected on several occasions
 * and not agglomerated into one file.
 *
 * Note, the number of columns in all files must be equal, and the dataset
 * needs to be of the same natures. Please do not load different datasets using
 * the following function.
 *
 * The user needs to specify all the filesname in one std::vector before using
 * this function.
 *
 * @param filename Names of files to load.
 * @param matrix Matrix to load contents of files into.
 * @param opts DataOptions to be passed to the function
 * @return Boolean value indicating success or failure of load.
 */
template<typename MatType, typename DataOptionsType>
bool Load(const std::vector<std::string>& filesname,
          MatType& matrix,
          const DataOptionsType& opts);


} // namespace data
} // namespace mlpack

// Include implementation of Load() for matrix.
#include "load_impl.hpp"
// Include implementation of Load() for vectors.
#include "load_vec_impl.hpp"

#endif
