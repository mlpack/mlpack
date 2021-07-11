/**
 * @file preprocess_string_util.cpp
 * @author Jeffin Sam
 *
 * A CLI executable to encode string dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_PREPROCESS_PREPROCESS_STRING_UTIL_HPP
#define MLPACK_METHODS_PREPROCESS_PREPROCESS_STRING_UTIL_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>
#include <unordered_set>
#include <mlpack/core/data/extension.hpp>

namespace mlpack {
namespace data {

/**
 * Function neccessary to create a vector<vector<string>> by readin
 * the contents of a file.
 *
 * @param filename Name of the file whose contents need to be preproccessed.
 * @param columnDelimiter Delimiter used to split the columns of file.
 */
std::vector<std::vector<std::string>> CreateDataset(const std::string& filename,
                                                    char columnDelimiter);
/**
 * Function to check whether the given column contains only digits or not.
 *
 * @param column Column index to check.
 */
bool IsNumber(const std::string& column);

/**
 * The function parses the given column indices and ranges.
 *
 * @param dimensions A vector of column indices or column ranges.
 */
std::vector<size_t> GetColumnIndices(const std::vector<std::string>& dimensions);

/**
 * Function to get the type of column delimiter base on file extension.
 *
 * @param filename Name of the input file.
 */
std::string ColumnDelimiterType(const std::string& filename);

} // namespace data
} // namespace mlpack

#endif
