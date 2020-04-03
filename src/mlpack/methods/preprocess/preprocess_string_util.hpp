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
 * Function to check whether the given colum has only digits.
 *
 * @param column The column no
 */
bool IsNumber(const std::string& column);

/**
 * Function used to get the columns which has non numeric dataset.
 *
 * @param tempDimesnion A vector of string passed which has column number or
 *    column ranges.
 */
std::unordered_set<size_t> GetColumnIndices
    (const std::vector<std::string>& tempDimension);

/**
 * Function to get the type of column delimiter base on file extension.
 *
 * @param filename Name of the input file.
 */
std::string ColumnDelimiterType(const std::string& filename);

} // namespace data
} // namespace mlpack

#endif
