/**
 * @file dict_encoding.hpp
 * @author Jeffin Sam
 *
 * Implementation of dictionary encoding functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_DICT_ENCODING_HPP
#define MLPACK_CORE_DATA_DICT_ENCODING_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {
/**
 * Dictionary Encoding
 * here we simply assign a word (or a character) to a numeric index
 * and treat the dataset as categorical.
 *
 * @param vector of documents.
 * @param mapping of string to their encoded number.
 * @param output matrix.
 */
template<typename eT>
void Encode(const std::vector<std::string>& strings,
            std::unordered_map<std::string, size_t>& mappings,
            arma::Mat<eT>& output);
} // namespace data
} // namespace mlpack

// Include implementation.
#include "dict_encoding_impl.hpp"

#endif
