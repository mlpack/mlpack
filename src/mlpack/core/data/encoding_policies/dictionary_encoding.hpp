/**
 * @file dictionary_encoding.hpp
 * @author Jeffin Sam
 *
 * Definition of dictionary encoding functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_DICTIONARY_ENCODING_HPP
#define MLPACK_CORE_DATA_DICTIONARY_ENCODING_HPP

#include <mlpack/prereqs.hpp>
#include "mlpack/core/boost_backport/boost_backport_string_view.hpp"
#include <utility>

namespace mlpack {
namespace data {

/**
 * A simple Dicitonary Enocding class.
 *
 * DicitonaryEnocding is used as a helper class for StringEncoding.
 * The encoding here simply assigns a word (or a character) to a numeric
 * index and treat the dataset as categorical.The numeric index is simply
 * integer just as they would occur in dictionary.
 */
class DictionaryEncoding
{
 public:
  /**
  * A function used to create the matrix depending upon the size.
  *
  * @param output Output matrix to store encoded results (sp_mat or mat).
  * rowSize Number of rows of matrix
  * colSize Number of Columns of matrix 
  */
  template<typename MatType>
  static void InitMatrix(MatType& output, size_t datasetSize, size_t colSize,
                         size_t /*mappingsSize*/)
  {
      output.zeros(datasetSize, colSize);
  }

  /** 
  * A function to store the encoded word at exact index.
  *
  * @param ele The encoded word
  * @param output Output matrix to store encoded results (sp_mat or mat).
  * @param row The row at which the encoding belongs to.
  * @param col The column at which the encoding belongs to.
  */
  template<typename MatType>
  static void Encode(std::unordered_map<boost::string_view, size_t,
                     boost::hash<boost::string_view>>& mappings,
                     std::deque<std::string>& /*originalStrings*/,
                     std::vector< std::vector<boost::string_view> >& dataset,
                     MatType& output)
  {
    for (size_t i = 0; i < dataset.size(); ++i)
      for (size_t j = 0; j < dataset[i].size(); ++j)
        output.at(i, j) = mappings.at(dataset[i][j]);
  }
}; // class DicitonaryEncoding

} // namespace data
} // namespace mlpack



#endif
