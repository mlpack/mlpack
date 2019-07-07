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
  static void creatmat(MatType& output, size_t rowSize, size_t colSize)
  {
      output.zeros(rowSize, colSize);
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
  static void Encode(size_t ele, MatType& output, size_t row, size_t col)
  {
    output.at(row, col) = ele;
  }

  /**
  * A function to encode given array of strings using a particular delimiter,
  * with custom tokenization.
  * The function does not paddes 0 in this case.
  *
  * @param input Vector of strings.
  * @param output Vector of vectors to store encoded results.
  * @param mappings Data structure carriying the information about the word
  *     and their mapping.
  * @param originalString Data structure carrying the original String of the 
  *     view created in mappings. 
  * @param tokenizer A function that accepts a boost::string_view as
  *                  an argument and returns a token.
  * This can either be a function pointer or function object or a lamda.
  * function. Its return value should be a boost::string_view, a token.
  *
  * Definition of function should be of type
  * boost::string_view fn(boost::string_view& str)
  *
 
  */
  template<typename TokenizerType>
  static void EncodeWithoutPad(const std::vector<std::string>& input,
                    std::vector<std::vector<size_t> >& output,
                    TokenizerType tokenizer,
                    std::unordered_map<boost::string_view, size_t,
                    boost::hash<boost::string_view>>& mappings,
                    std::deque<std::string>& originalStrings)
  {
    boost::string_view strView;
    boost::string_view token;
    size_t curLabel = mappings.size() + 1;
    for (size_t i = 0; i < input.size(); ++i)
    {
      output.push_back(std::vector<size_t>() );
      strView = input[i];
      token = tokenizer(strView);
      while (!token.empty())
      {
        if (mappings.count(token) == 0)
        {
          originalStrings.push_back(std::string(token));
          mappings[originalStrings.back()] = curLabel++;
        }
        output[i].push_back(mappings.at(token));
        token = tokenizer(strView);
      }
    }
  }
}; // class DicitonaryEncoding

} // namespace data
} // namespace mlpack



#endif
