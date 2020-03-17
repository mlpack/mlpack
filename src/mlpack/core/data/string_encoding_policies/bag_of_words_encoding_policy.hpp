/**
 * @file bag_of_words_encoding_policy.hpp
 * @author Jeffin Sam
 *
 * Definition of the BagOfWordsEncodingPolicy class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_STR_ENCODING_POLICIES_BAG_OF_WORDS_ENCODING_POLICY_HPP
#define MLPACK_CORE_DATA_STR_ENCODING_POLICIES_BAG_OF_WORDS_ENCODING_POLICY_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/data/string_encoding_policies/policy_traits.hpp>
#include <mlpack/core/data/string_encoding.hpp>

namespace mlpack {
namespace data {

/**
 * Definition of the BagOfWordsEncodingPolicy class.
 *
 * BagOfWords is used as a helper class for StringEncoding. The encoder maps
 * each dataset item to a vector of size N, where N is equal to the total number
 * of tokens. If an item of the dataset has the i-th token, then the i-th
 * coordinate of the corresponding vector is equal to 1, otherwise it's equal to
 * zero. The order in which the tokens are labeled is defined by the dictionary
 * used by the StringEncoding class.
 */
class BagOfWordsEncodingPolicy
{
 public:
  /**
   * The function initializes the output matrix.
   *
   * @tparam MatType The output matrix type.
   *
   * @param output Output matrix to store the encoded results (sp_mat or mat).
   * @param datasetSize The number of strings in the input dataset.
   * @param maxNumTokens The maximum number of tokens in the strings of the 
                         input dataset (not used).
   * @param dictionarySize The size of the dictionary.
   */
  template<typename MatType>
  static void InitMatrix(MatType& output,
                         const size_t datasetSize,
                         const size_t /* maxNumTokens */,
                         const size_t dictionarySize)
  {
    output.zeros(dictionarySize, datasetSize);
  }

  /**
   * The function initializes the output matrix.
   * Overloaded function to store result in vector<vector<OutputType>>
   * 
   * @tparam OutputType Type of the output vector.
   *
   * @param output Output matrix to store the encoded results.
   * @param datasetSize The number of strings in the input dataset.
   * @param maxNumTokens The maximum number of tokens in the strings of the 
                         input dataset (not used).
   * @param dictionarySize The size of the dictionary.
   */
  template<typename OutputType>
  static void InitMatrix(std::vector<std::vector<OutputType>>& output,
                         const size_t datasetSize,
                         const size_t /* maxNumTokens */,
                         const size_t dictionarySize)
  {
    output.resize(datasetSize, std::vector<OutputType>(dictionarySize, 0));
  }

  /** 
   * The function performs the bag of words encoding algorithm i.e. it writes
   * the encoded token to the output.
   * Returns the encodings in column-major format.
   *
   * @tparam MatType The output matrix type.
   *
   * @param output Output matrix to store the encoded results (sp_mat or mat).
   * @param value The encoded token.
   * @param row The row number at which the encoding is performed.
   * @param col The token index in the row.
   */
  template<typename MatType>
  static void Encode(MatType& output,
                     const size_t value,
                     const size_t row,
                     const size_t /* col */)
  {
    // The labels are assigned sequentially starting from one.
    output(value - 1, row) = 1;
  }

  /** 
   * The function performs the bag of words encoding algorithm i.e. it writes
   * the encoded token to the output.
   * Overload function to accepted vector<vector<OutputType>> as output type.
   * Returns the encodings in row-major format.
   *
   * @param output Output matrix to store the encoded results.
   * @param value The encoded token.
   * @param row The row number at which the encoding is performed.
   * @param col The row token number at which the encoding is performed.
   * @tparam OutputType The type of output vector.
   */
  template<typename OutputType>
  static void Encode(std::vector<std::vector<OutputType>>& output,
                     const size_t value,
                     const size_t row,
                     const size_t /* col */)
  {
    // The labels are assigned sequentially starting from one.
    output[row][value - 1] = 1;
  }

  /**
   * The function is not used by the bag of words encoding policy.
   *
   * @param row The row number at which the encoding is performed.
   * @param col The token sequence number in the row.
   * @param value The encoded token.
   */
  static void PreprocessToken(size_t /* row */,
                              size_t /* col */,
                              size_t /* value */)
  { }

  /**
   * Serialize the class to the given archive.
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const unsigned int /* version */)
  {
    // Nothing to serialize.
  }
};

/**
 * A convenient alias for the StringEncoding class with BagOfWordsEncodingPolicy
 * and the default dictionary for the given token type.
 *
 * @tparam TokenType Type of the tokens.
 */
template<typename TokenType>
using BagOfWordsEncoding = StringEncoding<BagOfWordsEncodingPolicy,
                                          StringEncodingDictionary<TokenType>>;
} // namespace data
} // namespace mlpack

#endif
