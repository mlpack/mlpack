/**
 * @file core/data/string_encoding_policies/bag_of_words_encoding_policy.hpp
 * @author Jeffin Sam
 * @author Mikhail Lozhnikov
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
 * each dataset item to a vector of size N, where N is equal to the total unique
 * number of tokens. The i-th coordinate of the output vector is equal to
 * the number of times when the i-th token occurs in the corresponding dataset
 * item. The order in which the tokens are labeled is defined by the dictionary
 * used by the StringEncoding class. The encoder writes data either in the
 * column-major order or in the row-major order depending on the output data
 * type.
 */
class BagOfWordsEncodingPolicy
{
 public:
  /**
   * Clear the necessary internal variables.
   */
  static void Reset()
  {
    // Nothing to do.
  }

  /**
   * The function initializes the output matrix. The encoder writes data
   * in the column-major order.
   *
   * @tparam MatType The output matrix type.
   *
   * @param output Output matrix to store the encoded results (sp_mat or mat).
   * @param datasetSize The number of strings in the input dataset.
   * @param * (maxNumTokens) The maximum number of tokens in the strings of the
   *                     input dataset (not used).
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
   * The function initializes the output matrix. The encoder writes data
   * in the row-major order.
   *
   * Overloaded function to save the result in vector<vector<ElemType>>.
   *
   * @tparam ElemType Type of the output values.
   *
   * @param output Output matrix to store the encoded results.
   * @param datasetSize The number of strings in the input dataset.
   * @param * (maxNumTokens) The maximum number of tokens in the strings of the
   *                     input dataset (not used).
   * @param dictionarySize The size of the dictionary.
   */
  template<typename ElemType>
  static void InitMatrix(std::vector<std::vector<ElemType>>& output,
                         const size_t datasetSize,
                         const size_t /* maxNumTokens */,
                         const size_t dictionarySize)
  {
    output.resize(datasetSize, std::vector<ElemType>(dictionarySize));
  }

  /**
   * The function performs the bag of words encoding algorithm i.e. it writes
   * the encoded token to the output. The encoder writes data in the
   * column-major order.
   *
   * @tparam MatType The output matrix type.
   *
   * @param output Output matrix to store the encoded results (sp_mat or mat).
   * @param value The encoded token.
   * @param line The line number at which the encoding is performed.
   * @param * (index) The token index in the line.
   */
  template<typename MatType>
  static void Encode(MatType& output,
                     const size_t value,
                     const size_t line,
                     const size_t /* index */)
  {
    // The labels are assigned sequentially starting from one.
    output(value - 1, line) += 1;
  }

  /**
   * The function performs the bag of words encoding algorithm i.e. it writes
   * the encoded token to the output. The encoder writes data in the
   * row-major order.
   *
   * Overloaded function to accept vector<vector<ElemType>> as the output
   * type.
   *
   * @tparam ElemType Type of the output values.
   *
   * @param output Output matrix to store the encoded results.
   * @param value The encoded token.
   * @param line The line number at which the encoding is performed.
   * @param * (index) The line token number at which the encoding is performed.
   */
  template<typename ElemType>
  static void Encode(std::vector<std::vector<ElemType>>& output,
                     const size_t value,
                     const size_t line,
                     const size_t /* index */)
  {
    // The labels are assigned sequentially starting from one.
    output[line][value - 1] += 1;
  }

  /**
   * The function is not used by the bag of words encoding policy.
   *
   * @param * (line) The line number at which the encoding is performed.
   * @param * (index) The token sequence number in the line.
   * @param * (value) The encoded token.
   */
  static void PreprocessToken(size_t /* line */,
                              size_t /* index */,
                              size_t /* value */)
  { }

  /**
   * Serialize the class to the given archive.
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */)
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
