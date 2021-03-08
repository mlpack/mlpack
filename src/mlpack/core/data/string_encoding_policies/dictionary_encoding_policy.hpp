/**
 * @file core/data/string_encoding_policies/dictionary_encoding_policy.hpp
 * @author Jeffin Sam
 * @author Mikhail Lozhnikov
 *
 * Definition of the DictionaryEncodingPolicy class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_STRING_ENCODING_POLICIES_DICTIONARY_ENCODING_POLICY_HPP
#define MLPACK_CORE_DATA_STRING_ENCODING_POLICIES_DICTIONARY_ENCODING_POLICY_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/data/string_encoding_policies/policy_traits.hpp>
#include <mlpack/core/data/string_encoding.hpp>

namespace mlpack {
namespace data {

/**
 * DicitonaryEnocdingPolicy is used as a helper class for StringEncoding.
 * The encoder assigns a positive integer number to each unique token and treats
 * the dataset as categorical. The numbers are assigned sequentially starting
 * from one. The order in which the tokens are labeled is defined by
 * the dictionary used by the StringEncoding class. The encoder writes data
 * either in the column-major order or in the row-major order depending on
 * the output data type.
 */
class DictionaryEncodingPolicy
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
   * @param maxNumTokens The maximum number of tokens in the strings of the
   *                     input dataset.
   * @param * (dictionarySize) The size of the dictionary (not used).
   */
  template<typename MatType>
  static void InitMatrix(MatType& output,
                         const size_t datasetSize,
                         const size_t maxNumTokens,
                         const size_t /* dictionarySize */)
  {
    output.zeros(maxNumTokens, datasetSize);
  }

  /**
   * The function performs the dictionary encoding algorithm i.e. it writes
   * the encoded token to the output. The encoder writes data in the
   * column-major order.
   *
   * @tparam MatType The output matrix type.
   *
   * @param output Output matrix to store the encoded results (sp_mat or mat).
   * @param value The encoded token.
   * @param line The line number at which the encoding is performed.
   * @param index The token index in the line.
   */
  template<typename MatType>
  static void Encode(MatType& output,
                     const size_t value,
                     const size_t line,
                     const size_t index)
  {
    output(index, line) = value;
  }

  /**
   * The function performs the dictionary encoding algorithm i.e. it writes
   * the encoded token to the output. This is an overloaded function which saves
   * the result into the given vector to avoid padding. The encoder writes data
   * in the row-major order.
   *
   * @tparam ElemType Type of the output values.
   *
   * @param output Output vector to store the encoded line.
   * @param value The encoded token.
   */
  template<typename ElemType>
  static void Encode(std::vector<ElemType>& output, size_t value)
  {
    output.push_back(value);
  }

  /**
   * The function is not used by the dictionary encoding policy.
   *
   * @param * (line) The line number at which the encoding is performed.
   * @param * (index) The token sequence number in the line.
   * @param * (value) The encoded token.
   */
  static void PreprocessToken(const size_t /* line */,
                              const size_t /* index */,
                              const size_t /* value */)
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
 * The specialization provides some information about the dictionary encoding
 * policy.
 */
template<>
struct StringEncodingPolicyTraits<DictionaryEncodingPolicy>
{
  /**
   * Indicates if the policy is able to encode the token at once without
   * any information about other tokens as well as the total tokens count.
   */
  static const bool onePassEncoding = true;
};

/**
 * A convenient alias for the StringEncoding class with DictionaryEncodingPolicy
 * and the default dictionary for the given token type.
 *
 * @tparam TokenType Type of the tokens.
 */
template<typename TokenType>
using DictionaryEncoding = StringEncoding<DictionaryEncodingPolicy,
                                          StringEncodingDictionary<TokenType>>;
} // namespace data
} // namespace mlpack

#endif
