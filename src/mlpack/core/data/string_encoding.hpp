/**
 * @file core/data/string_encoding.hpp
 * @author Jeffin Sam
 * @author Mikhail Lozhnikov
 *
 * Definition of the StringEncoding class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_STRING_ENCODING_HPP
#define MLPACK_CORE_DATA_STRING_ENCODING_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/data/string_encoding_dictionary.hpp>
#include <mlpack/core/data/string_encoding_policies/policy_traits.hpp>
#include <vector>

namespace mlpack {
namespace data {

/**
 * The class translates a set of strings into numbers using various encoding
 * algorithms. The encoder writes data either in the column-major order or
 * in the row-major order depending on the output data type.
 *
 * @tparam EncodingPolicyType Type of the encoding algorithm itself.
 * @tparam DictionaryType Type of the dictionary.
 */
template<typename EncodingPolicyType,
         typename DictionaryType>
class StringEncoding
{
 public:
  /**
   * Pass the given arguments to the policy constructor and create
   * the StringEncoding object using the policy.
   */
  template<typename ... ArgTypes>
  StringEncoding(ArgTypes&& ... args);

  /**
   * Construct the class from the given encoding policy.
   *
   * @param encodingPolicy The given encoding policy.
   */
  StringEncoding(EncodingPolicyType encodingPolicy);

  /**
   * A variant of the copy constructor for non-constant objects.
   */
  StringEncoding(StringEncoding&);

  //! Default copy-constructor.
  StringEncoding(const StringEncoding&);

  //! Default copy assignment operator.
  StringEncoding& operator=(const StringEncoding&) = default;

  //! Default move-constructor.
  StringEncoding(StringEncoding&&);

  //! Default move assignment operator.
  StringEncoding& operator=(StringEncoding&&) = default;

  /**
   * Initialize the dictionary using the given corpus.
   *
   * @tparam TokenizerType Type of the tokenizer.
   *
   * @param input Corpus of text to encode.
   * @param tokenizer The tokenizer object.
   *
   * The tokenization algorithm has to be an object with two public methods:
   * 1. operator() which accepts a reference to std::string_view, extracts
   * the next token from the given view, removes the prefix containing
   * the extracted token and returns the token;
   * 2. IsTokenEmpty() that accepts a token and returns true if the given
   *    token is empty.
   */
  template<typename TokenizerType>
  void CreateMap(const std::string& input,
                 const TokenizerType& tokenizer);

  /**
   * Clear the dictionary.
   */
  void Clear();

  /**
   * Encode the given text and write the result to the given output. The encoder
   * writes data in the column-major order or in the row-major order depending
   * on the output data type.
   *
   * If the output type is either arma::mat or arma::sp_mat then the function
   * writes it in the column-major order. If the output type is 2D std::vector
   * then the function writes it in the row major order.
   *
   * @tparam OutputType Type of the output container. The function supports
   *                    the following types: arma::mat, arma::sp_mat,
   *                    std::vector<std::vector<>>.
   * @tparam TokenizerType Type of the tokenizer.
   *
   * @param input Corpus of text to encode.
   * @param output Output container to store the result.
   * @param tokenizer The tokenizer object.
   *
   * The tokenization algorithm has to be an object with two public methods:
   * 1. operator() which accepts a reference to std::string_view, extracts
   * the next token from the given view, removes the prefix containing
   * the extracted token and returns the token;
   * 2. IsTokenEmpty() that accepts a token and returns true if the given
   *    token is empty.
   */
  template<typename OutputType, typename TokenizerType>
  void Encode(const std::vector<std::string>& input,
              OutputType& output,
              const TokenizerType& tokenizer);

  //! Return the dictionary.
  const DictionaryType& Dictionary() const { return dictionary; }
  //! Modify the dictionary.
  DictionaryType& Dictionary() { return dictionary; }

  //! Return the encoding policy object.
  const EncodingPolicyType& EncodingPolicy() const { return encodingPolicy; }
  //! Modify the encoding policy object.
  EncodingPolicyType& EncodingPolicy() { return encodingPolicy; }

  /**
   * Serialize the class to the given archive.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /**
   * A helper function to encode the given text and write the result to
   * the given output. The encoder writes data in the column-major order or
   * in the row-major order depending on the output data type.
   *
   * If the output type is either arma::mat or arma::sp_mat then the function
   * writes it in the column-major order. If the output type is 2D std::vector
   * then the function writes it in the row major order.
   *
   * @tparam OutputType Type of the output container. The function supports
   *                    the following types: arma::mat, arma::sp_mat,
   *                    std::vector<std::vector<>>.
   * @tparam TokenizerType Type of the tokenizer.
   * @tparam PolicyType The type of the encoding policy. It has to be
   *                    equal to EncodingPolicyType.
   *
   * @param input Corpus of text to encode.
   * @param output Output container to store the result.
   * @param tokenizer The tokenizer object.
   * @param policy The policy object.
   *
   * The tokenization algorithm has to be an object with two public methods:
   * 1. operator() which accepts a reference to std::string_view, extracts
   * the next token from the given view, removes the prefix containing
   * the extracted token and returns the token;
   * 2. IsTokenEmpty() that accepts a token and returns true if the given
   *    token is empty.
   */
  template<typename OutputType, typename TokenizerType, typename PolicyType>
  void EncodeHelper(const std::vector<std::string>& input,
                    OutputType& output,
                    const TokenizerType& tokenizer,
                    PolicyType& policy);

  /**
   * A helper function to encode the given text and write the result to
   * the given output. This is an optimized overload for policies that support
   * the one pass encoding algorithm. The encoder writes data in the row-major
   * order.
   *
   * @tparam TokenizerType Type of the tokenizer.
   * @tparam PolicyType The type of the encoding policy. It has to be
   *                    equal to EncodingPolicyType.
   * @tparam ElemType Type of the output values.
   *
   * @param input Corpus of text to encode.
   * @param output Output container to store the result.
   * @param tokenizer The tokenizer object.
   * @param policy The policy object.
   *
   * The tokenization algorithm has to be an object with two public methods:
   * 1. operator() which accepts a reference to std::string_view, extracts
   * the next token from the given view, removes the prefix containing
   * the extracted token and returns the token;
   * 2. IsTokenEmpty() that accepts a token and returns true if the given
   *    token is empty.
   */
  template<typename TokenizerType, typename PolicyType, typename ElemType>
  void EncodeHelper(const std::vector<std::string>& input,
                    std::vector<std::vector<ElemType>>& output,
                    const TokenizerType& tokenizer,
                    PolicyType& policy,
                    std::enable_if_t<StringEncodingPolicyTraits<
                        PolicyType>::onePassEncoding>* = 0);

 private:
  //! The encoding policy object.
  EncodingPolicyType encodingPolicy;
  //! The dictionary that contains the tokens and their labels.
  DictionaryType dictionary;
};

} // namespace data
} // namespace mlpack

// Include implementation.
#include "string_encoding_impl.hpp"

#endif
