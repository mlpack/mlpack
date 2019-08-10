/**
 * @file string_encoding_impl.hpp
 * @author Jeffin Sam
 *
 * Implementation of string encoding functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_STRING_ENCODING_IMPL_HPP
#define MLPACK_CORE_DATA_STRING_ENCODING_IMPL_HPP

// In case it hasn't been included yet.
#include "string_encoding.hpp"
#include <type_traits>

namespace mlpack {
namespace data {

template<typename EncodingPolicyType, typename DictionaryType>
template<typename ... ArgTypes>
StringEncoding<EncodingPolicyType, DictionaryType>::StringEncoding(
    ArgTypes&& ... args) :
    encodingPolicy(std::forward<ArgTypes>(args)...)
{
}

template<typename EncodingPolicyType, typename DictionaryType>
StringEncoding<EncodingPolicyType, DictionaryType>::StringEncoding(
    EncodingPolicyType encodingPolicy) :
    encodingPolicy(std::move(encodingPolicy))
{
}

template<typename EncodingPolicyType, typename DictionaryType>
StringEncoding<EncodingPolicyType, DictionaryType>::StringEncoding(
    StringEncoding& other) :
    encodingPolicy(other.encodingPolicy),
    dictionary(other.dictionary)
{
}

template<typename EncodingPolicyType, typename DictionaryType>
StringEncoding<EncodingPolicyType, DictionaryType>::StringEncoding(
    const StringEncoding& other) :
    encodingPolicy(other.encodingPolicy),
    dictionary(other.dictionary)
{
}

template<typename EncodingPolicyType, typename DictionaryType>
StringEncoding<EncodingPolicyType, DictionaryType>::StringEncoding(
    StringEncoding&& other) :
    encodingPolicy(std::move(other.encodingPolicy)),
    dictionary(std::move(other.dictionary))
{
}

template<typename EncodingPolicyType, typename DictionaryType>
void StringEncoding<EncodingPolicyType, DictionaryType>::Clear()
{
  dictionary.Clear();
}

template<typename EncodingPolicyType, typename DictionaryType>
template<typename TokenizerType>
void StringEncoding<EncodingPolicyType, DictionaryType>::CreateMap(
    std::string& input,
    const TokenizerType& tokenizer)
{
  boost::string_view strView(input);
  auto token = tokenizer(strView);

    static_assert(
        std::is_same<typename std::remove_reference<decltype(token)>::type,
                     typename std::remove_reference<typename DictionaryType::
                        TokenType>::type>::value,
        "The dictionary token type doesn't match the return value type "
        "of the tokenizer.");

  while (!tokenizer.IsTokenEmpty(token))
  {
    if (!dictionary.HasToken(token))
      dictionary.AddToken(token);

    token = tokenizer(strView);
  }
}

template<typename EncodingPolicyType, typename DictionaryType>
template<typename OutputType, typename TokenizerType>
void StringEncoding<EncodingPolicyType, DictionaryType>::Encode(
    const std::vector<std::string>& input,
    OutputType& output,
    const TokenizerType& tokenizer)
{
  EncodeHelper(input, output, tokenizer, encodingPolicy);
}


template<typename EncodingPolicyType, typename DictionaryType>
template<typename MatType, typename TokenizerType, typename PolicyType>
void StringEncoding<EncodingPolicyType, DictionaryType>::
EncodeHelper(const std::vector<std::string>& input,
             MatType& output,
             const TokenizerType& tokenizer,
             PolicyType& policy)
{
  size_t numColumns = 0;

  for (size_t i = 0; i < input.size(); i++)
  {
    boost::string_view strView(input[i]);
    auto token = tokenizer(strView);

    static_assert(
        std::is_same<typename std::remove_reference<decltype(token)>::type,
                     typename std::remove_reference<typename DictionaryType::
                        TokenType>::type>::value,
        "The dictionary token type doesn't match the return value type "
        "of the tokenizer.");

    size_t numTokens = 0;

    while (!tokenizer.IsTokenEmpty(token))
    {
      if (!dictionary.HasToken(token))
        dictionary.AddToken(token);
      policy.PreprocessToken(i, numTokens, dictionary.Value(token));
      token = tokenizer(strView);
      numTokens++;
    }
    numColumns = std::max(numColumns, numTokens);
  }
  policy.InitMatrix(output, input.size(), numColumns, dictionary.Size());
  for (size_t i = 0; i < input.size(); i++)
  {
    boost::string_view strView(input[i]);
    auto token = tokenizer(strView);
    size_t numTokens = 0;

    while (!tokenizer.IsTokenEmpty(token))
    {
      policy.Encode(output, dictionary.Value(token), i, numTokens);
      token = tokenizer(strView);
      numTokens++;
    }
  }
}

template<typename EncodingPolicyType, typename DictionaryType>
template<typename TokenizerType, typename PolicyType,
typename OutputType>
void StringEncoding<EncodingPolicyType, DictionaryType>::
EncodeHelper(const std::vector<std::string>& input,
             std::vector<std::vector<OutputType>>& output,
             const TokenizerType& tokenizer,
             PolicyType& policy,
             typename std::enable_if<StringEncodingPolicyTraits<
                 PolicyType>::onePassEncoding>::type*)
{
  for (size_t i = 0; i < input.size(); i++)
  {
    boost::string_view strView(input[i]);
    auto token = tokenizer(strView);

    static_assert(
        std::is_same<typename std::remove_reference<decltype(token)>::type,
                     typename std::remove_reference<typename DictionaryType::
                        TokenType>::type>::value,
        "The dictionary token type doesn't match the return value type "
        "of the tokenizer.");

    output.emplace_back();

    while (!tokenizer.IsTokenEmpty(token))
    {
      if (!dictionary.HasToken(token))
        dictionary.AddToken(token);

      policy.Encode(output[i], dictionary.Value(token));
      token = tokenizer(strView);
    }
  }
}

template<typename EncodingPolicyType, typename DictionaryType>
template<typename Archive>
void StringEncoding<EncodingPolicyType, DictionaryType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(encodingPolicy);
  ar & BOOST_SERIALIZATION_NVP(dictionary);
}

} // namespace data
} // namespace mlpack

#endif
