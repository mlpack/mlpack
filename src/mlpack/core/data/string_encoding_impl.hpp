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

template<typename EncodingPolicy, typename DictionaryType>
template<typename ... ArgTypes>
StringEncoding<EncodingPolicy, DictionaryType>::StringEncoding(
    ArgTypes&& ... args) :
    policy(std::forward<ArgTypes>(args)...)
{
}

template<typename EncodingPolicy, typename DictionaryType>
StringEncoding<EncodingPolicy, DictionaryType>::StringEncoding(
    EncodingPolicy policy) :
    policy(std::move(policy))
{
}

template<typename EncodingPolicy, typename DictionaryType>
StringEncoding<EncodingPolicy, DictionaryType>::StringEncoding(
    StringEncoding& other) :
    policy(other.policy),
    dictionary(other.dictionary)
{
}

template<typename EncodingPolicy, typename DictionaryType>
void StringEncoding<EncodingPolicy, DictionaryType>::Clear()
{
  dictionary.Clear();
}

template<typename EncodingPolicy, typename DictionaryType>
template<typename TokenizerType>
void StringEncoding<EncodingPolicy, DictionaryType>::CreateMap(
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

  while (!token.empty())
  {
    if (!dictionary.HasToken(token))
      dictionary.AddToken(token);

    token = tokenizer(strView);
  }
}

template<typename EncodingPolicy, typename DictionaryType>
template<typename OutputType, typename TokenizerType>
void StringEncoding<EncodingPolicy, DictionaryType>::Encode(
    const std::vector<std::string>& input,
    OutputType& output,
    const TokenizerType& tokenizer)
{
  EncodeHelper(input, output, tokenizer, policy);
}


template<typename EncodingPolicy, typename DictionaryType>
template<typename MatType, typename TokenizerType, typename EncodingPolicyType>
void StringEncoding<EncodingPolicy, DictionaryType>::
EncodeHelper(const std::vector<std::string>& input,
             MatType& output,
             const TokenizerType& tokenizer,
             EncodingPolicyType& policy)
{
  size_t numColumns = 0;

  for (const std::string& line : input)
  {
    boost::string_view strView(line);
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

template<typename EncodingPolicy, typename DictionaryType>
template<typename TokenizerType, typename EncodingPolicyType>
void StringEncoding<EncodingPolicy, DictionaryType>::
EncodeHelper(const std::vector<std::string>& input,
             std::vector<std::vector<size_t>>& output,
             const TokenizerType& tokenizer,
             EncodingPolicyType& policy,
             typename std::enable_if<StringEncodingPolicyTraits<
                 EncodingPolicyType>::onePassEncoding>::type*)
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

template<typename EncodingPolicy, typename DictionaryType>
template<typename Archive>
void StringEncoding<EncodingPolicy, DictionaryType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(dictionary);
}

} // namespace data
} // namespace mlpack

#endif
