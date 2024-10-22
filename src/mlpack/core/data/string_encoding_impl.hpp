/**
 * @file core/data/string_encoding_impl.hpp
 * @author Jeffin Sam
 * @author Mikhail Lozhnikov
 *
 * Implementation of the StringEncoding class.
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
{ }

template<typename EncodingPolicyType, typename DictionaryType>
StringEncoding<EncodingPolicyType, DictionaryType>::StringEncoding(
    EncodingPolicyType encodingPolicy) :
    encodingPolicy(std::move(encodingPolicy))
{ }

template<typename EncodingPolicyType, typename DictionaryType>
StringEncoding<EncodingPolicyType, DictionaryType>::StringEncoding(
    StringEncoding& other) :
    encodingPolicy(other.encodingPolicy),
    dictionary(other.dictionary)
{ }

template<typename EncodingPolicyType, typename DictionaryType>
StringEncoding<EncodingPolicyType, DictionaryType>::StringEncoding(
    const StringEncoding& other) :
    encodingPolicy(other.encodingPolicy),
    dictionary(other.dictionary)
{ }

template<typename EncodingPolicyType, typename DictionaryType>
StringEncoding<EncodingPolicyType, DictionaryType>::StringEncoding(
    StringEncoding&& other) :
    encodingPolicy(std::move(other.encodingPolicy)),
    dictionary(std::move(other.dictionary))
{ }

template<typename EncodingPolicyType, typename DictionaryType>
void StringEncoding<EncodingPolicyType, DictionaryType>::Clear()
{
  dictionary.Clear();
}

template<typename EncodingPolicyType, typename DictionaryType>
template<typename TokenizerType>
void StringEncoding<EncodingPolicyType, DictionaryType>::CreateMap(
    const std::string& input,
    const TokenizerType& tokenizer)
{
  std::string_view strView(input);
  auto token = tokenizer(strView);

  static_assert(
      std::is_same_v<std::remove_reference_t<decltype(token)>,
                     std::remove_reference_t<typename DictionaryType::
                        TokenType>>,
      "The dictionary token type doesn't match the return value type "
      "of the tokenizer.");

  // The loop below adds the extracted tokens to the dictionary.
  while (!tokenizer.IsTokenEmpty(token))
  {
    if (!dictionary.HasToken(token))
      dictionary.AddToken(std::move(token));

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

  policy.Reset();

  // The first pass adds the extracted tokens to the dictionary.
  for (size_t i = 0; i < input.size(); ++i)
  {
    std::string_view strView(input[i]);
    auto token = tokenizer(strView);

    static_assert(
        std::is_same_v<std::remove_reference_t<decltype(token)>,
                       std::remove_reference_t<typename DictionaryType::
                          TokenType>>,
        "The dictionary token type doesn't match the return value type "
        "of the tokenizer.");

    size_t numTokens = 0;

    while (!tokenizer.IsTokenEmpty(token))
    {
      if (!dictionary.HasToken(token))
        dictionary.AddToken(std::move(token));

      policy.PreprocessToken(i, numTokens, dictionary.Value(token));

      token = tokenizer(strView);
      numTokens++;
    }

    numColumns = std::max(numColumns, numTokens);
  }

  policy.InitMatrix(output, input.size(), numColumns, dictionary.Size());

  // The second pass writes the encoded values to the output.
  for (size_t i = 0; i < input.size(); ++i)
  {
    std::string_view strView(input[i]);
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
template<typename TokenizerType, typename PolicyType, typename ElemType>
void StringEncoding<EncodingPolicyType, DictionaryType>::
EncodeHelper(const std::vector<std::string>& input,
             std::vector<std::vector<ElemType>>& output,
             const TokenizerType& tokenizer,
             PolicyType& policy,
             std::enable_if_t<StringEncodingPolicyTraits<
                 PolicyType>::onePassEncoding>*)
{
  policy.Reset();

  // The loop below extracts the tokens and writes the encoded values
  // at once.
  for (size_t i = 0; i < input.size(); ++i)
  {
    std::string_view strView(input[i]);
    auto token = tokenizer(strView);

    static_assert(
        std::is_same_v<std::remove_reference_t<decltype(token)>,
                       std::remove_reference_t<typename DictionaryType::
                          TokenType>>,
        "The dictionary token type doesn't match the return value type "
        "of the tokenizer.");

    output.emplace_back();

    while (!tokenizer.IsTokenEmpty(token))
    {
      if (dictionary.HasToken(token))
        policy.Encode(output[i], dictionary.Value(token));
      else
        policy.Encode(output[i], dictionary.AddToken(std::move(token)));

      token = tokenizer(strView);
    }
  }
}

template<typename EncodingPolicyType, typename DictionaryType>
template<typename Archive>
void StringEncoding<EncodingPolicyType, DictionaryType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(encodingPolicy));
  ar(CEREAL_NVP(dictionary));
}

} // namespace data
} // namespace mlpack

#endif
