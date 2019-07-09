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

namespace mlpack {
namespace data {

template<typename EncodingPolicy>
void StringEncoding<EncodingPolicy>::Reset()
{
  mappings.clear();
  originalStrings.clear();
}

template<typename EncodingPolicy>
StringEncoding<EncodingPolicy>::StringEncoding(const
    StringEncoding& oldObject) : originalStrings(oldObject.originalStrings)
{
  std::deque<std::string>::iterator jt = originalStrings.begin();
  for (auto it = oldObject.originalStrings.begin();
      it != oldObject.originalStrings.end(); it++)
  {
    mappings[*jt] = oldObject.mappings.at(*it);
    jt++;
  }
}

template<typename EncodingPolicy>
StringEncoding<EncodingPolicy>& StringEncoding<EncodingPolicy>::operator= (
    const StringEncoding &oldObject)
{
  if (this != &oldObject)
  {
    mappings.clear();
    originalStrings.clear();
    originalStrings = oldObject.originalStrings;
    std::deque<std::string>::iterator jt = originalStrings.begin();
    for (auto it = oldObject.originalStrings.begin();
        it != oldObject.originalStrings.end(); it++)
    {
      mappings[*jt] = oldObject.mappings.at(*it);
      jt++;
    }
  }
  return *this;
}

template<typename EncodingPolicy>
template<typename TokenizerType>
void StringEncoding<EncodingPolicy>::CreateMap(std::string& input,
    TokenizerType tokenizer)
{
  boost::string_view strView(input);
  boost::string_view token;
  token = tokenizer(strView);
  std::size_t curLabel = mappings.size() + 1;
  while (!token.empty())
  {
    if (mappings.find(token) == mappings.end())
    {
        originalStrings.push_back(std::string(token));
        mappings[originalStrings.back()] = curLabel++;
    }
    token = tokenizer(strView);
  }
}

template<typename EncodingPolicy>
template<typename MatType, typename TokenizerType>
void StringEncoding<EncodingPolicy>::Encode(
    const std::vector<std::string>& input,
    MatType& output, TokenizerType tokenizer)
{
  boost::string_view strView;
  boost::string_view token;
  std::vector< std::vector<boost::string_view> > dataset;
  size_t colSize = 0;
  for (size_t i = 0; i < input.size(); i++)
  {
    strView = input[i];
    token = tokenizer(strView);
    dataset.push_back(std::vector<boost::string_view>() );
    while (!token.empty())
    {
      dataset[i].push_back(token);
      token = tokenizer(strView);
    }
    colSize = std::max(colSize, dataset[i].size());
  }
  size_t curLabel = mappings.size() + 1;
  for (size_t i = 0; i < dataset.size(); ++i)
  {
    for (size_t j = 0; j < dataset[i].size(); ++j)
    {
      if (mappings.find(dataset[i][j]) == mappings.end())
      {
        originalStrings.push_back(std::string(dataset[i][j]));
        mappings[originalStrings.back()] = curLabel++;
      }
    }
  }
  EncodingPolicy::InitMatrix(output, dataset.size(), colSize, mappings.size());
  for (size_t i = 0; i < dataset.size(); ++i)
    for (size_t j = 0; j < dataset[i].size(); ++j)
        EncodingPolicy::Encode(mappings.at(dataset[i][j]), output, i, j);
  }

template<typename EncodingPolicy>
template<typename TokenizerType>
void StringEncoding<EncodingPolicy>::Encode(
    const std::vector<std::string>& input,
    std::vector<std::vector<size_t>>& output,
    TokenizerType tokenizer,
    typename std::enable_if<PolicyTraits<EncodingPolicy>::
    outputWithNoPadding>::type*)
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

template<typename EncodingPolicy>
template<typename Archive>
void StringEncoding<EncodingPolicy>::serialize(Archive& ar, const unsigned int
    /* version */)
{
  size_t count = originalStrings.size();
  ar & BOOST_SERIALIZATION_NVP(count);
  if (Archive::is_saving::value)
  {
    for (size_t i = 0; i < count; i++)
    {
      ar & BOOST_SERIALIZATION_NVP(originalStrings[i]);
      ar & BOOST_SERIALIZATION_NVP(mappings.at(originalStrings[i]));
    }
  }
  if (Archive::is_loading::value)
  {
    originalStrings.resize(count);
    for (size_t i = 0; i < count; i++)
    {
      ar & BOOST_SERIALIZATION_NVP(originalStrings[i]);
      ar & BOOST_SERIALIZATION_NVP(mappings[originalStrings[i]]);
    }
  }
}

} // namespace data
} // namespace mlpack

#endif
