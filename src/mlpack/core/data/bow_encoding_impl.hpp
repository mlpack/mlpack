/**
 * @file bow_impl.hpp
 * @author Jeffin Sam
 *
 * Implementation of Bag of Words encoding functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_BOW_ENCODING_IMPL_HPP
#define MLPACK_CORE_DATA_BOW_ENCODING_IMPL_HPP

// In case it hasn't been included yet.
#include "bow_encoding.hpp"

namespace mlpack {
namespace data {

void Bow::Reset()
{
  mappings.clear();
  originalStrings.clear();
}

Bow::Bow(const Bow& oldObject)
{
  originalStrings = oldObject.originalStrings;
  std::deque<std::string>::iterator jt = originalStrings.begin();
  for (auto it = oldObject.originalStrings.begin(); it !=
      oldObject.originalStrings.end(); it++)
  {
    mappings[*jt] = oldObject.mappings.at(*it);
    jt++;
  }
}

Bow& Bow::operator= (const Bow &oldObject)
{
  if (this != &oldObject)
  {
    mappings.clear();
    originalStrings.clear();
    originalStrings = oldObject.originalStrings;
    std::deque<std::string>::iterator jt = originalStrings.begin();
    for (auto it = oldObject.originalStrings.begin(); it !=
        oldObject.originalStrings.end(); it++)
    {
      mappings[*jt] = oldObject.mappings.at(*it);
      jt++;
    }
  }
}

template<typename TokenizerType>
void Bow::CreateMap(const std::string& input,
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

template<typename MatType, typename TokenizerType>
void Bow::Encode(const std::vector<std::string>& input,
                                MatType& output, TokenizerType tokenizer)
{
  boost::string_view strView;
  boost::string_view token;
  std::vector< std::vector<boost::string_view> > dataset;
  std::size_t curLabel = mappings.size();
  for (size_t i = 0; i < input.size(); i++)
  {
    strView = input[i];
    token = tokenizer(strView);
    dataset.push_back(std::vector<boost::string_view>() );
    while (!token.empty())
    {
      dataset[i].push_back(token);
      if (mappings.find(token) == mappings.end())
      {
        originalStrings.push_back(std::string(token));
        mappings[originalStrings.back()] = curLabel++;
      }
      token = tokenizer(strView);
    }
  }
  output.zeros(dataset.size(), mappings.size());
  for (size_t i = 0; i < dataset.size(); ++i)
  {
    for (size_t j = 0; j < dataset[i].size(); ++j)
    {
      output.at(i, mappings.at(dataset[i][j])) = 1;
    }
  }
}

template<typename Archive>
void Bow::serialize(Archive& ar, const unsigned int
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
