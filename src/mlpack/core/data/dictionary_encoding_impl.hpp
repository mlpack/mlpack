/**
 * @file dictionary_encoding_impl.hpp
 * @author Jeffin Sam
 *
 * Implementation of dictionary encoding functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_DICT_ENCODING_IMPL_HPP
#define MLPACK_CORE_DATA_DICT_ENCODING_IMPL_HPP

// In case it hasn't been included yet.
#include "dictionary_encoding.hpp"

namespace mlpack {
namespace data {

void DicitonaryEncoding::Reset()
{
  mappings.clear();
  stringq.clear();
}

DicitonaryEncoding::DicitonaryEncoding(const DicitonaryEncoding &old_obj)
{
  this->stringq = old_obj.stringq;
  std::deque<std::string>::iterator jt = stringq.begin();
  for (auto it = old_obj.mappings.begin(); it != old_obj.mappings.end(); it++)
  {
    this->mappings[*jt] = it->second;
    jt++;
  }
}

void DicitonaryEncoding::operator= (const
    DicitonaryEncoding &old_obj)
{
  if (this != &old_obj)
  {
    this->mappings.clear();
    this->stringq.clear();
    this->stringq = old_obj.stringq;
    std::deque<std::string>::iterator jt = stringq.begin();
    for (auto it = old_obj.mappings.begin(); it != old_obj.mappings.end(); it++)
    {
      this->mappings[*jt] = it->second;
      jt++;
    }
  }
}

template<typename TokenizerType>
void DicitonaryEncoding::CreateMap(std::string& strings,
    TokenizerType tokenizer)
{
  boost::string_view strView(strings);
  boost::string_view token;
  token = tokenizer(strView);
  std::size_t curLabels = mappings.size() + 1;
  while (!token.empty())
  {
    if (mappings.find(token) == mappings.end())
    {
        stringq.push_back(std::string(token));
        mappings[stringq.back()] = curLabels++;
    }
    token = tokenizer(strView);
  }
}

template<typename MatType, typename TokenizerType>
void DicitonaryEncoding::Encode(const std::vector<std::string>& strings,
                MatType& output, TokenizerType tokenizer)
{
  boost::string_view strView;
  boost::string_view token;
  std::vector< std::vector<boost::string_view> > dataset;
  size_t colSize = 0;
  for (size_t i = 0; i < strings.size(); i++)
  {
    strView = strings[i];
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
  output.zeros(dataset.size(), colSize);
  for (size_t i = 0; i < dataset.size(); ++i)
  {
    for (size_t j = 0; j < dataset[i].size(); ++j)
    {
      if (mappings.find(dataset[i][j]) == mappings.end())
      {
        stringq.push_back(std::string(dataset[i][j]));
        mappings[stringq.back()] = curLabel++;
      }
      output.at(i, j) = mappings.at(dataset[i][j]);
    }
  }
}

template<typename TokenizerType>
void DicitonaryEncoding::Encode(const std::vector<std::string>& strings,
            std::vector<std::vector<size_t> >& output, TokenizerType tokenizer)
{
  boost::string_view strView;
  boost::string_view token;
  size_t curLabel = mappings.size() + 1;
  for (size_t i = 0; i < strings.size(); ++i)
  {
    output.push_back(std::vector<size_t>() );
    strView = strings[i];
    token = tokenizer(strView);
    while (!token.empty())
    {
      if (mappings.count(token) == 0)
      {
        stringq.push_back(std::string(token));
        mappings[stringq.back()] = curLabel++;
      }
    output[i].push_back(mappings.at(token));
    token = tokenizer(strView);
    }
  }
}

template<typename Archive>
void DicitonaryEncoding::serialize(Archive& ar, const unsigned int
    /* version */)
{
  size_t count = stringq.size();
  ar & BOOST_SERIALIZATION_NVP(count);
  if (Archive::is_saving::value)
  {
    for (size_t i = 0; i < count; i++)
    {
      ar & BOOST_SERIALIZATION_NVP(stringq[i]);
      ar & BOOST_SERIALIZATION_NVP(mappings.at(stringq[i]));
    }
  }
  if (Archive::is_loading::value)
  {
    stringq.resize(count);
    for (size_t i = 0; i < count; i++)
    {
      ar & BOOST_SERIALIZATION_NVP(stringq[i]);
      ar & BOOST_SERIALIZATION_NVP(mappings[stringq[i]]);
    }
  }
}

} // namespace data
} // namespace mlpack

#endif
