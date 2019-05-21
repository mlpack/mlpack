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
    if (mappings.find(std::string(token)) == mappings.end())
    {
      mappings[std::move(std::string(token))] = curLabels++;
    }
    token = tokenizer(strView);
  }
}

template<typename MatType, typename TokenizerType>
void DicitonaryEncoding::DictEncode(const std::vector<std::string>& strings,
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
      if (mappings.count(std::string(dataset[i][j])) == 0)
      {
        mappings[std::string(dataset[i][j])] = curLabel++;
      }
      output.at(i, j) = mappings.at(std::move(std::string(dataset[i][j])));
    }
  }
}

template<typename TokenizerType>
void DicitonaryEncoding::DictEncode(const std::vector<std::string>& strings,
            std::vector<std::vector<int> >& output, TokenizerType tokenizer)
{
  boost::string_view strView;
  boost::string_view token;
  size_t curLabel = mappings.size() + 1;
  for (size_t i = 0; i < strings.size(); ++i)
  {
    output.push_back(std::vector<int>() );
    strView = strings[i];
    token = tokenizer(strView);
    while (!token.empty())
    {
      if (mappings.count(std::string(token)) == 0)
      {
        mappings[std::string(token)] = curLabel++;
      }
    output[i].push_back(mappings[std::move(std::string(token))]);
    token = tokenizer(strView);
    }
  }
}

} // namespace data
} // namespace mlpack

#endif
