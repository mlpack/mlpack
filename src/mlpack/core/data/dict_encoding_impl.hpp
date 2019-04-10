/**
 * @file dict_encoding_impl.hpp
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
#include "dict_encoding.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace mlpack {
namespace data {

  DicitonaryEncoding::DicitonaryEncoding()
  {
    // nothing to do here
  }

  DicitonaryEncoding::~DicitonaryEncoding()
  {
    // nothing to do here
  }

  void DicitonaryEncoding::Reset()
  {
    mappings.clear();
  }

  void DicitonaryEncoding::CreateMap(std::string strings, char deliminator)
  {
    std::vector<std::string> temp;
    boost::split(temp, strings, boost::is_any_of(std::string(1, deliminator)));
    size_t curLabel = mappings.size() + 1;
    for (size_t i = 0; i < temp.size(); i++)
    {
      mappings[temp[i]] = curLabel;
      curLabel++;
    }
  }

  void DicitonaryEncoding::CreateCharMap(std::string strings)
  {
    std::vector<char> temp;
    std::copy(strings.begin(), strings.end(), std::back_inserter(temp));
    size_t curLabel = mappings.size();
    for (size_t i = 0; i < temp.size(); i++)
    {
      mappings[(std::string(1, temp[i]))] = curLabel;
      curLabel++;
    }
  }

template<typename eT>
void DicitonaryEncoding::CharEncode(const std::vector<std::string>& strings,
            arma::Mat<eT>& output)
{
    CharEncode(strings, output, mappings);
}

template<typename eT>
void DicitonaryEncoding::CharEncode(const std::vector<std::string>& strings,
            arma::Mat<eT>& output,
            std::unordered_map<std::string, size_t>& mapping)
{
  std::vector< std::vector<char> > dataset;
  std::vector<char> temp;
  size_t globalsize = 0;
  for (size_t i = 0; i < strings.size(); ++i)
  {
    std::copy(strings[i].begin(), strings[i].end(), std::back_inserter(temp));
    dataset.push_back(temp);
    globalsize = std::max(globalsize, temp.size());
    temp.clear();
  }
  size_t curLabel = mapping.size() + 1;
  for (size_t i = 0; i < dataset.size(); ++i)
  {
    for (size_t j = 0; j < dataset[i].size(); ++j)
    {
      if (mapping.count((std::string(1, dataset[i][j]))) == 0)
      {
        mapping[(std::string(1, dataset[i][j]))] = curLabel;
        ++curLabel;
      }
    }
  }
  output.zeros(dataset.size(), globalsize);
  for (size_t i = 0; i < dataset.size(); ++i)
  {
    for (size_t j = 0; j < dataset[i].size(); ++j)
    {
      output(i, j) = mapping[(std::string(1, dataset[i][j]))];
    }
  }
}

template<typename eT>
void DicitonaryEncoding::DictEncode(const std::vector<std::string>& strings,
            arma::Mat<eT>& output,
            const char deliminator)
{
  DictEncode(strings, output, mappings, deliminator);
}


template<typename eT>
void DicitonaryEncoding::DictEncode(const std::vector<std::string>& strings,
            arma::Mat<eT>& output,
            std::unordered_map<std::string, size_t>& mapping,
            const char deliminator)
{
  std::vector< std::vector<std::string> > dataset;
  std::vector<std::string> temp;
  size_t globalsize = 0;
  for (size_t i = 0; i < strings.size(); ++i)
  {
    boost::split(temp, strings[i], boost::is_any_of(std::string(1,
          deliminator)));
    dataset.push_back(temp);
    globalsize = std::max(globalsize, temp.size());
    temp.clear();
  }
  size_t curLabel = 1;
  for (size_t i = 0; i < dataset.size(); ++i)
  {
    for (size_t j = 0; j < dataset[i].size(); ++j)
    {
      if (mapping.count(dataset[i][j]) == 0)
      {
        mapping[dataset[i][j]] = curLabel;
        ++curLabel;
      }
    }
  }
  output.zeros(dataset.size(), globalsize);
  for (size_t i = 0; i < dataset.size(); ++i)
  {
    for (size_t j = 0; j < dataset[i].size(); ++j)
    {
      output(i, j) = mapping[dataset[i][j]];
    }
  }
}
template<typename OutputColType>
void DicitonaryEncoding::DictEncode(const std::string& input,
                                    OutputColType&& output,
                                    std::unordered_map<std::string,
                                    size_t>& dictionary)
{
    std::vector<std::string> temp;
    boost::split(temp, input, boost::is_any_of(" "));
    size_t curLabel = dictionary.size() + 1;
    output.resize(temp.size());
    for (size_t i = 0; i < temp.size(); i++)
    {
      if (dictionary.count(temp[i]) == 0)
      {
        dictionary[temp[i]] = curLabel;
        curLabel++;
      }
      output(i) = dictionary[temp[i]];
    }
}

template<typename OutputColType>
void DicitonaryEncoding::DictEncode(const std::string& input,
                                    OutputColType&& output)
{
  DictEncode(input, output, mappings);
}

template<typename eT>
void DicitonaryEncoding::DictEncode(const std::vector<std::string>& strings,
            std::vector<arma::Row<eT>>& output,
            std::unordered_map<std::string,
            size_t>& mapping,
            const char deliminator)
{
  std::vector< std::vector<std::string> > dataset;
  std::vector<std::string> temp;
  size_t globalsize = 0;
  for (size_t i = 0; i < strings.size(); ++i)
  {
    boost::split(temp, strings[i], boost::is_any_of(std::string(1,
          deliminator)));
    dataset.push_back(temp);
    globalsize = std::max(globalsize, temp.size());
    temp.clear();
  }
  size_t curLabel = 1;
  for (size_t i = 0; i < dataset.size(); ++i)
  {
    for (size_t j = 0; j < dataset[i].size(); ++j)
    {
      if (mapping.count(dataset[i][j]) == 0)
      {
        mapping[dataset[i][j]] = curLabel;
        ++curLabel;
      }
    }
  }
  arma::Row<eT>row;
  for (size_t i = 0; i < dataset.size(); ++i)
  {
    row.zeros(dataset[i].size());
    for (size_t j = 0; j < dataset[i].size(); ++j)
    {
      row(j) = mapping[dataset[i][j]];
    }
    output.push_back(row);
  }
}

template<typename eT>
void DicitonaryEncoding::DictEncode(const std::vector<std::string>& strings,
          std::vector<arma::Row<eT>>& output, const char deliminator)
{
  DictEncode(strings, output, mappings, deliminator);
}

} // namespace data
} // namespace mlpack

#endif
