/**
 * @file dict_encoding_impl.hpp
 * @author Jeffin Sam
 *
 * Implementation of dictionary encoding functions;
 * 
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
#include <map>

namespace mlpack {
namespace data {

/**
 * Dictionary Encoding
 * here we simply assign a word (or a character) to a numeric index
 * and treat the dataset as categorical
 * 
 *
 * @param vector of documents.
 * @param mapping of string to their encoded number.
 * @param output matrix.
 */
template<typename eT>
void Encode(const std::vector<std::string>& strings,
            std::map<std::string, size_t>& mappings,
            arma::Mat<eT>& output)
{
  std::vector< std::vector<std::string> > dataset;
  std::vector<std::string> temp;
  size_t globalsize = 0;
  for (size_t i = 0; i < strings.size(); ++i)
  {
    boost::split(temp, strings[i], boost::is_any_of(" "));
    dataset.push_back(temp);
    globalsize = std::max(globalsize, temp.size());
    temp.clear();
  }
  size_t curLabel = 1;
  for (size_t i = 0; i < dataset.size(); ++i)
  {
    for (size_t j = 0; j < dataset[i].size(); ++j)
    {
      if (mappings[dataset[i][j]] == 0)
      {
        mappings[dataset[i][j]] = curLabel;
        ++curLabel;
      }
    }
  }
  output.set_size(dataset.size(), globalsize);
  output.fill(0);
  for (size_t i = 0; i < dataset.size(); ++i)
  {
    for (size_t j = 0; j < dataset[i].size(); ++j)
    {
      output(i, j) = mappings[dataset[i][j]];
    }
  }
}
} // namespace data
} // namespace mlpack

#endif
