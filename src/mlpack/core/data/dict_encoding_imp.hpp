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
void Encode(const std::vector<std::string>& strings,
            std::map<std::string, size_t>& mappings,
            arma::Mat<eT>& output)
{
  std::vector< std::vector<string> > dataset;
  std::std::vector<string> temp;
  size_t globalsize = -1;
  for(size_t i = 0; i < string.size(); ++i)
  {
    boost::split(temp,strings[i],boost::is_any_of(" "));
    dataset.push_back(temp);
    globalsize = max(globalsize,temp.size());
    temp.clear();
  }
  size_t curLabel = 0;
  for (size_t i = 0; i < dataset.size(); ++i)
  {
    for (size_t j = 0; j < dataset[i].size; ++j)
    {
      if(mapping[dataset[i][j]] == 0)
      {
        mapping[dataset[i][j]] = curLabel;
        ++curLabel;

      }
    }
  }
  output.set_size(globalsize, globalsize);
  output.fill(0);
  for (size_t i = 0; i < dataset.size(); ++i)
  {
    for (size_t j = 0; j < dataset[i].size; ++j)
    {
      output(i,j) = mapping[dataset[i][j]];
    }
  } 
}
} // namespace data
} // namespace mlpack

#endif
