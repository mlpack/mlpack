/** 
 * @file core/data/string_algorithms.hpp
 * @author Gopi M. Tatiraju
 *
 * Utility fucntions related to string manipulation
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DATA_STRING_ALGORITHMS_HPP
#define MLPACK_CORE_DATA_STRING_ALGORITHMS_HPP

namespace mlpack{
namespace data{

/**
 * A simple trim fucntion to strip off whitespaces 
 * from both the side of string.
 */
inline void trim(std::string& str)
{
  if(str.size() < 2)
  {
    str = "";
    return;
  }

  size_t startIndex = 0;

  while(std::isspace(str[startIndex]))
  {
    startIndex++;
  }

  size_t endIndex = str.size() - 1;

  while(std::isspace(str[endIndex]))
  {
    endIndex--;
  }

  std::string trimmedStr = (endIndex - startIndex == str.size()) ? 
                             std::move(str) : str.substr(startIndex, endIndex - startIndex + 1);

  str = trimmedStr;
}
    
}  // namespace data
}  // namespace mlpack

#endif

  
