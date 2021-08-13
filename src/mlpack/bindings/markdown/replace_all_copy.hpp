/**
 * @file bindings/markdown/replace_all_copy.hpp
 * @author Nippun Sharma
 *
 * Replace a substring in a string.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_MARKDOWN_REPLACE_ALL_COPY_HPP
#define MLPACK_BINDINGS_MARKDOWN_REPLACE_ALL_COPY_HPP

#include <mlpack/prereqs.hpp>

// Replaces all occurences of "from" in "str" to "to".
inline std::string replace_all_copy(const std::string& str,
                                    const std::string& from,
                                    const std::string& to)
{
  std::string str_copy = str;
  size_t start_pos = 0;
  while ((start_pos = str_copy.find(from, start_pos)) != std::string::npos)
  {
    str_copy.replace(start_pos, from.length(), to);
    start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
  }
  return str_copy;
}

#endif
