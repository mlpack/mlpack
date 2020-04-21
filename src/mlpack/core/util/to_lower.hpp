/**
 * @file to_lower.hpp
 * @author Himanshu Pathak
 *
 * Convert a string to lower-case.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_LOWER_STRING_HPP
#define MLPACK_CORE_UTIL_LOWER_STRING_HPP

namespace mlpack {
namespace util {

/**
 * ToLower convert a string into a string of lower case characters
 * only.
 *
 * @param str String to convert string.
 */
inline std::string ToLower(const std::string& str)
{
  std::string out = str;

  std::transform(str.begin(), str.end(), out.begin(),
      [](unsigned char c){ return std::tolower(c); });
  return out;
}

} // namespace util
} // namespace mlpack

#endif
