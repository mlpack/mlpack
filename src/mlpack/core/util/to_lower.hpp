/**
 * @file core/util/to_lower.hpp
 * @author Himanshu Pathak
 *
 * Convert a string to lowercase.
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
 * Convert a string to lowercase letters.
 *
 * @param input The string to convert.
 */
inline std::string ToLower(const std::string& input)
{
  std::string output;
  std::transform(input.begin(), input.end(), std::back_inserter(output),
      [](unsigned char c){ return std::tolower(c); });
  return output;
}

} // namespace util
} // namespace mlpack

#endif
