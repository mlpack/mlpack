/**
 * @file core/util/hyphenate_string.hpp
 * @author Ryan Curtin
 *
 * Hyphenate a string.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_HYPHENATE_STRING_HPP
#define MLPACK_CORE_UTIL_HYPHENATE_STRING_HPP

namespace mlpack {
namespace util {

/**
 * Hyphenate a string or split it onto multiple 80-character lines, with some
 * amount of padding on each line.  This is used for option output.
 *
 * @param str String to hyphenate (splits are on ' ').
 * @param prefix Prefix to hyphenate a string with.
 * @param force Hyphenate the string even if the length is less then 80.
 * @throw std::invalid_argument if prefix.size() >= 80.
 */
inline std::string HyphenateString(const std::string& str,
                                   const std::string& prefix,
                                   const bool force = false)
{
  if (prefix.size() >= 80)
  {
    throw std::invalid_argument("Prefix size must be less than 80");
  }

  size_t margin = 80 - prefix.size();
  if (str.length() < margin && !force)
    return str;
  std::string out("");
  unsigned int pos = 0;
  // First try to look as far as possible.
  while (pos < str.length())
  {
    size_t splitpos;
    // Check that we don't have a newline first.
    splitpos = str.find('\n', pos);
    if (splitpos == std::string::npos || splitpos > (pos + margin))
    {
      // We did not find a newline.
      if (str.length() - pos < margin)
      {
        splitpos = str.length(); // The rest fits on one line.
      }
      else
      {
        splitpos = str.rfind(' ', margin + pos); // Find nearest space.
        if (splitpos <= pos || splitpos == std::string::npos) // Not found.
          splitpos = pos + margin;
      }
    }
    out += str.substr(pos, (splitpos - pos));
    if (splitpos < str.length())
    {
      out += '\n';
      out += prefix;
    }

    pos = splitpos;
    if (str[pos] == ' ' || str[pos] == '\n')
      pos++;
  }
  return out;
}

/**
 * Hyphenate a string or split it onto multiple 80-character lines, with some
 * amount of padding on each line.  This is used for option output.
 *
 * @param str String to hyphenate (splits are on ' ').
 * @param padding Amount of padding on the left for each new line.
 */
inline std::string HyphenateString(const std::string& str, int padding)
{
  return HyphenateString(str, std::string(padding, ' '));
}

} // namespace util
} // namespace mlpack

#endif
