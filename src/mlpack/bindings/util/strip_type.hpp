/**
 * @file bindings/util/strip_type.hpp
 * @author Ryan Curtin
 *
 * Given a C++ type name, turn it into something that has no special characters
 * that can simply be printed.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_UTIL_STRIP_TYPE_HPP
#define MLPACK_BINDINGS_UTIL_STRIP_TYPE_HPP

namespace mlpack {
namespace util {

/**
 * Given a C++ type name, turn it into something that has no special characters
 * that can simply be printed.  This is similar to but not identical to
 * mlpack::util::StripType().
 *
 * @param cppType C++ type as a string.
 * @return Stripped type with no special characters.
 */
inline std::string StripType(std::string cppType)
{
  // Basically what we need to do is strip any '<' (template bits) from the
  // type.  We'll try first by removing any instances of <>.
  const size_t loc = cppType.find("<>");
  if (loc != std::string::npos)
    cppType.replace(loc, 2, "");

  // Let's just replace any invalid characters with valid '_' characters.
  std::replace(cppType.begin(), cppType.end(), '<', '_');
  std::replace(cppType.begin(), cppType.end(), '>', '_');
  std::replace(cppType.begin(), cppType.end(), ' ', '_');
  std::replace(cppType.begin(), cppType.end(), ',', '_');

  return cppType;
}

} // namespace util
} // namespace mlpack

#endif
