/**
 * @file string_util.hpp
 * @author Trironk Kiatkungwanglai
 * @author Ryan Birmingham
 *
 * Declares methods that are useful for writing formatting output.
 *
 * This file is part of mlpack 2.0.1.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_CORE_STRING_UTIL_HPP
#define __MLPACK_CORE_STRING_UTIL_HPP

#include <string>

namespace mlpack {
namespace util {

//! A utility function that replaces all all newlines with a number of spaces
//! depending on the indentation level.
std::string Indent(std::string input, const size_t howManyTabs = 1);

} // namespace util
} // namespace mlpack

#endif
