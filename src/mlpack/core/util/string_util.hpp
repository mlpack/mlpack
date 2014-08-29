/**
 * @file string_util.hpp
 * @author Trironk Kiatkungwanglai
 * @author Ryan Birmingham
 *
 * Declares methods that are useful for writing formatting output.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_STRING_UTIL_HPP
#define __MLPACK_CORE_STRING_UTIL_HPP

#include <string>

namespace mlpack {
namespace util {

//! A utility function that replaces all all newlines with a number of spaces
//! depending on the indentation level.
std::string Indent(std::string input, const size_t howManyTabs = 1);

}; // namespace util
}; // namespace mlpack

#endif
