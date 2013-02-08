/**
 * @file string_util.hpp
 *
 * Declares methods that are useful for writing formatting output.
 */
#ifndef __MLPACK_CORE_STRING_UTIL_HPP
#define __MLPACK_CORE_STRING_UTIL_HPP

#include <string>

namespace mlpack {
namespace util {

//! A utility function that replaces all all newlines with a number of spaces
//! depending on the indentation level.
std::string Indent(std::string input);

}; // namespace util
}; // namespace mlpack

#endif
