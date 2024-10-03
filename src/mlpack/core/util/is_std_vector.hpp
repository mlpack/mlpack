/**
 * @file core/util/is_std_vector.hpp
 * @author Ryan Curtin
 *
 * Simple template metaprogramming struct to detect if a type is a
 * std::vector<T>.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_IS_STD_VECTOR_HPP
#define MLPACK_CORE_UTIL_IS_STD_VECTOR_HPP

#include <vector>

namespace mlpack {
namespace util {

//! Metaprogramming structure for vector detection.
template<typename T>
struct IsStdVector { static const bool value = false; };

//! Metaprogramming structure for vector detection.
template<typename T, typename A>
struct IsStdVector<std::vector<T, A>> { static const bool value = true; };

} // namespace util
} // namespace mlpack

#endif
