/**
 * @file is_std_vector.hpp
 * @author Ryan Curtin
 *
 * Simple template metaprogramming struct to detect if a type is a
 * std::vector<T>.
 */
#ifndef MLPACK_CORE_UTIL_IS_STD_VECTOR_HPP
#define MLPACK_CORE_UTIL_IS_STD_VECTOR_HPP

#include <vector>

namespace mlpack {
namespace util {

//! Metaprogramming structure for vector detection.
template<typename T>
struct IsStdVector { const static bool value = false; };

//! Metaprogramming structure for vector detection.
template<typename T, typename A>
struct IsStdVector<std::vector<T, A>> { const static bool value = true; };

} // namespace util
} // namespace mlpack

#endif
