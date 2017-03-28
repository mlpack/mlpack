/**
 * @file is_std_vector.hpp
 * @author Ryan Curtin
 *
 * Simple template metaprogramming struct to detect if a type is a
 * std::vector<T>.
 */
#ifndef MLPACK_BINDINGS_TESTS_IS_STD_VECTOR_HPP
#define MLPACK_BINDINGS_TESTS_IS_STD_VECTOR_HPP

#include <vector>

namespace mlpack {
namespace bindings {
namespace tests {

//! Metaprogramming structure for vector detection.
template<typename T>
struct IsStdVector { const static bool value = false; };

//! Metaprogramming structure for vector detection.
template<typename T, typename A>
struct IsStdVector<std::vector<T, A>> { const static bool value = true; };

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
