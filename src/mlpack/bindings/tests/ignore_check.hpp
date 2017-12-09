/**
 * @file ignore_check.hpp
 * @author Ryan Curtin
 *
 * Implementation of IgnoreCheck() for Python bindings.
 */
#ifndef MLPACK_BINDINGS_TEST_IGNORE_CHECK_HPP
#define MLPACK_BINDINGS_TEST_IGNORE_CHECK_HPP

namespace mlpack {
namespace bindings {
namespace tests {

/**
 * Return whether or not a parameter check should be ignored.  For test
 * bindings, we do not ignore any checks, so this always returns false.
 */
template<typename T>
inline bool IgnoreCheck(const T& /* t */) { return false; }

} // namespace tests
} // namespace bindings
} // namespace mlpack

#endif
