/**
 * @file test_helper.hpp
 * @author Eugene Freyman
 *
 * Helper functions for testing.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_TESTS_MAIN_TESTS_TEST_HELPER_HPP
#define MLPACK_TESTS_MAIN_TESTS_TEST_HELPER_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace util {

/**
 * Utility function that is used in binding tests for setting a parameter and
 * marking it as passed; it uses copy semantics for lvalues and move semantics
 * for rvalues.
 *
 * @param name Name of parameter to set.
 * @param value Value to set parameter to.
 */
template<typename T>
void SetInputParam(const std::string& name, T&& value)
{
  CLI::GetParam<typename std::remove_reference<T>::type>(name) =
    std::forward<T>(value);
  CLI::SetPassed(name);
}

} // namespace util
} // namespace mlpack

#endif
