/**
 * @file bindings/tests/get_param.hpp
 * @author Ryan Curtin
 *
 * Use template metaprogramming to get the right type of parameter.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_TESTS_GET_PARAM_HPP
#define MLPACK_BINDINGS_TESTS_GET_PARAM_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace tests {

/**
 * This overload is called when nothing special needs to happen to the name of
 * the parameter.
 */
template<typename T>
T& GetParam(util::ParamData& d)
{
  // No mapping is needed, so just cast it directly.
  return *MLPACK_ANY_CAST<T>(&d.value);
}

/**
 * Return a parameter casted to the given type.  Type checking does not happen
 * here!
 *
 * @param d Parameter information.
 * @param * (input) Unused parameter.
 * @param output Place to store pointer to value.
 */
template<typename T>
void GetParam(util::ParamData& d, const void* /* input */, void* output)
{
  // Cast to the correct type.
  *((T**) output) = &GetParam<T>(const_cast<util::ParamData&>(d));
}

} // namespace tests
} // namespace bindings
} // namespace mlpack

#endif
