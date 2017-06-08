/**
 * @file get_param.hpp
 * @author Ryan Curtin
 *
 * Use template metaprogramming to get the right type of parameter.
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
  return *boost::any_cast<T>(&d.value);
}

/**
 * Return a parameter casted to the given type.  Type checking does not happen
 * here!
 *
 * @param d Parameter information.
 * @param input Unused parameter.
 * @param output Place to store pointer to value.
 */
template<typename T>
void GetParam(const util::ParamData& d, const void* /* input */, void* output)
{
  // Cast to the correct type.
  *((T**) output) = &GetParam<T>(const_cast<util::ParamData&>(d));
}

} // namespace tests
} // namespace bindings
} // namespace mlpack

#endif
