/**
 * @file print_input_param.hpp
 * @author Ryan Curtin
 *
 * Print the declaration of an input parameter as part of a line in a Julia
 * function definition.
 */
#ifndef MLPACK_BINDINGS_JULIA_PRINT_INPUT_PARAM_HPP
#define MLPACK_BINDINGS_JULIA_PRINT_INPUT_PARAM_HPP

#include "get_julia_type.hpp"

namespace mlpack {
namespace bindings {
namespace julia {

/**
 * Print the declaration of an input parameter as part of a line in a Julia
 * function definition.  This doesn't include any commas or anything.
 */
template<typename T>
void PrintInputParam(const util::ParamData& d,
                     const void* /* input */,
                     void* /* output */)
{
  std::cout << d.name << "::";
  // If it's required, then we need the type.
  if (d.required)
  {
    std::cout << GetJuliaType<typename std::remove_pointer<T>::type>();
  }
  else
  {
    std::cout << "Union{"
        << GetJuliaType<typename std::remove_pointer<T>::type>()
        << ", Missing} = missing";
  }
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif
