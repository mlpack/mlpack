/**
 * @file bindings/julia/print_member_defn.hpp
 * @author Ryan Curtin
 *
 * Print the declaration of an input parameter in a Julia struct definition.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_JULIA_PRINT_MEMBER_DEFN_HPP
#define MLPACK_BINDINGS_JULIA_PRINT_MEMBER_DEFN_HPP

#include "get_julia_type.hpp"

namespace mlpack {
namespace bindings {
namespace julia {

/**
 * Print the declaration of an input parameter as part of a line in a Julia
 * function definition.  This doesn't include any commas or anything.
 */
template<typename T>
void PrintMemberDefn(util::ParamData& d,
                     const void* /* input */,
                     void* /* output */)
{
  // "type" is a reserved keyword or function.
  const std::string juliaName = (d.name == "type") ? "type_" : d.name;

  std::cout << juliaName;

  if (!arma::is_arma_type<T>::value)
  {
    std::cout << "::";
    std::cout << GetJuliaType<std::remove_pointer_t<T>>(d);
  }
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif
