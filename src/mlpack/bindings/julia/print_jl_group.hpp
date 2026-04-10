/**
 * @file bindings/julia/print_jl_group.hpp
 * @author Ryan Curtin
 *
 * Definition of utility PrintJLGroup() function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_JULIA_PRINT_JL_GROUP_HPP
#define MLPACK_BINDINGS_JULIA_PRINT_JL_GROUP_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace bindings {
namespace julia {

/**
 * Print the code for a .jl file that groups several bindings into a single
 * class.
 */
void PrintJLGroup(const std::string& category,
                  const std::string& groupName,
                  const std::string& validGroupMethods);

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif
