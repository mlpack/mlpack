/**
 * @file bindings/julia/print_jl.hpp
 * @author Ryan Curtin
 *
 * Definition of utility PrintJL() function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_JULIA_PRINT_JL_HPP
#define MLPACK_BINDINGS_JULIA_PRINT_JL_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace bindings {
namespace julia {

/**
 * Print the code for a .jl binding for an mlpack program to stdout.
 */
void PrintJL(const util::BindingDetails& doc,
             const std::string& functionName,
             const std::string& mlpackJuliaLibSuffix);

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif
