/**
 * @file print_jl.hpp
 * @author Ryan Curtin
 *
 * Definition of utility PrintJL() function.
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
void PrintJL(const util::ProgramDoc& programInfo,
             const std::string& functionName,
             const std::string& mlpackJuliaLibSuffix);

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif
