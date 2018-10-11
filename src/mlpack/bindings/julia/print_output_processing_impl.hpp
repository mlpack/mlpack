/**
 * @file print_output_processing_impl.hpp
 * @author Ryan Curtin
 *
 * Print Julia code to handle output arguments.
 */
#ifndef MLPACK_BINDINGS_JULIA_PRINT_OUTPUT_PROCESSING_IMPL_HPP
#define MLPACK_BINDINGS_JULIA_PRINT_OUTPUT_PROCESSING_IMPL_HPP

#include "print_output_processing.hpp"

#include "strip_type.hpp"
#include "get_julia_type.hpp"

namespace mlpack {
namespace bindings {
namespace julia {

/**
 * Print the output processing (basically calling CLI::GetParam<>()) for a
 * non-serializable type.
 */
template<typename T>
void PrintOutputProcessing(
    const util::ParamData& d,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type*)
{
  std::cout << "CLIGetParam(\"" << d.name << "\")";
}

/**
 * Print the output processing for an Armadillo type.
 */
template<typename T>
void PrintOutputProcessing(
    const util::ParamData& d,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type*)
{
  std::string uChar = (std::is_same<T, size_t>::value) ? "U" : "";
  std::string matTypeSuffix = "";
  if (T::is_row)
    matTypeSuffix = "Row";
  else if (T::is_col)
    matTypeSuffix = "Col";

  std::cout << "CLIGetParam" << uChar << matTypeSuffix << "(\"" << d.name
      << "\")";
}

/**
 * Print the output processing for a serializable type.
 */
template<typename T>
void PrintOutputProcessing(
    const util::ParamData& d,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type*,
    const typename std::enable_if<data::HasSerialize<T>::value>::type*)
{
  std::cout << "CLIGetParam" << StripType(d.cppType) << "Ptr(\"" << d.name
      << "\")";
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif
