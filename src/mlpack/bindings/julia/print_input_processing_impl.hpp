/**
 * @file bindings/julia/print_input_processing_impl.hpp
 * @author Ryan Curtin
 *
 * Print Julia code to handle input arguments.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_JULIA_PRINT_INPUT_PROCESSING_IMPL_HPP
#define MLPACK_BINDINGS_JULIA_PRINT_INPUT_PROCESSING_IMPL_HPP

#include <mlpack/bindings/util/strip_type.hpp>
#include "get_julia_type.hpp"

namespace mlpack {
namespace bindings {
namespace julia {

/**
 * Print the input processing (basically calling IO::GetParam<>()) for a
 * non-serializable type.
 */
template<typename T>
void PrintInputProcessing(
    util::ParamData& d,
    const std::string& /* functionName */,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type*,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  // "type" is a reserved keyword or function.
  const std::string juliaName = (d.name == "type") ? "type_" : d.name;

  // Here we can just call IOSetParam() directly; we don't need a separate
  // overload.
  if (d.required)
  {
    // This gives us code like the following:
    //
    // IOSetParam("<param_name>", <paramName>)
    std::cout << "  IOSetParam(\"" << d.name << "\", " << juliaName << ")"
        << std::endl;
  }
  else
  {
    // This gives us code like the following:
    //
    // if !ismissing(<param_name>)
    //   IOSetParam("<param_name>", convert(<type>, <param_name>))
    // end
    std::cout << "  if !ismissing(" << juliaName << ")" << std::endl;
    std::cout << "    IOSetParam(\"" << d.name << "\", convert("
        << GetJuliaType<T>(d) << ", " << juliaName << "))" << std::endl;
    std::cout << "  end" << std::endl;
  }
}

/**
 * Print the input processing for an Armadillo type.
 */
template<typename T>
void PrintInputProcessing(
    util::ParamData& d,
    const std::string& /* functionName */,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type*,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  // "type" is a reserved keyword or function.
  const std::string juliaName = (d.name == "type") ? "type_" : d.name;

  // If the argument is not required, then we have to encase the code in an if.
  size_t extraIndent = 0;
  if (!d.required)
  {
    std::cout << "  if !ismissing(" << juliaName << ")" << std::endl;
    extraIndent = 2;
  }

  // For an Armadillo type, we have to call a different overload for columns and
  // rows than for regular matrices.
  std::string uChar = (std::is_same<typename T::elem_type, size_t>::value) ?
      "U" : "";
  std::string indent(extraIndent + 2, ' ');
  std::string matTypeModifier = "";
  std::string extra = "";
  if (T::is_row)
  {
    matTypeModifier = "Row";
  }
  else if (T::is_col)
  {
    matTypeModifier = "Col";
  }
  else
  {
    matTypeModifier = "Mat";
    extra = ", points_are_rows";
  }

  // Now print the IOSetParam call.
  std::cout << indent << "IOSetParam" << uChar << matTypeModifier << "(\""
      << d.name << "\", " << juliaName << extra << ")" << std::endl;

  if (!d.required)
  {
    std::cout << "  end" << std::endl;
  }
}

/**
 * Print the input processing for a serializable type.
 */
template<typename T>
void PrintInputProcessing(
    util::ParamData& d,
    const std::string& functionName,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type*,
    const typename std::enable_if<data::HasSerialize<T>::value>::type*,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  // "type" is a reserved keyword or function.
  const std::string juliaName = (d.name == "type") ? "type_" : d.name;

  // If the argument is not required, then we have to encase the code in an if.
  size_t extraIndent = 0;
  if (!d.required)
  {
    std::cout << "  if !ismissing(" << juliaName << ")" << std::endl;
    extraIndent = 2;
  }

  std::string indent(extraIndent + 2, ' ');
  std::string type = util::StripType(d.cppType);
  std::cout << indent << functionName << "_internal.IOSetParam" << type
      << "(\"" << d.name << "\", convert("
      << GetJuliaType<typename std::remove_pointer<T>::type>(d) << ", "
      << juliaName << "))" << std::endl;

  if (!d.required)
  {
    std::cout << "  end" << std::endl;
  }
}

/**
 * Print the input processing (basically calling IO::GetParam<>()) for a
 * matrix with DatasetInfo type.
 */
template<typename T>
void PrintInputProcessing(
    util::ParamData& d,
    const std::string& /* functionName */,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  // "type" is a reserved keyword or function.
  const std::string juliaName = (d.name == "type") ? "type_" : d.name;

  // Here we can just call IOSetParam() directly; we don't need a separate
  // overload.  But we do have to pass in points_are_rows.
  if (d.required)
  {
    // This gives us code like the following:
    //
    // IOSetParam("<param_name>", convert(<type>, <paramName>))
    std::cout << "  IOSetParam(\"" << d.name << "\", convert("
        << GetJuliaType<T>(d) << ", " << juliaName << "), points_are_rows)"
        << std::endl;
  }
  else
  {
    // This gives us code like the following:
    //
    // if !ismissing(<param_name>)
    //   IOSetParam("<param_name>", convert(<type>, <param_name>))
    // end
    std::cout << "  if !ismissing(" << juliaName << ")" << std::endl;
    std::cout << "    IOSetParam(\"" << d.name << "\", convert("
        << GetJuliaType<T>(d) << ", " << juliaName << "), points_are_rows)"
        << std::endl;
    std::cout << "  end" << std::endl;
  }
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif
