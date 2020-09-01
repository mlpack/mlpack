/**
 * @file bindings/go/print_method_config.hpp
 * @author Yashwant Singh
 *
 * Print configuration struct for optional parameter type of a method for a
 * Go binding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_PRINT_METHOD_CONFIG_HPP
#define MLPACK_BINDINGS_GO_PRINT_METHOD_CONFIG_HPP

#include <mlpack/prereqs.hpp>
#include "get_go_type.hpp"
#include "strip_type.hpp"
#include <mlpack/bindings/util/camel_case.hpp>

namespace mlpack {
namespace bindings {
namespace go {

/**
 * Print param in configuration struct for a standard option type.
 */
template<typename T>
void PrintMethodConfig(
    util::ParamData& d,
    const size_t indent,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  const std::string prefix(indent, ' ');

  std::string def = "nil";
  if (std::is_same<T, bool>::value)
    def = "false";

  // Capitalize the first letter of parameter name so it is
  // of exported type in Go.
  std::string name = d.name;
  std::string goParamName = name;
  if (!name.empty())
  {
    goParamName = util::CamelCase(goParamName, false);
  }

  // Only print param that are not required.
  if (!d.required)
  {
    std::cout << prefix << goParamName << " " << GetGoType<T>(d)
              << std::endl;
  }
}

/**
 * Print param in configuration struct for a matrix type.
 */
template<typename T>
void PrintMethodConfig(
    util::ParamData& d,
    const size_t indent,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
{
  const std::string prefix(indent, ' ');

  std::string def = "nil";
  if (std::is_same<T, bool>::value)
    def = "false";

  // Capitalize the first letter of parameter name so it is
  // of exported type in Go.
  std::string name = d.name;
  std::string goParamName = name;
  if (!name.empty())
  {
    goParamName = util::CamelCase(goParamName, false);
  }

  // Only print param that are not required.
  if (!d.required)
  {
    std::cout << prefix << goParamName << " *" << GetGoType<T>(d)
              << std::endl;
  }
}

/**
 * Print param in configuration struct for a matrix with info type.
 */
template<typename T>
void PrintMethodConfig(
    util::ParamData& d,
    const size_t indent,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  const std::string prefix(indent, ' ');

  std::string def = "nil";
  if (std::is_same<T, bool>::value)
    def = "false";

  // Capitalize the first letter of parameter name so it is
  // of exported type in Go.
  std::string name = d.name;
  std::string goParamName = name;
  if (!name.empty())
  {
    goParamName = util::CamelCase(goParamName, false);
  }

  // Only print param that are not required.
  if (!d.required)
  {
    std::cout << prefix << goParamName << " *" << GetGoType<T>(d)
              << std::endl;
  }
}

/**
 *  Print param in method configuration struct for a serializable type.
 */
template<typename T>
void PrintMethodConfig(
    util::ParamData& d,
    const size_t indent,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  const std::string prefix(indent, ' ');

  std::string def = "nil";
  if (std::is_same<T, bool>::value)
    def = "false";

  // Capitalize the first letter of parameter name so it is
  // of exported type in Go.
  std::string name = d.name;
  std::string goParamName = name;
  if (!name.empty())
  {
    goParamName = util::CamelCase(goParamName, false);
  }

  // Only print param that are not required.
  if (!d.required)
  {
    std::cout << prefix << goParamName << " *" << GetGoType<T>(d)
              << std::endl;
  }
}

/**
 * Given parameter information and the current number of spaces for indentation,
 * print the code to process the input to cout.  This code assumes that
 * data.input is true, and should not be called when data.input is false.
 *
 * The number of spaces to indent should be passed through the input pointer.
 *
 * @param d Parameter data struct.
 * @param input Pointer to size_t holding the indentation.
 * @param * (output) Unused parameter.
 */
template<typename T>
void PrintMethodConfig(util::ParamData& d,
                       const void* input,
                       void* /* output */)
{
  PrintMethodConfig<typename std::remove_pointer<T>::type>(d,
      *((size_t*) input));
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
