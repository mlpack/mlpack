/**
 * @file bindings/go/print_defn_input.hpp
 * @author Yasmine Dumouchel
 *
 * Print the definition of an input in a binding .go file for a given
 * parameter.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_PRINT_DEFN_INPUT_HPP
#define MLPACK_BINDINGS_GO_PRINT_DEFN_INPUT_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/bindings/util/camel_case.hpp>
#include "get_go_type.hpp"
#include "strip_type.hpp"

namespace mlpack {
namespace bindings {
namespace go {

/**
 * Print input in method definition for a regular parameter type.
 */
template<typename T>
void PrintDefnInput(
    util::ParamData& d,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  if (d.required)
  {
    std::string name = d.name;
    std::cout << util::CamelCase(name, true) << " " << GetGoType<T>(d);
  }
}

/**
 * Print input in method definition for a matrix type.
 */
template<typename T>
void PrintDefnInput(
    util::ParamData& d,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
{
  // param_name *mat.Dense
  if (d.required)
  {
    std::string name = d.name;
    std::cout << util::CamelCase(name, true) << " *" << GetGoType<T>(d);
  }
}

/**
 * Print input in method definition for a matrix with info type.
 */
template<typename T>
void PrintDefnInput(
    util::ParamData& d,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  // param_name *DataWithInfo
  if (d.required)
  {
    std::string name = d.name;
    std::cout << util::CamelCase(name, true) << " *" << GetGoType<T>(d);
  }
}

/**
 * Print input in method definition for a serializable model.
 */
template<typename T>
void PrintDefnInput(
    util::ParamData& d,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  // Get the type names we need to use.
  std::string goStrippedType, strippedType, printedType, defaultsType;
  StripType(d.cppType, goStrippedType, strippedType, printedType, defaultsType);

  // param_name *ModelName
  if (d.required)
  {
    std::string name = d.name;
    std::cout << util::CamelCase(name, true) << " *" << goStrippedType;
  }
}

/**
 * Given parameter information and the current number of spaces for indentation,
 * print the code to process the output to cout.  This code assumes that
 * data.input is false, and should not be called when data.input is true.  If
 * this is the only output, the results will be different.
 *
 * The input pointer should be a pointer to a std::tuple<size_t, bool> where the
 * first element is the indentation and the second element is a boolean
 * representing whether or not this is the only output parameter.
 *
 * @param d Parameter data struct.
 * @param * (input) Pointer to size_t holding the indentation.
 * @param * (output) Unused parameter.
 */
template<typename T>
void PrintDefnInput(util::ParamData& d,
                    const void* /* input */,
                    void* /* output */)
{
  PrintDefnInput<typename std::remove_pointer<T>::type>(d);
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
