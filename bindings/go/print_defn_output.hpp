/**
 * @file bindings/go/print_defn_output.hpp
 * @author Yasmine Dumouchel
 *
 * Print the output for the definition in a Go binding .go file for a given
 * parameter.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_PRINT_DEFN_OUTPUT_HPP
#define MLPACK_BINDINGS_GO_PRINT_DEFN_OUTPUT_HPP

#include <mlpack/prereqs.hpp>
#include "get_go_type.hpp"
#include "strip_type.hpp"

namespace mlpack {
namespace bindings {
namespace go {

/**
 * Print output for a regular parameter type.
 */
template<typename T>
void PrintDefnOutput(
    util::ParamData& d,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  std::cout << GetGoType<T>(d);
}

/**
 * Print output for a matrix type.
 */
template<typename T>
void PrintDefnOutput(
    util::ParamData& d,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
{
  // *mat.Dense
  std::cout << "*" << GetGoType<T>(d);
}

/**
 * Print output for a matrix with info type.
 */
template<typename T>
void PrintDefnOutput(
    util::ParamData& d,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  // *mat.Dense
  std::cout << "*" << GetGoType<T>(d);
}

/**
 * Print output for a serializable model.
 */
template<typename T>
void PrintDefnOutput(
    util::ParamData& d,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0)
{
  // Get the type names we need to use.
  std::string goStrippedType, strippedType, printedType, defaultsType;
  StripType(d.cppType, goStrippedType, strippedType, printedType, defaultsType);
  std::cout << goStrippedType;
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
void PrintDefnOutput(util::ParamData& d,
                     const void* /* input */,
                     void* /* output */)
{
  PrintDefnOutput<typename std::remove_pointer<T>::type>(d);
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
