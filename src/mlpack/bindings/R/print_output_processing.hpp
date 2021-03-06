/**
 * @file bindings/R/print_output_processing.hpp
 * @author Yashwant Singh Parihar
 *
 * Print the output processing in a R binding .R file for a given
 * parameter.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_R_PRINT_OUTPUT_PROCESSING_HPP
#define MLPACK_BINDINGS_R_PRINT_OUTPUT_PROCESSING_HPP

#include <mlpack/prereqs.hpp>
#include "get_type.hpp"

namespace mlpack {
namespace bindings {
namespace r {

/**
 * Print output processing for a regular parameter type.
 */
template<typename T>
void PrintOutputProcessing(
    util::ParamData& d,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  /**
   * This gives us code like:
   *
   *     "<param_name>" = IO_GetParam<Type>("param_name")
   *
   */

  MLPACK_COUT_STREAM << "  \"" << d.name << "\" = IO_GetParam" << GetType<T>(d)
            << "(\"" << d.name << "\")";
}

/**
 * Print output processing for a matrix type.
 */
template<typename T>
void PrintOutputProcessing(
    util::ParamData& d,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  /**
   * This gives us code like:
   *
   *     "<param_name>" = IO_GetParam<Type>("param_name")
   *
   */

  MLPACK_COUT_STREAM << "  \"" << d.name << "\" = IO_GetParam" << GetType<T>(d)
            << "(\"" << d.name << "\")";
}

/**
 * Print output processing for a matrix with info type.
 */
template<typename T>
void PrintOutputProcessing(
    util::ParamData& d,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  /**
   * This gives us code like:
   *
   *     "<param_name>" = IO_GetParam<Type>("param_name")
   *
   */

  MLPACK_COUT_STREAM << "  \"" << d.name << "\" = IO_GetParam" << GetType<T>(d)
            << "(\"" << d.name << "\")";
}

/**
 * Print output processing for a serializable model.
 */
template<typename T>
void PrintOutputProcessing(
    util::ParamData& d,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  /**
   * This gives us code like:
   *
   *     "<param_name>" = <param_name>
   *
   */

  MLPACK_COUT_STREAM << "  \"" << d.name << "\" = " << d.name;
}

/**
 * @param d Parameter data struct.
 * @param * (input) Unused parameter.
 * @param * (output) Unused parameter.
 */
template<typename T>
void PrintOutputProcessing(util::ParamData& d,
                           const void* /*input*/,
                           void* /* output */)
{
  PrintOutputProcessing<typename std::remove_pointer<T>::type>(d);
}

} // namespace r
} // namespace bindings
} // namespace mlpack

#endif
