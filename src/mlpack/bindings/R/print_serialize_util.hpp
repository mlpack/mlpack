/**
 * @file bindings/R/print_serialize_util.hpp
 * @author Yashwant Singh Parihar
 *
 * Print the serialize utility in a R binding .R file for a given
 * parameter.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_R_PRINT_SERIALIZE_UTIL_HPP
#define MLPACK_BINDINGS_R_PRINT_SERIALIZE_UTIL_HPP

#include <mlpack/bindings/util/strip_type.hpp>

namespace mlpack {
namespace bindings {
namespace r {

/**
 * If the type is not serializable, print nothing.
 */
template<typename T>
void PrintSerializeUtil(
    util::ParamData& /* d */,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0)
{
  // Do Nothing.
}

/**
 * Matrices are serializable but here we also print nothing.
 */
template<typename T>
void PrintSerializeUtil(
    util::ParamData& /* d */,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
{
  // Do Nothing.
}

/**
 * For non-matrix serializable types we need to print something.
 */
template<typename T>
void PrintSerializeUtil(
    util::ParamData& d,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0)
{
  /**
   * This gives us code like:
   *
   *     <param_name> <- IO_GetParam<ModelType>Ptr("<param_name>")
   *     attr(<param_name>, "type") <- "<ModelType>"
   *
   */
  MLPACK_COUT_STREAM << "  " << d.name << " <- IO_GetParam"
      << util::StripType(d.cppType) << "Ptr(\"" << d.name << "\")";
  MLPACK_COUT_STREAM << std::endl;
  MLPACK_COUT_STREAM << "  attr(" << d.name << ", \"type\") <- \""
      << util::StripType(d.cppType) << "\"";
  MLPACK_COUT_STREAM << std::endl;
}

/**
 * @param d Parameter data struct.
 * @param * (input) Unused parameter.
 * @param * (output) Unused parameter.
 */
template<typename T>
void PrintSerializeUtil(util::ParamData& d,
                        const void* /*input*/,
                        void* /* output */)
{
  PrintSerializeUtil<typename std::remove_pointer<T>::type>(d);
}

} // namespace r
} // namespace bindings
} // namespace mlpack

#endif
