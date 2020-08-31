/**
 * @file bindings/R/print_type_doc.hpp
 * @author Yashwant Singh Parihar
 *
 * Print documentation for a given type, detailing what the type actually is to
 * the user.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_R_PRINT_TYPE_DOC_HPP
#define MLPACK_BINDINGS_R_PRINT_TYPE_DOC_HPP

#include <mlpack/core/util/is_std_vector.hpp>

namespace mlpack {
namespace bindings {
namespace r {

/**
 * Return a string representing the command-line type of an option.
 */
template<typename T>
std::string PrintTypeDoc(
    util::ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0);

/**
 * Return a string representing the command-line type of a vector.
 */
template<typename T>
std::string PrintTypeDoc(
    util::ParamData& data,
    const typename std::enable_if<util::IsStdVector<T>::value>::type* = 0);

/**
 * Return a string representing the command-line type of a matrix option.
 */
template<typename T>
std::string PrintTypeDoc(
    util::ParamData& data,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0);

/**
 * Return a string representing the command-line type of a matrix tuple option.
 */
template<typename T>
std::string PrintTypeDoc(
    util::ParamData& data,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);

/**
 * Return a string representing the command-line type of a model.
 */
template<typename T>
std::string PrintTypeDoc(
    util::ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0);

/**
 * Print the command-line type of an option into a string.
 */
template<typename T>
void PrintTypeDoc(util::ParamData& data,
                  const void* /* input */,
                  void* output)
{
  *((std::string*) output) =
      PrintTypeDoc<typename std::remove_pointer<T>::type>(data);
}

} // namespace r
} // namespace bindings
} // namespace mlpack

#include "print_type_doc_impl.hpp"

#endif
