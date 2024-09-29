/**
 * @file bindings/python/get_printable_type.hpp
 * @author Ryan Curtin
 *
 * Template metaprogramming to return the string representation of the Python
 * type for a given Python binding parameter.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_GET_PRINTABLE_TYPE_HPP
#define MLPACK_BINDINGS_PYTHON_GET_PRINTABLE_TYPE_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/is_std_vector.hpp>

namespace mlpack {
namespace bindings {
namespace python {

template<typename T>
inline std::string GetPrintableType(
    util::ParamData& /* d */,
    const typename std::enable_if<!util::IsStdVector<T>::value>::type* = 0,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);

template<>
inline std::string GetPrintableType<int>(
    util::ParamData& /* d */,
    const typename std::enable_if<!util::IsStdVector<int>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<int>::value>::type*,
    const typename std::enable_if<!arma::is_arma_type<int>::value>::type*,
    const typename std::enable_if<!std::is_same<int,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*);

template<>
inline std::string GetPrintableType<double>(
    util::ParamData& /* d */,
    const typename std::enable_if<!util::IsStdVector<double>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<double>::value>::type*,
    const typename std::enable_if<!arma::is_arma_type<double>::value>::type*,
    const typename std::enable_if<!std::is_same<double,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*);

template<>
inline std::string GetPrintableType<std::string>(
    util::ParamData& /* d */,
    const typename std::enable_if<
        !util::IsStdVector<std::string>::value>::type*,
    const typename std::enable_if<
        !data::HasSerialize<std::string>::value>::type*,
    const typename std::enable_if<
        !arma::is_arma_type<std::string>::value>::type*,
    const typename std::enable_if<!std::is_same<std::string,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*);

template<>
inline std::string GetPrintableType<size_t>(
    util::ParamData& /* d */,
    const typename std::enable_if<!util::IsStdVector<size_t>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<size_t>::value>::type*,
    const typename std::enable_if<!arma::is_arma_type<size_t>::value>::type*,
    const typename std::enable_if<!std::is_same<size_t,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*);

template<>
inline std::string GetPrintableType<bool>(
    util::ParamData& /* d */,
    const typename std::enable_if<!util::IsStdVector<bool>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<bool>::value>::type*,
    const typename std::enable_if<!arma::is_arma_type<bool>::value>::type*,
    const typename std::enable_if<!std::is_same<bool,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*);

template<typename T>
inline std::string GetPrintableType(
    util::ParamData& d,
    const typename std::enable_if<util::IsStdVector<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);

template<typename T>
inline std::string GetPrintableType(
    util::ParamData& /* d */,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);

template<typename T>
inline std::string GetPrintableType(
    util::ParamData& /* d */,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);

template<typename T>
inline std::string GetPrintableType(
    util::ParamData& d,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);

template<typename T>
void GetPrintableType(util::ParamData& d,
                      const void* /* input */,
                      void* output)
{
  *((std::string*) output) =
      GetPrintableType<typename std::remove_pointer<T>::type>(d);
}

} // namespace python
} // namespace bindings
} // namespace mlpack

#include "get_printable_type_impl.hpp"

#endif
