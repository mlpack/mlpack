/**
 * @file bindings/cli/get_raw_param.hpp
 * @author Ryan Curtin
 *
 * Use template metaprogramming to get the right type of parameter, but without
 * any processing.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CLI_GET_RAW_PARAM_HPP
#define MLPACK_BINDINGS_CLI_GET_RAW_PARAM_HPP

#include <mlpack/prereqs.hpp>
#include "parameter_type.hpp"

namespace mlpack {
namespace bindings {
namespace cli {

/**
 * This overload is called when nothing special needs to happen to the name of
 * the parameter.
 */
template<typename T>
T& GetRawParam(
    util::ParamData& d,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<mlpack::data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  // No mapping is needed, so just cast it directly.
  return *MLPACK_ANY_CAST<T>(&d.value);
}

/**
 * Return a matrix parameter.
 */
template<typename T>
T& GetRawParam(
    util::ParamData& d,
    const typename std::enable_if<
        arma::is_arma_type<T>::value ||
        std::is_same<T, std::tuple<mlpack::data::DatasetInfo,
                                   arma::mat>>::value>::type* = 0)
{
  // Don't load the matrix.
  typedef std::tuple<T, std::tuple<std::string, size_t, size_t>> TupleType;
  T& value = std::get<0>(*MLPACK_ANY_CAST<TupleType>(&d.value));
  return value;
}

/**
 * Return the name of a model parameter.
 */
template<typename T>
T*& GetRawParam(
    util::ParamData& d,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0)
{
  // Don't load the model.
  typedef std::tuple<T*, std::string> TupleType;
  T*& value = std::get<0>(*MLPACK_ANY_CAST<TupleType>(&d.value));
  return value;
}

/**
 * Return a parameter casted to the given type.  Type checking does not happen
 * here!
 *
 * @param d Parameter information.
 * @param * (input) Unused parameter.
 * @param output Place to store pointer to value.
 */
template<typename T>
void GetRawParam(util::ParamData& d,
                 const void* /* input */,
                 void* output)
{
  // Cast to the correct type.
  *((T**) output) = &GetRawParam<typename std::remove_pointer<T>::type>(
      const_cast<util::ParamData&>(d));
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
