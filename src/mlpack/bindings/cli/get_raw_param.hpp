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
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0,
    const std::enable_if_t<!data::HasSerialize<T>::value>* = 0,
    const std::enable_if_t<!std::is_same_v<T,
        std::tuple<mlpack::data::DatasetInfo, arma::mat>>>* = 0)
{
  // No mapping is needed, so just cast it directly.
  return *std::any_cast<T>(&d.value);
}

/**
 * Return a matrix parameter.
 */
template<typename T>
T& GetRawParam(
    util::ParamData& d,
    const std::enable_if_t<
        arma::is_arma_type<T>::value ||
        std::is_same_v<T, std::tuple<mlpack::data::DatasetInfo,
                                     arma::mat>>>* = 0)
{
  // Don't load the matrix.
  using TupleType = std::tuple<T, std::tuple<std::string, size_t, size_t>>;
  T& value = std::get<0>(*std::any_cast<TupleType>(&d.value));
  return value;
}

/**
 * Return the name of a model parameter.
 */
template<typename T>
T*& GetRawParam(
    util::ParamData& d,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0,
    const std::enable_if_t<data::HasSerialize<T>::value>* = 0)
{
  // Don't load the model.
  using TupleType = std::tuple<T*, std::string>;
  T*& value = std::get<0>(*std::any_cast<TupleType>(&d.value));
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
  *((T**) output) = &GetRawParam<std::remove_pointer_t<T>>(
      const_cast<util::ParamData&>(d));
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
