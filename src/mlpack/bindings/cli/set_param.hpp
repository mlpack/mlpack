/**
 * @file bindings/cli/set_param.hpp
 * @author Ryan Curtin
 *
 * Use template metaprogramming to get the right type of parameter.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CLI_SET_PARAM_HPP
#define MLPACK_BINDINGS_CLI_SET_PARAM_HPP

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
void SetParam(
    util::ParamData& d,
    const std::any& value,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0,
    const std::enable_if_t<!data::HasSerialize<T>::value>* = 0,
    const std::enable_if_t<!std::is_same_v<T,
        std::tuple<mlpack::data::DatasetInfo, arma::mat>>>* = 0,
    const std::enable_if_t<!std::is_same_v<T, bool>>* = 0)
{
  // No mapping is needed.
  d.value = *std::any_cast<T>(&value);
}

/**
 * This overload is called to set a boolean.
 */
template<typename T>
void SetParam(
    util::ParamData& d,
    const std::any& /* value */,
    const std::enable_if_t<std::is_same_v<T, bool>>* = 0)
{
  // Force set to the value of whether or not this was passed.
  d.value = d.wasPassed;
}

/**
 * Set a matrix parameter, a matrix/dataset info parameter.
 * These set the filename referring to the parameter.
 */
template<typename T>
void SetParam(
    util::ParamData& d,
    const std::any& value,
    const std::enable_if_t<arma::is_arma_type<T>::value || std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>* = 0)
{
  // We're setting the string filename.
  using TupleType = std::tuple<T, typename ParameterType<T>::type>;
  TupleType& tuple = *std::any_cast<TupleType>(&d.value);
  std::get<0>(std::get<1>(tuple)) = std::any_cast<std::string>(value);
}

/**
 * Set a serializable object.  This sets the filename referring to the
 * parameter.
 */
template<typename T>
void SetParam(
    util::ParamData& d,
    const std::any& value,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0,
    const std::enable_if_t<data::HasSerialize<T>::value>* = 0)
{
  // We're setting the string filename.
  using TupleType = std::tuple<T*, typename ParameterType<T>::type>;
  TupleType& tuple = *std::any_cast<TupleType>(&d.value);
  std::get<1>(tuple) = std::any_cast<std::string>(value);
}

/**
 * Return a parameter casted to the given type.  Type checking does not happen
 * here!
 *
 * @param d Parameter information.
 * @param input Unused parameter.
 * @param * (output) Place to store pointer to value.
 */
template<typename T>
void SetParam(util::ParamData& d, const void* input, void* /* output */)
{
  SetParam<std::remove_pointer_t<T>>(
      const_cast<util::ParamData&>(d), *((std::any*) input));
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
