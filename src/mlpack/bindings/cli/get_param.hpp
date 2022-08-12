/**
 * @file bindings/cli/get_param.hpp
 * @author Ryan Curtin
 *
 * Use template metaprogramming to get the right type of parameter.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CLI_GET_PARAM_HPP
#define MLPACK_BINDINGS_CLI_GET_PARAM_HPP

#include <mlpack/prereqs.hpp>
#include "parameter_type.hpp"

namespace mlpack {
namespace bindings {
namespace cli {

/**
 * This overload is called when nothing special needs to happen to the name of
 * the parameter.
 *
 * @param d ParamData object to get parameter value from.
 */
template<typename T>
T& GetParam(
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
 *
 * @param d ParamData object to get parameter value from.
 */
template<typename T>
T& GetParam(
    util::ParamData& d,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
{
  // If the matrix is an input matrix, we have to load the matrix.  'value'
  // contains the filename.  It's possible we could load empty matrices many
  // times, but I am not bothered by that---it shouldn't be something that
  // happens.
  typedef std::tuple<T, typename ParameterType<T>::type> TupleType;
  TupleType& tuple = *MLPACK_ANY_CAST<TupleType>(&d.value);
  const std::string& value = std::get<0>(std::get<1>(tuple));
  T& matrix = std::get<0>(tuple);
  size_t& n_rows = std::get<1>(std::get<1>(tuple));
  size_t& n_cols = std::get<2>(std::get<1>(tuple));
  if (d.input && !d.loaded)
  {
    // Call correct data::Load() function.
    if (arma::is_Row<T>::value || arma::is_Col<T>::value)
      data::Load(value, matrix, true);
    else
      data::Load(value, matrix, true, !d.noTranspose);
    n_rows = matrix.n_rows;
    n_cols = matrix.n_cols;
    d.loaded = true;
  }

  return matrix;
}

/**
 * Return a matrix/dataset info parameter.
 *
 * @param d ParamData object to get parameter value from.
 */
template<typename T>
T& GetParam(
    util::ParamData& d,
    const typename std::enable_if<std::is_same<T,
        std::tuple<mlpack::data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  // If this is an input parameter, we need to load both the matrix and the
  // dataset info.
  typedef std::tuple<T, std::tuple<std::string, size_t, size_t>> TupleType;
  TupleType* tuple = MLPACK_ANY_CAST<TupleType>(&d.value);
  const std::string& value = std::get<0>(std::get<1>(*tuple));
  T& t = std::get<0>(*tuple);
  size_t& n_rows = std::get<1>(std::get<1>(*tuple));
  size_t& n_cols = std::get<2>(std::get<1>(*tuple));
  if (d.input && !d.loaded)
  {
    data::Load(value, std::get<1>(t), std::get<0>(t), true, !d.noTranspose);
    n_rows = std::get<1>(t).n_rows;
    n_cols = std::get<1>(t).n_cols;
    d.loaded = true;
  }

  return t;
}

/**
 * Return a serializable object.
 *
 * @param d ParamData object to get parameter value from.
 */
template<typename T>
T*& GetParam(
    util::ParamData& d,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0)
{
  // If the model is an input model, we have to load it from file.  'value'
  // contains the filename.
  typedef std::tuple<T*, std::string> TupleType;
  TupleType* tuple = MLPACK_ANY_CAST<TupleType>(&d.value);
  const std::string& value = std::get<1>(*tuple);
  if (d.input && !d.loaded)
  {
    T* model = new T();
    data::Load(value, "model", *model, true);
    d.loaded = true;
    std::get<0>(*tuple) = model;
  }
  return std::get<0>(*tuple);
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
void GetParam(util::ParamData& d, const void* /* input */, void* output)
{
  // Cast to the correct type.
  *((T**) output) = &GetParam<typename std::remove_pointer<T>::type>(d);
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
