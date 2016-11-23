/**
 * @file param_data_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of utility functions for ParamData structures.
 */
#ifndef MLPACK_CORE_UTIL_PARAM_DATA_IMPL_HPP
#define MLPACK_CORE_UTIL_PARAM_DATA_IMPL_HPP

#include "param_data.hpp"
#include <mlpack/core/data/load.hpp>

namespace mlpack {
namespace util {

//! This overload is called when nothing special needs to happen to the name of
//! the parameter.
template<typename T>
std::string MapParameterName(
    const std::string& identifier,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* /* junk */,
    const typename boost::disable_if<data::HasSerialize<T>>::type* /* junk */)
{
  return identifier;
}

//! This is called for matrices, which have a different boost name.
template<typename T>
std::string MapParameterName(
    const std::string& identifier,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* /* junk */)
{
  return identifier + "_file";
}

//! This is called for serializable mlpack objects, which have a different boost
//! name.
template<typename T>
std::string MapParameterName(
    const std::string& identifier,
    const typename boost::enable_if<data::HasSerialize<T>>::type* /* junk */)
{
  return identifier + "_file";
}

//! This overload is called when T == ParameterType<T>::value.
template<typename T>
T& HandleParameter(
    typename ParameterType<T>::type& value,
    ParamData& /* d */,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* /* junk */,
    const typename boost::disable_if<data::HasSerialize<T>>::type* /* junk */)
{
  return value;
}

//! This overload is called for matrices, which return a different type.
template<typename T>
T& HandleParameter(
    typename ParameterType<T>::type& value,
    ParamData& d,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* /* junk */)
{
  // If the matrix is an input matrix, we have to load the matrix.  'value'
  // contains the filename.  It's possible we could load empty matrices many
  // times, but I am not bothered by that---it shouldn't be something that
  // happens.
  T& matrix = *boost::any_cast<T>(&d.mappedValue);
  if (d.input && !d.loaded)
  {
    data::Load(value, matrix, true, !d.noTranspose);
    d.loaded = true;
  }

  return matrix;
}

//! This is called for serializable mlpack objects, which have a different boost
//! name.
template<typename T>
T& HandleParameter(
    typename ParameterType<T>::type& value,
    ParamData& d,
    const typename boost::enable_if<data::HasSerialize<T>>::type* /* junk */)
{
  // If the model is an input model, we have to load it from file.  'value'
  // contains the filename.
  T& model = *boost::any_cast<T>(&d.mappedValue);
  if (d.input && !d.loaded)
  {
    data::Load(value, "model", model, true);
  }

  return model;
}

} // namespace util
} // namespace mlpack

#endif
