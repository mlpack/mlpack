/**
 * @file bindings/cli/output_param_impl.hpp
 * @author Ryan Curtin
 *
 * Implementations of functions to output parameters of different types.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_OUTPUT_PARAM_IMPL_HPP
#define MLPACK_CORE_UTIL_OUTPUT_PARAM_IMPL_HPP

#include "output_param.hpp"
#include <mlpack/core/data/save.hpp>
#include <iostream>

namespace mlpack {
namespace bindings {
namespace cli {

//! Output an option.
template<typename T>
void OutputParamImpl(
    util::ParamData& data,
    const std::enable_if_t<!arma::is_arma_type<T>::value>*,
    const std::enable_if_t<!util::IsStdVector<T>::value>*,
    const std::enable_if_t<!data::HasSerialize<T>::value>*,
    const std::enable_if_t<!std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>*)
{
  std::cout << data.name << ": " << *std::any_cast<T>(&data.value)
      << std::endl;
}

//! Output a vector option.
template<typename T>
void OutputParamImpl(
    util::ParamData& data,
    const std::enable_if_t<util::IsStdVector<T>::value>*)
{
  std::cout << data.name << ": ";
  const T& t = *std::any_cast<T>(&data.value);
  for (size_t i = 0; i < t.size(); ++i)
    std::cout << t[i] << " ";
  std::cout << std::endl;
}

//! Output a matrix option (this saves it to file).
template<typename T>
void OutputParamImpl(
    util::ParamData& data,
    const std::enable_if_t<arma::is_arma_type<T>::value>*)
{
  using TupleType = std::tuple<T, std::tuple<std::string, size_t, size_t>>;
  const T& output = std::get<0>(*std::any_cast<TupleType>(&data.value));
  const std::string& filename =
      std::get<0>(std::get<1>(*std::any_cast<TupleType>(&data.value)));

  if (output.n_elem > 0 && filename != "")
  {
    if (arma::is_Row<T>::value || arma::is_Col<T>::value)
      data::Save(filename, output, false);
    else
      data::Save(filename, output, false, !data.noTranspose);
  }
}

//! Output a model option (this saves it to file).
template<typename T>
void OutputParamImpl(
    util::ParamData& data,
    const std::enable_if_t<!arma::is_arma_type<T>::value>*,
    const std::enable_if_t<data::HasSerialize<T>::value>*)
{
  // The const cast is necessary here because Serialize() can't ever be marked
  // const.  In this case we can assume it though, since we will be saving and
  // not loading.
  using TupleType = std::tuple<T*, std::string>;
  T*& output = const_cast<T*&>(std::get<0>(*std::any_cast<TupleType>(
      &data.value)));
  const std::string& filename =
      std::get<1>(*std::any_cast<TupleType>(&data.value));

  if (filename != "")
    data::Save(filename, "model", *output);
}

//! Output a mapped dataset.
template<typename T>
void OutputParamImpl(
    util::ParamData& data,
    const std::enable_if_t<std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>* /* junk */)
{
  // Output the matrix with the mappings.
  using TupleType = std::tuple<T, std::tuple<std::string, size_t, size_t>>;
  const T& tuple = std::get<0>(*std::any_cast<TupleType>(&data.value));
  const std::string& filename =
      std::get<0>(std::get<1>(*std::any_cast<TupleType>(&data.value)));
  const arma::mat& matrix = std::get<1>(tuple);

  // The mapping isn't taken into account.  We should write a data::Save()
  // overload for this.
  if (filename != "")
    data::Save(filename, matrix, false, !data.noTranspose);
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
