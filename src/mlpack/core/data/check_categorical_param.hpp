/**
 * @file core/data/check_categorical_param.hpp
 * @author Ryan Curtin
 *
 * This file provides an implementation of a simple function to check the values
 * of a categorical parameter.  It cannot be defined in util/, since the
 * DatasetMapper class is not fully defined when that is included..
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_CHECK_CATEGORICAL_PARAM_HPP
#define MLPACK_CORE_DATA_CHECK_CATEGORICAL_PARAM_HPP

#include <mlpack/core/util/params.hpp>

namespace mlpack {
namespace data {

inline void CheckCategoricalParam(util::Params& params,
                                  const std::string& paramName)
{
  using TupleType = std::tuple<DatasetInfo, arma::mat>;
  arma::mat& matrix = std::get<1>(params.Get<TupleType>(paramName));

  // This comes from Params::CheckInputMatrix().
  const std::string errMsg1 = "The input '" + paramName + "' has NaN values.";
  const std::string errMsg2 = "The input '" + paramName + "' has Inf values.";

  if (matrix.has_nan())
    Log::Fatal << errMsg1 << std::endl;
  if (matrix.has_inf())
    Log::Fatal << errMsg2 << std::endl;
}

} // namespace data
} // namespace mlpack

#endif
