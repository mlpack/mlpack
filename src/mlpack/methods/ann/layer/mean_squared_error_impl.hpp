/**
 * @file mean_squared_error_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the mean squared error performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MEAN_SQUARED_ERROR_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_MEAN_SQUARED_ERROR_IMPL_HPP

// In case it hasn't yet been included.
#include "mean_squared_error.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
MeanSquaredError<InputDataType, OutputDataType>::MeanSquaredError()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
double MeanSquaredError<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input, const arma::Mat<eT>&& target)
{
  return arma::mean(arma::mean(arma::square(input - target)));
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void MeanSquaredError<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& input,
    const arma::Mat<eT>&& target,
    arma::Mat<eT>&& output)
{
  output = (input - target);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MeanSquaredError<InputDataType, OutputDataType>::Serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
