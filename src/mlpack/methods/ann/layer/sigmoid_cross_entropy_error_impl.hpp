/**
 * @file sigmoid_cross_entropy_error_impl.hpp
 * @author Kris Singh
 *
 * Implementation of the sigmoid cross entropy error performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SIGMOID_CROSS_ENTROPY_ERROR
#define MLPACK_METHODS_ANN_LAYER_SIGMOID_CROSS_ENTROPY_ERROR

// In case it hasn't yet been included.
#include "sigmoid_cross_entropy_error.hpp"
#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
SigmoidCrossEntropyError<InputDataType, OutputDataType>
::SigmoidCrossEntropyError()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
double SigmoidCrossEntropyError<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input, const arma::Mat<eT>&& target)
{
  arma::uvec positive = arma::find(input > 0);
  arma::mat output;
  SoftplusFunction::Fn(static_cast<arma::mat>(-arma::abs(input)), output);
  double loss = arma::accu(input(positive)) - arma::accu(input % target) +
        arma::accu(output);

  return loss;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void SigmoidCrossEntropyError<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& input,
    const arma::Mat<eT>&& target,
    arma::Mat<eT>&& output)
{
  arma::mat temp;
  output = input;

  arma::uvec positive = arma::find(input > 0);
  arma::uvec negative = arma::find(input < 0 || input == 0);

  SoftplusFunction::Deriv(static_cast<arma::mat>(-arma::abs(input(positive))),
      temp);
  output(positive) = 1 - target(positive) - temp;
  SoftplusFunction::Deriv(static_cast<arma::mat>(-arma::abs(input(negative))),
      temp);
  output(negative) = -(target(negative)) - negative;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void SigmoidCrossEntropyError<InputDataType, OutputDataType>::Serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here
}

} // namespace ann
} // namespace mlpack

#endif
