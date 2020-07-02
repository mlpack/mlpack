/**
 * @file methods/ann/loss_functions/huber_loss_impl.hpp
 * @author Mrityunjay Tripathi
 *
 * Implementation of the Huber loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_HUBER_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_HUBER_LOSS_IMPL_HPP

// In case it hasn't yet been included.
#include "huber_loss.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
HuberLoss<InputDataType, OutputDataType>::HuberLoss(
  const double delta,
  const bool reduction):
  delta(delta),
  reduction(reduction)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
typename InputType::elem_type
HuberLoss<InputDataType, OutputDataType>::Forward(const InputType& input,
                                                  const TargetType& target)
{
  InputType loss;
  loss.zeros(size(input));
  typedef typename InputType::elem_type ElemType;
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    const ElemType absError = std::abs(target[i] - input[i]);
    loss[i] = absError > delta
        ? delta * (absError - 0.5 * delta) : 0.5 * std::pow(absError, 2);
  }

  ElemType lossSum = arma::accu(loss);

  if (reduction)
    return lossSum;

  return lossSum / input.n_elem;
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void HuberLoss<InputDataType, OutputDataType>::Backward(
    const InputType& input,
    const TargetType& target,
    OutputType& output)
{
  typedef typename InputType::elem_type ElemType;

  output.set_size(size(input));
  for (size_t i = 0; i < output.n_elem; ++i)
  {
    const ElemType absError = std::abs(target[i] - input[i]);
    output[i] = absError > delta
        ? - delta * (target[i] - input[i]) / absError : input[i] - target[i];
  }

  if (!reduction)
    output = output / input.n_elem;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void HuberLoss<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(delta);
  ar & BOOST_SERIALIZATION_NVP(reduction);
}

} // namespace ann
} // namespace mlpack

#endif
