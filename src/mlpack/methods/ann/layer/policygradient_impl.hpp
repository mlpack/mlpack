/**
 * @file Advantage_error_impl.hpp
 * @author Rohan Raj
 *
 * Implementation of the Generalised Advantage Estimation method using policy gradient.
 * John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan and Pieter Abbeel
 * HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION, ICLR 2016
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ADVANTAGE_ERROR_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ADVANTAGE_ERROR_IMPL_HPP

// In case it hasn't yet been included.
#include "policygradient.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
PolicyGradient<InputDataType, OutputDataType>::PolicyGradient()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
double PolicyGradient<InputDataType, OutputDataType>::Forward(
    const InputType&& input, const TargetType&& target)
{
  return (-1*arma::mean(arma::mean(arma::log(input) % target)));
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void PolicyGradient<InputDataType, OutputDataType>::Backward(
    const InputType&& input,
    const TargetType&& target,
    OutputType&& output)
{
  /**
  I am trying to compute backwards by multiplying p(i) - 1 * advantage of that action
  Since the advantage value of action not taken is 0, they wont create any errors. 
  The dot product will output 0 where advantage is 0.
  */
  output = (input - 1) % target;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void PolicyGradient<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
