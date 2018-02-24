/**
 * @file Advantage_error_impl.hpp
 * @author Rohan Raj
 *
 * Implementation of the Advantage error performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ADVANTAGE_ERROR_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ADVANTAGE_ERROR_IMPL_HPP

// In case it hasn't yet been included.
#include "advantage_loss.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
AdvantageError<InputDataType, OutputDataType>::AdvantageError()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
double AdvantageError<InputDataType, OutputDataType>::Forward(
    const InputType&& input, const TargetType&& target)
{
  /*
  The loss function is mean of log(probability) * target
  The target captures the advantage of taking the action.
  The -1 is multiplied to take negative of mean.

  Reference : https://karpathy.github.io/2016/05/31/rl/

  % does element wise multiplication
  Refer : https://stackoverflow.com/questions/26790229/element-wise-vector-or-matrix-multiplication-in-armadillo-c
  
  arma::log does element wise logarithm
  Refer : http://arma.sourceforge.net:80/docs.html#misc_fns
 */
  return (-1*arma::mean(arma::mean(arma::log(input) % target)));
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void AdvantageError<InputDataType, OutputDataType>::Backward(
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
void AdvantageError<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
