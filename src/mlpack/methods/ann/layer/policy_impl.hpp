/**
 * @file policy_impl.hpp
 * @author Chirag Pabbaraju
 *
 * Implementation of the Policy class for Vanilla Policy Gradient with Monte Carlo updates.
 * Here, we are maximising the objective function sum(log(p)*reward) - basically using softmax regression - and not minimising a typical loss like cross entropy.
 * Also, we are using step updates, and not batch updates.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_POLICY_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_POLICY_IMPL_HPP

// In case it hasn't yet been included.
#include "policy.hpp"

namespace mlpack {
namespace ann {

template<typename InputDataType, typename OutputDataType>
Policy<InputDataType, OutputDataType>::Policy()
{ /* Nothing to do here. */ }

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void Policy<InputDataType, OutputDataType>::Forward(
    const InputType &&input, OutputType &&output)
{
  // compute max across every column for scaling while computing softmax to avoid overflow
  arma::mat maxInput = arma::repmat(arma::max(input), input.n_rows, 1);
  arma::mat expInput = arma::exp(input - maxInput);
  // compute denominator for softmax
  arma::mat dense = arma::repmat(arma::sum(expInput), expInput.n_rows, 1);
  output = expInput / dense;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Policy<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT> &&prob,
          arma::Mat<eT> &&advantage,
          arma::Mat<eT> &&g)
{
  // Gradient with respect to nodes corresponding to actions other than the action taken is -p*advantage.
  // Gradient with respect to node corresponding to taken action is (1-p)*advantage.
  // The advantage vector has been correspondingly modified to accommodate this.
  g = - (prob % advantage);
  // We want to maiximise the objective function i.e take a positive step towards this gradient. Hence reverse the sign.
  g = -g; 
}

} // namespace ann
} // namespace mlpack

#endif