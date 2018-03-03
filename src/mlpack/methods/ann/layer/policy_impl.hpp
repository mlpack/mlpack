/**
 * @file policy_impl.hpp
 * @author Shangtong Zhang
 *
 * Implementation of the Policy class.
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
Policy<InputDataType, OutputDataType>::Policy(
    double entropyRegularizationWeight) :
    entropyRegularizationWeight(entropyRegularizationWeight)
{ /* Nothing to do here. */ }

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void Policy<InputDataType, OutputDataType>::Forward(
    const InputType &&input, OutputType &&output)
{
  arma::mat maxInput = arma::repmat(arma::max(input), input.n_rows, 1);
  arma::mat expInput = arma::exp(input - maxInput);
  arma::mat dense = arma::repmat(arma::sum(expInput), expInput.n_rows, 1);
  output = expInput / dense;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Policy<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT> &&prob, arma::Mat<eT> &&advantage, arma::Mat<eT> &&g)
{
  arma::mat adv = arma::repmat(arma::sum(advantage), advantage.n_rows, 1);
  arma::mat gradientPolicy = advantage - adv % prob;
  arma::mat logProb = arma::trunc_log(prob);
  arma::mat tmp = prob % (logProb + 1);
  size_t n = advantage.n_rows;
  arma::mat gradientKL = arma::sum((arma::eye(n, n) -
      arma::repmat(arma::trans(prob), n, 1)) % arma::repmat(tmp, 1, n));
  arma::inplace_trans(gradientKL);
  g = -gradientPolicy + entropyRegularizationWeight * gradientKL;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Policy<InputDataType, OutputDataType>::Serialize(
        Archive& /* ar */,
        const unsigned int /* version */)
{ /* Nothing to do there. */ }

} // namespace ann
} // namespace mlpack

#endif