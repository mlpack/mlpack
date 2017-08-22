#ifndef MLPACK_METHODS_ANN_LAYER_POLICY_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_POLICY_IMPL_HPP

#include "policy.hpp"

namespace mlpack {
namespace ann {

template<typename InputDataType, typename OutputDataType>
Policy<InputDataType, OutputDataType>::Policy(
        double entropyRegularizationWeight) : entropyRegularizationWeight(entropyRegularizationWeight)
{

}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void Policy<InputDataType, OutputDataType>::Forward(const InputType &&input,
                                                    OutputType &&output)
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
//  arma::mat gradientProb = prob % (1 - prob);
//  arma::mat gradientLogProb = 1 - prob;
  arma::mat adv = arma::repmat(arma::sum(advantage), advantage.n_rows, 1);
  arma::mat gradientPolicy = advantage - adv % prob;
  arma::mat logProb = arma::log(prob);
  arma::mat tmp = prob % (logProb + 1);
  size_t n = advantage.n_rows;
  arma::mat gradientKL = arma::sum((arma::trans(arma::eye(n, n) - arma::repmat(prob, 1, n)) % arma::repmat(tmp, 1, n)));
  arma::inplace_trans(gradientKL);

//  for (size_t i = 0; i < advantage.n_rows; ++i)
//  {
//    arma::colvec e(advantage.n_rows, arma::fill::zeros);
//    e[i] = 1;
//    gradientKL[i] =
//
//  }

//  arma::mat gradientPolicy = prob + advantage;
//  arma::mat logProb = arma::log(prob);
//  arma::mat gradientKL = gradientProb % logProb + prob % gradientLogProb;
//  g = -gradientPolicy + entropyRegularizationWeight * gradientKL;
  g = -gradientPolicy + entropyRegularizationWeight * gradientKL;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Policy<InputDataType, OutputDataType>::Serialize(
        Archive& /* ar */,
        const unsigned int /* version */)
{
  // Nothing to do here.
}

}
}

#endif