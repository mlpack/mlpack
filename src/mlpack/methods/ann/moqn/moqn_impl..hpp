/**
 * @file methods/ann/morl_network_impl.hpp
 * @author Nanubala Gnana Sai
 *
 * Definition of the Multi-objective Q-network (MOQN) class, extends feed forward neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_MORL_NETWORK_IMPL_HPP
#define MLPACK_METHODS_ANN_MORL_NETWORK_IMPL_HPP

// In case it hasn't been included yet.
#include "moqn.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename OutputLayerType, typename InitializationRuleType,
         typename... CustomLayers>
template<typename PredictorsType,
         typename TargetsType,
         typename WeightsType,
         typename GradientsType>
double MOQN<OutputLayerType, InitializationRuleType, CustomLayers...>::Backward(
    const PredictorsType& inputs,
    const TargetsType& targets,
    const WeightsType& extendedWeightSpace,
    double lambda,
    GradientsType& gradients)
{
  double loss = 0.0;
  arma::mat predictions =
      boost::apply_visitor(outputParameterVisitor, network.back());
  const size_t numElem = arma::sum((predictions - targets) != 0);
  // Homotopy loss.
  const double lossA =
      std::pow(arma::norm((predictions - targets).vectorise()), 2) / numElem;
  const double lossB =
    std::pow(arma::norm(arma::sum(extendedWeightSpace %
                                 (predictions - targets))), 2) / numElem;

  const double homotopyLoss = (1 - lambda) * lossA + lambda * lossB;

  // Store the error.
  arma::mat errorA = (predictions - targets) / numElem;
  arma::mat errorB = arma::sum(extendedWeightSpace % (predictions - targets)) %
                     extendedWeightSpace;
  error = 2 * ((1 - lambda) * errorA + lambda * errorB);

  gradients = arma::zeros<arma::mat>(parameter.n_rows, parameter.n_cols);

  Backward();
  ResetGradients(gradients);
  Gradient(inputs);

  return loss;
}

} // namespace ann
} // namespace mlpack

#endif
