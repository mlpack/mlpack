/**
 * @file cyclic_net_impl.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Implementation of the CyclicNet class, which represents a cyclic neural
 * network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_NEAT_CYCLIC_NET_IMPL_HPP
#define MLPACK_METHODS_NEAT_CYCLIC_NET_IMPL_HPP

// In case it hasn't been included yet.
#include "cyclic_net.hpp"

namespace mlpack{
namespace neat /** NeuroEvolution of Augmenting Topologies */{

template <class ActivationFunction>
CyclicNet<ActivationFunction>::CyclicNet(const size_t nodeCount,
                                         const size_t inputNodeCount,
                                         const size_t outputNodeCount,
                                         const double bias):
    nodeCount(nodeCount),
    inputNodeCount(inputNodeCount),
    outputNodeCount(outputNodeCount),
    bias(bias)
{ /* Nothing to do here */ }

template <class ActivationFunction>
void CyclicNet<ActivationFunction>::Evaluate(const arma::vec& input,
                                             arma::vec& output,
                                             std::vector<double>&
                                                outputNodeValues,
                                             const std::map<size_t, std::map
                                                <size_t, ConnectionGene>>&
                                                directedGraph)
{
  arma::vec inputNodeValues(nodeCount, arma::fill::zeros);

  while (outputNodeValues.size() < nodeCount)
    outputNodeValues.push_back(0);

  // Load the bias.
  outputNodeValues[0] = bias;

  // Load the input.
  for (size_t i = 1; i < nodeCount; i++)
  {
    if (i <= inputNodeCount)
      outputNodeValues[i] = input[i - 1];
    else
      inputNodeValues[i] = outputNodeValues[i];
  }

  for (size_t i = 0; i < nodeCount; i++)
  {
    if (i > inputNodeCount + outputNodeCount)
      outputNodeValues[i] = ActivationFunction::Fn(inputNodeValues[i]);
    for (auto const& x : directedGraph[i])
      inputNodeValues[x.first] += outputNodeValues[i] * x.second.Weight();
  }

  for (size_t i = 0; i < output.n_elem; i++)
    output[i] = ActivationFunction::Fn(inputNodeValues[i + inputNodeCount + 1]);
}

} // namespace neat
} // namespace mlpack

#endif
