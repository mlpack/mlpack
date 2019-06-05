/**
 * @file genome.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Definition of the Genome classes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NEAT_GENOME_IMPL_HPP
#define MLPACK_METHODS_NEAT_GENOME_IMPL_HPP

// In case it hasn't been included already.
#include "genome.hpp"
#include "acyclic_net.hpp"

namespace mlpack{
namespace neat /** NeuroEvolution of Augmenting Topologies */{

template <class ActivationFunction>
Genome<ActivationFunction>::Genome(const size_t inputNodeCount,
                                   const size_t outputNodeCount,
                                   ActivationFunction& actFn,
                                   const double bias,
                                   const double weightMutationRate,
                                   const double weightMutationSize,
                                   const double biasMutationRate,
                                   const double biasMutationSize,
                                   const double nodeAdditionRate,
                                   const double connAdditionRate,
                                   const bool isAcyclic):
    inputNodeCount(inputNodeCount),
    outputNodeCount(outputNodeCount),
    actFn(actFn),
    bias(bias),
    weightMutationRate(weightMutationRate),
    weightMutationSize(weightMutationSize),
    biasMutationRate(biasMutationRate),
    biasMutationSize(biasMutationSize),
    nodeAdditionRate(nodeAdditionRate),
    connAdditionRate(connAdditionRate),
    isAcyclic(isAcyclic)
{
  // Create the node gene list
  for (int i = 0; i <= inputNodeCount + outputNodeCount; i++)
  {
    NodeGeneList.push_back(i);
  }

  // Create connections and add them to the lists.
  for (int i = 0; i <= inputNodeCount; i++)
  {
    for (int j = inputNodeCount+1; j <= outputNodeCount + inputNodeCount; j++)
    {
      ConnectionGeneList.emplace_back(ConnectionGene(nextInnovID++, 1, i, j));
      if (DirectedGraph.find(i) == DirectedGraph.end())
      {
        DirectedGraph.emplace(std::piecewise_construct, std::forward_as_tuple(i),
            std::initializer_list<std::pair<int, ConnectionGene>>{{j,
            ConnectionGene(nextInnovID++, 1, i, j)}});
      }
      else
        DirectedGraph[i].emplace(j, ConnectionGene(nextInnovID++, 1, i, j));
    }
  }
}

template <class ActivationFunction>
void Genome<ActivationFunction>::Input(const arma::vec& input)
{
  if (input.n_elem != inputNodeCount)
  {
    Log::Fatal << "The input should have the same length as the number of"
        "input nodes" << std::endl;
  }
  input = input;
}

template <class ActivationFunction>
arma::vec Genome<ActivationFunction>::Output()
{
  AcyclicNet<ActivationFunction> net(this, actFn);
  return net.Evaluate(input);
}

template <class ActivationFunction>
void Genome<ActivationFunction>::Mutate()
{
  for (size_t i = 0; i < ConnectionGeneList.size(); i++)
  {
    // Mutate weight.
    if (arma::randu<double>() < weightMutationRate)
      ConnectionGeneList[i].Mutate(weightMutationSize);
    
    // Add new node.
    if (arma::randu<double>() < nodeAdditionRate)
    {
      size_t sourceID = ConnectionGeneList[i].source;
      size_t targetID = ConnectionGeneList[i].target;
      size_t newNodeID = nextNodeID++;
      DirectedGraph[sourceID].emplace(newNodeID, ConnectionGene(nextInnovID++, 1, sourceID, newNodeID));
      DirectedGraph.emplace(std::piecewise_construct, std::forward_as_tuple(newNodeID),
            std::initializer_list<std::pair<int, ConnectionGene>>{{targetID,
            ConnectionGene(nextInnovID++, 1, newNodeID, targetID)}});
      DirectedGraph[sourceID].erase(targetID);
      ConnectionGeneList[i].enabled = false;
    }

    // Mutate bias.
    if (arma::randu<double>() < biasMutationRate)
      bias += biasMutationSize * arma::randn<double>();
  }

  // Add new connection.
  for(size_t i = 0; i < NodeGeneList.size(); i++)
  {
    if (arma::randu<double>() < connAdditionRate)
    {
      size_t sourceID = ConnectionGeneList[i].source;
      size_t newTarget = arma::randi<int>(arma::distr_param(0, NodeGeneList.size()));
      if (i != newTarget)
      {
        ConnectionGeneList.emplace_back(ConnectionGene(nextInnovID++, 1, i, newTarget));
        DirectedGraph[sourceID].emplace(newTarget, ConnectionGene(nextInnovID - 1, 1, i, newTarget));
      }
    }
  }
}

} // namespace neat
} // namespace mlpack

#endif
