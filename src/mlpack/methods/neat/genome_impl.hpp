/**
 * @file genome.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Implementation of the Genome class which represents a genome in the 
 * population.
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
                                   const double weightMutationProb,
                                   const double weightMutationSize,
                                   const double biasMutationProb,
                                   const double biasMutationSize,
                                   const double nodeAdditionProb,
                                   const double connAdditionProb,
                                   const bool isAcyclic):
    inputNodeCount(inputNodeCount),
    outputNodeCount(outputNodeCount),
    actFn(actFn),
    bias(bias),
    weightMutationProb(weightMutationProb),
    weightMutationSize(weightMutationSize),
    biasMutationProb(biasMutationProb),
    biasMutationSize(biasMutationSize),
    nodeAdditionProb(nodeAdditionProb),
    connAdditionProb(connAdditionProb),
    isAcyclic(isAcyclic)
{
  // Create the node gene list
  for (size_t i = 0; i <= inputNodeCount + outputNodeCount; i++)
    nodeGeneList.push_back(i);

  // Create connections and add them to the lists.
  for (size_t i = 0; i <= inputNodeCount; i++)
  {
    for (size_t j = inputNodeCount+1; j <= outputNodeCount + inputNodeCount; j++)
    {
      connectionGeneList.emplace_back(ConnectionGene(nextInnovID++, 1, i, j));
      if (directedGraph.find(i) == directedGraph.end())
      {
        directedGraph.emplace(std::piecewise_construct, std::forward_as_tuple(i),
            std::initializer_list<std::pair<int, ConnectionGene>>{{j,
            ConnectionGene(nextInnovID++, 1, i, j)}});
      }
      else
        directedGraph[i].emplace(j, ConnectionGene(nextInnovID++, 1, i, j));
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
  AcyclicNet<ActivationFunction> net(nodeGeneList, directedGraph, actFn,
      inputNodeCount, outputNodeCount, bias);
  return net.Evaluate(input);
}

template <class ActivationFunction>
void Genome<ActivationFunction>::Mutate()
{
  for (size_t i = 0; i < connectionGeneList.size(); i++)
  {
    // Don't mutate the gene if it is not enabled.
    if (!connectionGeneList[i].isEnabled())
      continue;

    // Mutate weight.
    if (arma::randu<double>() < weightMutationProb)
    {
      connectionGeneList[i].Mutate(weightMutationSize);
      
      // Change the weight of the gene in the directed graph as well.
      directedGraph[connectionGeneList[i].source][connectionGeneList[i].target]
          .setWeight() = connectionGeneList[i].getWeight();
    }

    // Add new node.
    else if (arma::randu<double>() < nodeAdditionProb)
    {
      size_t sourceID = connectionGeneList[i].source;
      size_t targetID = connectionGeneList[i].target;
      size_t newNodeID = nextNodeID++;
      
      // Add the first connection to the containers.
      directedGraph[sourceID].emplace(newNodeID, ConnectionGene(nextInnovID, 1,
          sourceID, newNodeID));
      connectionGeneList.emplace_back(ConnectionGene(nextInnovID++, 1,
          sourceID, newNodeID));

      // Add the second connection to the containers.
      connectionGeneList.emplace_back(ConnectionGene(nextInnovID, 1,
          newNodeID, targetID));
      directedGraph.emplace(std::piecewise_construct, std::forward_as_tuple(newNodeID),
            std::initializer_list<std::pair<int, ConnectionGene>>{{targetID,
            ConnectionGene(nextInnovID++, 1, newNodeID, targetID)}});

      // Remove the lost connection.
      directedGraph[sourceID].erase(targetID);
      connectionGeneList[i].enabled = false;
    }
  }
    
  // Mutate bias.
  if (arma::randu<double>() < biasMutationProb)
    bias += biasMutationSize * arma::randn<double>();

  // Add new connection.
  for (size_t i = 0; i < nodeGeneList.size(); i++)
  {
    if (arma::randu<double>() < connAdditionProb)
    {
      size_t sourceID = connectionGeneList[i].source;
      size_t newTargetIdx = arma::randi<arma::vec>(1, arma::distr_param(0, nodeGeneList.size()))[0];
      size_t newTarget = nodeGeneList[newTarget];
      if (i != newTargetIdx)
      {
        connectionGeneList.emplace_back(ConnectionGene(nextInnovID, 1, i, newTarget));
        directedGraph[sourceID].emplace(newTarget, ConnectionGene(nextInnovID++, 1, i, newTarget));
      }
    }
  }
}

} // namespace neat
} // namespace mlpack

#endif
