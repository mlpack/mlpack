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
#include "cyclic_net.hpp"

namespace mlpack{
namespace neat /** NeuroEvolution of Augmenting Topologies */{

// Creates genome object.
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
  // Sets the number of IDs.
  nextNodeID = inputNodeCount + outputNodeCount + 1;

  size_t counter = 0;
  // Create connections and add them to the lists.
  for (size_t i = 0; i <= inputNodeCount; i++)
  {
    for (size_t j = inputNodeCount + 1; j <= outputNodeCount + inputNodeCount;
        j++)
    {
      connectionGeneList.emplace_back(ConnectionGene(counter++, 1, i, j));
      if (directedGraph.find(i) == directedGraph.end())
      {
        directedGraph.emplace(std::piecewise_construct, std::forward_as_tuple
            (i), std::initializer_list<std::pair<int, ConnectionGene>>{{j,
            ConnectionGene(counter++, 1, i, j)}});
      }
      else
        directedGraph[i].emplace(j, ConnectionGene(counter++, 1, i, j));
    }
  }

  // Set innovation ID.
  nextInnovID = counter;

  // If the genome is meant to be acyclic, we must maintain nodeDepths.
  if (isAcyclic)
  {
    for (size_t i = 0; i <= inputNodeCount; i++)
      nodeDepths.push_back(0);
    for (size_t i = inputNodeCount + 1; i <= outputNodeCount; i++)
      nodeDepths.push_back(1);
  }
}

template <class ActivationFunction>
Genome<ActivationFunction>::Genome(const size_t inputNodeCount,
                                   const size_t outputNodeCount,
                                   std::vector<ConnectionGene>& connectionGeneList,
                                   const size_t nextNodeID,
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
    connectionGeneList(connectionGeneList),
    nextNodeID(nextNodeID),
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
  // TODO: Decide whether using map::find() to check which nodes to include in
  // digraph is better, or preparing like this.
  for (size_t i = 0; i < nextNodeID; i++)
  {
    directedGraph.emplace(std::piecewise_construct,
                          std::make_tuple(i),
                          std::make_tuple());
  }

  for (size_t i = 0; i < connectionGeneList.size(); i++)
  {
    size_t sourceID = connectionGeneList[i].getSource();
    size_t targetID = connectionGeneList[i].getTarget();
    directedGraph[sourceID][targetID] = connectionGeneList[i];
  }

  if (isAcyclic)
  {
    for (size_t i = 0; i < nextNodeID; i++)
      nodeDepths.push_back(0);
    for (size_t i = 0; i <= inputNodeCount; i++)
      TraverseNode(i, 0);
  }
}

// Evaluates output based on input.
template <class ActivationFunction>
arma::vec Genome<ActivationFunction>::Evaluate(arma::vec& input)
{
  if (input.n_elem != inputNodeCount)
  {
    Log::Fatal << "The input should have the same length as the number of"
        "input nodes" << std::endl;
  }

  if (isAcyclic)
  {
    AcyclicNet<ActivationFunction> net(actFn, nextNodeID, inputNodeCount,
        outputNodeCount, bias);
    fitness = net.Evaluate(input, directedGraph, nodeDepths);
    return fitness;
  }
  else
  {
    CyclicNet<ActivationFunction> net(actFn, nextNodeID, inputNodeCount,
        outputNodeCount, 100 /* Placeholder */, bias);
    fitness = net.Evaluate(input, directedGraph);
    return fitness;
  }
}

// Mutate genome.
template <class ActivationFunction>
void Genome<ActivationFunction>::Mutate()
{
  // Add new connection.
  for (size_t i = 0; i < nextNodeID; i++)
  {
    if (arma::randu<double>() < connAdditionProb)
    {
      size_t sourceID = connectionGeneList[i].source;
      size_t newTarget = i;
      while (newTarget == i)
      {
        newTarget = arma::randi<arma::vec>(1 + inputNodeCount,
            arma::distr_param(0, nextNodeID - 1))[0];
      }

      std::pair<size_t, size_t> key = std::make_pair(sourceID, newTarget);
      size_t innovID = -1;
      if (mutationBuffer.find(key) == mutationBuffer.end())
      {
        innovID = nextInnovID++;
        mutationBuffer[key] = innovID;
      }
      else
        innovID = mutationBuffer[key];

      if (isAcyclic)
      {
        // Only create connections where the target has a higher depth.
        if (nodeDepths[i] >= nodeDepths[newTarget])
          continue;
      }

      // Add the new connection to the containers.
      connectionGeneList.emplace_back(ConnectionGene(innovID, 1, i,
          newTarget));
      directedGraph[sourceID].emplace(newTarget, ConnectionGene(innovID++,
          1, i, newTarget));
    }
  }

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
    if (arma::randu<double>() < nodeAdditionProb)
    {
      size_t sourceID = connectionGeneList[i].source;
      size_t targetID = connectionGeneList[i].target;
      size_t newNodeID = nextNodeID++;
      size_t innovID1 = -1, innovId2 = -1;

      std::pair<size_t, size_t> key1 = std::make_pair(sourceID, newNodeID);
      std::pair<size_t, size_t> key2 = std::make_pair(newNodeID, targetID);    
      
      if (mutationBuffer.find(key1) == mutationBuffer.end())
      {
        innovID1 = nextInnovID++;
        mutationBuffer[key1] = innovID;
      }
      else
        innovID1 = mutationBuffer[key1];
      
      if (mutationBuffer.find(key2) == mutationBuffer.end())
      {
        innovID1 = nextInnovID++;
        mutationBuffer[key2] = innovID;
      }
      else
        innovID1 = mutationBuffer[key2];  

      // Add the first connection to the containers.
      directedGraph[sourceID].emplace(newNodeID, ConnectionGene(innovID1, 1,
          sourceID, newNodeID));
      connectionGeneList.emplace_back(ConnectionGene(innovID1, 1,
          sourceID, newNodeID));

      // Add the second connection to the containers.
      connectionGeneList.emplace_back(ConnectionGene(innovId2, 1,
          newNodeID, targetID));
      directedGraph.emplace(std::piecewise_construct, std::forward_as_tuple(
            newNodeID), std::initializer_list<std::pair<int, ConnectionGene>>{{
            targetID, ConnectionGene(innovId2, 1, newNodeID, targetID)}});

      // Remove the lost connection.
      directedGraph[sourceID].erase(targetID);
      connectionGeneList[i].enabled = false;

      // If the genome is acyclic, change the depths.
      if (isAcyclic)
      {
        nodeDepths.push_back(nodeDepths[sourceID] + 1);

        // If this is the case, the connection we are splitting is part of the
        // longest path.
        if (nodeDepths[targetID] == nodeDepths[sourceID] + 1)
          nodeDepths[targetID] += 1;
      }
    }
  }

  // Mutate bias.
  if (arma::randu<double>() < biasMutationProb)
    bias += biasMutationSize * arma::randn<double>();
}

// Recursively traverse neighbours and assign depths.
template <class ActivationFunction>
void Genome<ActivationFunction>::TraverseNode(size_t nodeID, size_t depth)
{
  // Check if it has been traversed by a longer path.
  if (nodeDepths[nodeID] >= depth)
    return;
  else
    nodeDepths[nodeID] = depth;

  for (auto const& x : directedGraph)
    TraverseNode(x.first, depth + 1);
}

template <class ActivationFunction>
arma::mat Genome<ActivationFunction>::Parameters()
{
  arma::mat param(nextNodeID, nextNodeID);
  for (auto const& x : directedGraph)
  {
    for (auto const& y : x.second)
    {
      param[x.first][y.first] = y.second.getWeight();
    }
  }
  return param;
}

} // namespace neat
} // namespace mlpack

#endif
