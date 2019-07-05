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

template <class ActivationFunction>
size_t Genome<ActivationFunction>::nextInnovID;

template <class ActivationFunction>
std::map<std::pair<size_t, size_t>, size_t> Genome<ActivationFunction>::mutationBuffer;

// Creates genome object during initialization.
template <class ActivationFunction>
Genome<ActivationFunction>::Genome(const size_t inputNodeCount,
                                   const size_t outputNodeCount,
                                   const double bias,
                                   const double initialWeight,
                                   const double weightMutationProb,
                                   const double weightMutationSize,
                                   const double biasMutationProb,
                                   const double biasMutationSize,
                                   const double nodeAdditionProb,
                                   const double connAdditionProb,
                                   const double connDeletionProb,
                                   const double nodeDeletionProb,
                                   const bool isAcyclic):
    inputNodeCount(inputNodeCount),
    outputNodeCount(outputNodeCount),
    bias(bias),
    initialWeight(initialWeight),
    weightMutationProb(weightMutationProb),
    weightMutationSize(weightMutationSize),
    biasMutationProb(biasMutationProb),
    biasMutationSize(biasMutationSize),
    nodeAdditionProb(nodeAdditionProb),
    connAdditionProb(connAdditionProb),
    connDeletionProb(connDeletionProb),
    nodeDeletionProb(nodeDeletionProb),
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
      double weight = initialWeight + arma::randn<double>();
      connectionGeneList.emplace_back(ConnectionGene(counter, weight, i, j));
      if (directedGraph.find(i) == directedGraph.end())
      {
        directedGraph.emplace(std::piecewise_construct,
                              std::make_tuple(i),
                              std::make_tuple());
        directedGraph[i].emplace(j, ConnectionGene(counter++, weight, i, j));
      }
      else
        directedGraph[i].emplace(j, ConnectionGene(counter++, weight, i, j));
    }
  }

  // Set innovation ID.
  nextInnovID = counter;

  // If the genome is meant to be acyclic, we must maintain nodeDepths.
  if (isAcyclic)
  {
    for (size_t i = 0; i <= inputNodeCount; i++)
      nodeDepths.push_back(0);
    for (size_t i = inputNodeCount + 1; i <= outputNodeCount + inputNodeCount; i++)
      nodeDepths.push_back(1);
  }
  else
    outputNodeValues = std::vector<double>(nextNodeID, 0);
}

// Creates genome object during cyclic reproduction.
template <class ActivationFunction>
Genome<ActivationFunction>::Genome(std::vector<ConnectionGene>& connectionGeneList,
                                   const size_t inputNodeCount,
                                   const size_t outputNodeCount,
                                   const size_t nextNodeID,
                                   const double bias,
                                   const double weightMutationProb,
                                   const double weightMutationSize,
                                   const double biasMutationProb,
                                   const double biasMutationSize,
                                   const double nodeAdditionProb,
                                   const double connAdditionProb,
                                   const double connDeletionProb,
                                   const double nodeDeletionProb,
                                   const bool isAcyclic):
    connectionGeneList(connectionGeneList),
    inputNodeCount(inputNodeCount),
    outputNodeCount(outputNodeCount),
    nextNodeID(nextNodeID),
    bias(bias),
    weightMutationProb(weightMutationProb),
    weightMutationSize(weightMutationSize),
    biasMutationProb(biasMutationProb),
    biasMutationSize(biasMutationSize),
    nodeAdditionProb(nodeAdditionProb),
    connAdditionProb(connAdditionProb),
    connDeletionProb(connDeletionProb),
    nodeDeletionProb(nodeDeletionProb),
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
    size_t sourceID = connectionGeneList[i].Source();
    size_t targetID = connectionGeneList[i].Target();
    directedGraph[sourceID][targetID] = connectionGeneList[i];
  }
}

// Creates genome object during acyclic reproduction.
template <class ActivationFunction>
Genome<ActivationFunction>::Genome(std::vector<ConnectionGene>& connectionGeneList,
                                   std::vector<size_t>& nodeDepths,
                                   const size_t inputNodeCount,
                                   const size_t outputNodeCount,
                                   const size_t nextNodeID,
                                   const double bias,
                                   const double weightMutationProb,
                                   const double weightMutationSize,
                                   const double biasMutationProb,
                                   const double biasMutationSize,
                                   const double nodeAdditionProb,
                                   const double connAdditionProb,
                                   const double connDeletionProb,
                                   const double nodeDeletionProb,
                                   const bool isAcyclic):
    connectionGeneList(connectionGeneList),
    nodeDepths(nodeDepths),
    inputNodeCount(inputNodeCount),
    outputNodeCount(outputNodeCount),
    nextNodeID(nextNodeID),
    bias(bias),
    weightMutationProb(weightMutationProb),
    weightMutationSize(weightMutationSize),
    biasMutationProb(biasMutationProb),
    biasMutationSize(biasMutationSize),
    nodeAdditionProb(nodeAdditionProb),
    connAdditionProb(connAdditionProb),
    connDeletionProb(connDeletionProb),
    nodeDeletionProb(nodeDeletionProb),
    isAcyclic(isAcyclic)
{
  for (size_t i = 0; i < nextNodeID; i++)
  {
    directedGraph.emplace(std::piecewise_construct,
                          std::make_tuple(i),
                          std::make_tuple());
  }

  for (size_t i = 0; i < connectionGeneList.size(); i++)
  {
    size_t sourceID = connectionGeneList[i].Source();
    size_t targetID = connectionGeneList[i].Target();
    directedGraph[sourceID][targetID] = connectionGeneList[i];
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

  inputt = input;

  if (isAcyclic)
  {
    AcyclicNet<ActivationFunction> net(nextNodeID, inputNodeCount,
        outputNodeCount, bias);
    output = arma::vec(outputNodeCount, arma::fill::zeros);
    net.Evaluate(input, output, directedGraph, nodeDepths);
    return output;
  }
  else
  {
    CyclicNet<ActivationFunction> net(nextNodeID, inputNodeCount,
        outputNodeCount, bias);
    output = arma::vec(outputNodeCount, arma::fill::zeros);
    net.Evaluate(input, output, outputNodeValues, directedGraph);
    return output;
  }
}

// Mutate genome.
template <class ActivationFunction>
void Genome<ActivationFunction>::Mutate()
{
  // Mutates weights.
  for (size_t i = 0; i < connectionGeneList.size(); i++)
  {
    // Don't mutate the gene if it is not enabled.
    if (!connectionGeneList[i].Enabled())
      continue;

    // Mutate weight.
    if (arma::randu<double>() < weightMutationProb)
    {
      connectionGeneList[i].Mutate(weightMutationSize);
      size_t source = connectionGeneList[i].Source();
      size_t target = connectionGeneList[i].Target();
      directedGraph[source][target].Weight() = connectionGeneList[i].Weight();
    }
  }

  // Mutate bias.
  if (arma::randu<double>() < biasMutationProb)
    bias += biasMutationSize * arma::randn<double>();  

  // Add new connection.
  if (arma::randu<double>() < connAdditionProb)
    AddConnMutation();

  // Add new node.
  if (arma::randu<double>() < nodeAdditionProb)
    AddNodeMutation();

  // Deletes connection.
  if (arma::randu<double>() < connDeletionProb)
    DelConnMutation();

  // Deletes node.
  if (arma::randu<double>() < nodeDeletionProb)
    DelNodeMutation();
}

template <class ActivationFunction>
arma::mat Genome<ActivationFunction>::Parameters()
{
  arma::mat param(nextNodeID, nextNodeID, arma::fill::zeros);
  for (auto const& x : directedGraph)
  {
    for (auto const& y : directedGraph[x.first])
    {
      if (y.second.Enabled())
        param(x.first, y.first) = y.second.Weight();
    }
  }
  return param;
}

template <class ActivationFunction>
void Genome<ActivationFunction>::Print()
{
  std::cout << "Connections:" << std::endl;
  for (auto const& x : directedGraph)
  {
    for (auto const& y : directedGraph[x.first])
    {
      std::cout << x.first << "->" << y.first << " Weight: " << y.second.Weight()
          << " Enabled "<< y.second.Enabled() << std::endl;
    }
  }
  std::cout << "Input: " << inputt << std::endl;
  std::cout << "Output: " << output << std::endl;

  if(!isAcyclic)
  {
    for (size_t i = 0; i < outputNodeValues.size(); i++)
      std::cout << "Value stored in node " << i << " is " << outputNodeValues[i] << std::endl;
  }
}

template <class ActivationFunction>
void Genome<ActivationFunction>::Traverse(size_t startID)
{
  for (auto const& x : directedGraph[startID])
  {
    if (!x.second.Enabled())
      continue;
    if (nodeDepths[x.first] < nodeDepths[startID] + 1)
    {
      nodeDepths[x.first] = nodeDepths[startID] + 1;
      Traverse(x.first);
    }
  }
}

template <class ActivationFunction>
void Genome<ActivationFunction>::AddConnMutation()
{
  size_t sourceID = inputNodeCount + outputNodeCount;
  while (sourceID > inputNodeCount && sourceID <= inputNodeCount + outputNodeCount)
    sourceID = arma::randi<arma::uvec>(1, arma::distr_param(0, (int)(nextNodeID - 1)))[0];

  size_t newTarget = sourceID;
  size_t innovID;

  if (isAcyclic)
  {
    // Only create connections where the target has a higher depth.
    while (nodeDepths[sourceID] >= nodeDepths[newTarget])
    {
      newTarget = arma::randi<arma::uvec>(1, arma::distr_param(1 +
        (int)(inputNodeCount), (int)(nextNodeID) - 1))[0];
    }
  }
  else
  {
    while (newTarget == sourceID)
    {
      newTarget = arma::randi<arma::uvec>(1, arma::distr_param(1 +
          (int)(inputNodeCount), (int)(nextNodeID) - 1))[0];
    }
  }

  if (directedGraph[sourceID].find(newTarget) == directedGraph[sourceID].end())
  {
    std::pair<size_t, size_t> key = std::make_pair(sourceID, newTarget);
    if (mutationBuffer.find(key) == mutationBuffer.end())
    {
      innovID = nextInnovID++;
      mutationBuffer[key] = innovID;
    }
    else
      innovID = mutationBuffer[key];

    // Add the new connection to the containers.
    connectionGeneList.emplace_back(ConnectionGene(innovID, 1, sourceID,
        newTarget));
    directedGraph[sourceID].emplace(newTarget, ConnectionGene(innovID,
        1, sourceID, newTarget));
  }
  else
  {
    if (!directedGraph[sourceID][newTarget].Enabled())
    {
      directedGraph[sourceID][newTarget].Enabled() = true;
      for (size_t j  = 0; j < connectionGeneList.size(); j++)
      {
        if (connectionGeneList[j].Source() == sourceID && connectionGeneList[j].Target() == newTarget)
        {
          connectionGeneList[j].Enabled() = false;
          break;
        }
      }
    }
  }
}

template <class ActivationFunction>
void Genome<ActivationFunction>::AddNodeMutation()
{
  size_t i = 0;
  do
  {
    i = arma::randi<arma::uvec>(1, arma::distr_param(0, (int)(connectionGeneList.size() - 1)))[0];
  }
  while(connectionGeneList[i].Source() == 0);
  size_t sourceID = connectionGeneList[i].Source();
  size_t targetID = connectionGeneList[i].Target();
  size_t newNodeID = nextNodeID++;
  size_t innovID1, innovID2;

  // Check if these mutations have been made. Else, add them to the buffer.
  std::pair<size_t, size_t> key1 = std::make_pair(sourceID, newNodeID);
  std::pair<size_t, size_t> key2 = std::make_pair(newNodeID, targetID);

  if (mutationBuffer.find(key1) == mutationBuffer.end())
  {
    innovID1 = nextInnovID++;
    mutationBuffer[key1] = innovID1;
  }
  else
    innovID1 = mutationBuffer[key1];

  if (mutationBuffer.find(key2) == mutationBuffer.end())
  {
    innovID2 = nextInnovID++;
    mutationBuffer[key2] = innovID2;
  }
  else
    innovID2 = mutationBuffer[key2];

  // Add the first connection to the containers.
  directedGraph[sourceID].emplace(newNodeID, ConnectionGene(innovID1, 1,
      sourceID, newNodeID));
  connectionGeneList.emplace_back(ConnectionGene(innovID1, 1,
      sourceID, newNodeID));

  // Add the second connection to the containers.
  connectionGeneList.emplace_back(ConnectionGene(innovID2, 1,
      newNodeID, targetID));
  directedGraph.emplace(std::piecewise_construct,
                          std::make_tuple(newNodeID),
                          std::make_tuple());
  directedGraph[newNodeID].emplace(targetID, ConnectionGene(innovID2, 1, newNodeID, targetID));

  // Remove the lost connection.
  directedGraph[sourceID][targetID].Enabled() = false;
  connectionGeneList[i].Enabled() = false;

  // If the genome is acyclic, change the depths.
  if (isAcyclic)
  {
    nodeDepths.push_back(nodeDepths[sourceID] + 1);

    // If this is the case, the connection we are splitting is part of the
    // longest path.
    if (nodeDepths[targetID] == nodeDepths[sourceID] + 1)
    {
      nodeDepths[targetID]++;
      Traverse(targetID);
    }
  }
}

template <class ActivationFunction>
void Genome<ActivationFunction>::DelConnMutation()
{
  size_t i = 0;
  do
  {
    i = arma::randi<arma::uvec>(1, arma::distr_param(0, (int)(connectionGeneList.size() - 1)))[0];
  } while (!connectionGeneList[i].Enabled());
  
  size_t sourceID = connectionGeneList[i].Source();
  size_t targetID = connectionGeneList[i].Target();
  size_t outCount = 0, inCount = 0;
  for (size_t j = 0; j < connectionGeneList.size(); j++)
  {
    if (!connectionGeneList[j].Enabled())
      continue;
    if (connectionGeneList[j].Source() == sourceID)
      outCount++;
    if (connectionGeneList[j].Target() == targetID)
      inCount++;
  }

  if (outCount == 1 || inCount == 1)
    return;

  connectionGeneList[i].Enabled() = false;
  directedGraph[sourceID][targetID].Enabled() = false;

  // If the genome is acyclic, change the depths.
  // Think of a better way to do this.
  if (isAcyclic)
  {
    std::fill(nodeDepths.begin(), nodeDepths.end(), 0);
    for (size_t j = 0; j <= inputNodeCount; j++)
      Traverse(j);
  }
}

template <class ActivationFunction>
void Genome<ActivationFunction>::DelNodeMutation()
{
  if (inputNodeCount + outputNodeCount + 1 == nextNodeID)
    return;
  size_t i = arma::randi<arma::uvec>(1, arma::distr_param(inputNodeCount +
      outputNodeCount + 1, (int)(nextNodeID - 1)))[0];
  size_t outCount = 0, inCount = 0;
  size_t sourceID = 0, targetID = 0, conn1Idx = 0, conn2Idx = 0;
  for (size_t j = 0; j < connectionGeneList.size(); j++)
  {
    if (!connectionGeneList[j].Enabled())
      continue;
    if (connectionGeneList[j].Source() == i)
    {
      conn1Idx = j;
      targetID = connectionGeneList[j].Target();
      outCount++;
    }
    if (connectionGeneList[j].Target() == i)
    {
      conn2Idx = j;
      sourceID = connectionGeneList[j].Source();
      inCount++;
    }
  }

  if (outCount == 1 && inCount == 1)
  {
    connectionGeneList.emplace_back(ConnectionGene(nextInnovID, 1, sourceID,
        targetID));
    directedGraph[sourceID].emplace(targetID,ConnectionGene(nextInnovID++, 1, sourceID, targetID));

    // We should disable the deleted connections
    connectionGeneList[conn1Idx].Enabled() = false;
    connectionGeneList[conn2Idx].Enabled() = false;
    directedGraph[sourceID][i].Enabled() = false;
    directedGraph[i][targetID].Enabled() = false;

    if (isAcyclic)
      Traverse(sourceID);
  }
}

template <class ActivationFunction>
size_t Genome<ActivationFunction>::Complexity()
{
  size_t connCount = 0;
  for (size_t i = 0 ; i < connectionGeneList.size(); i++)
  {
    if (connectionGeneList[i].Enabled())
      connCount++;
  }
  return connCount + nextNodeID;
}

template <class ActivationFunction>
template <typename Archive>
void Genome<ActivationFunction>::serialize(Archive& ar,
                                           const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(inputNodeCount);
  ar & BOOST_SERIALIZATION_NVP(outputNodeCount);
  ar & BOOST_SERIALIZATION_NVP(nextNodeID);
  ar & BOOST_SERIALIZATION_NVP(bias);
  ar & BOOST_SERIALIZATION_NVP(initialWeight);
  ar & BOOST_SERIALIZATION_NVP(weightMutationProb);
  ar & BOOST_SERIALIZATION_NVP(weightMutationSize);
  ar & BOOST_SERIALIZATION_NVP(biasMutationProb);
  ar & BOOST_SERIALIZATION_NVP(biasMutationSize);
  ar & BOOST_SERIALIZATION_NVP(nodeAdditionProb);
  ar & BOOST_SERIALIZATION_NVP(connAdditionProb);
  ar & BOOST_SERIALIZATION_NVP(connDeletionProb);
  ar & BOOST_SERIALIZATION_NVP(fitness);
  ar & BOOST_SERIALIZATION_NVP(isAcyclic);
  ar & BOOST_SERIALIZATION_NVP(connectionGeneList);
  ar & BOOST_SERIALIZATION_NVP(directedGraph);
  ar & BOOST_SERIALIZATION_NVP(nextInnovID);
  ar & BOOST_SERIALIZATION_NVP(mutationBuffer);
  if (isAcyclic)
    ar & BOOST_SERIALIZATION_NVP(nodeDepths);
  else
    ar & BOOST_SERIALIZATION_NVP(outputNodeValues);
}

} // namespace neat
} // namespace mlpack

#endif
