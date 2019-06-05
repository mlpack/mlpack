/**
 * @file genome.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Definition of the Genome classes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If n
 ot, see
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
                                   const double connMutationRate,
                                   const bool isAcyclic):
    inputNodeCount(inputNodeCount),
    outputNodeCount(outputNodeCount),
    actFn(actFn),
    bias(bias),
    weightMutationRate(weightMutationRate),
    weightMutationSize(weightMutationSize),
    biasMutationRate(biasMutationRate),
    biasMutationSize(biasMutationSize),
    connMutationRate(connMutationRate),
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
            std::initializer_list<std::pair<int, ConnectionGene>>{{i,
            ConnectionGene(nextInnovID++, 1, i, j)}});
      }
      else
        DirectedGraph[i].emplace(ConnectionGene(nextInnovID++, 1, i, j));
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
  for (int i = 0; i < ConnectionGeneList.size(); i++)
  {
    if (arma::randu<double>() < weightMutationRate)
      ConnectionGeneList[i].Mutate(weightMutationSize);
    if (arma::randu<double>() < connMutationRate)
    {
      ConnectionGene temp1(nextInnovID++, 1, ConnectionGeneList[i].source,
          nextNodeID++);
      ConnectionGene temp2(nextInnovID++, 1, nextNodeID++,
          ConnectionGeneList[i].target);
      DirectedGraph[ConnectionGeneList[i].source].erase(ConnectionGeneList[i].target);
    }
    if (arma::randu<double>() < biasMutationRate)
    {
      bias += biasMutationSize * arma::randn<double>();
    }
  }
}

}
}

#endif
