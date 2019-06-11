/**
 * @file neat_impl.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Implementation of the NEAT class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_NEAT_NEAT_IMPL_HPP
#define MLPACK_METHODS_NEAT_NEAT_IMPL_HPP

// In case it hasn't been included.
#include "neat.hpp"

namespace mlpack{
namespace neat /** NeuroEvolution of Augmenting Topologies */ {

template <class TaskType,
          class ActivationFunction>
NEAT<TaskType, ActivationFunction>::NEAT(TaskType& task,
                                         ActivationFunction& actFn,
                                         const size_t inputNodeCount,
                                         const size_t outputNodeCount,
                                         const size_t popSize,
                                         const size_t maxGen,
                                         const double bias,
                                         const double weightMutationProb,
                                         const double weightMutationSize,
                                         const double biasMutationProb,
                                         const double biasMutationSize,
                                         const double nodeAdditionProb,
                                         const double connAdditionProb,
                                         const bool isAcyclic):
    task(task),
    actFn(actFn),
    inputNodeCount(inputNodeCount),
    outputNodeCount(outputNodeCount),
    popSize(popSize),
    maxGen(maxGen),
    bias(bias),
    weightMutationProb(weightMutationProb),
    weightMutationSize(weightMutationSize),
    biasMutationProb(biasMutationProb),
    biasMutationSize(biasMutationSize),
    nodeAdditionProb(nodeAdditionProb),
    connAdditionProb(connAdditionProb),
    isAcyclic(isAcyclic)
{ /* Nothing to do here yet */ }

template <class TaskType,
          class ActivationFunction>
Genome<ActivationFunction> NEAT<TaskType, ActivationFunction>::Train()
{
  Genome<ActivationFunction>::nextInnovID = 0;

  // Initialize.
  for (size_t i = 0; i < popSize; i++)
  {
    genomeList.emplace_back(Genome<ActivationFunction>(inputNodeCount,
        outputNodeCount, actFn, bias, weightMutationProb,
        weightMutationSize, biasMutationProb, biasMutationSize,
        nodeAdditionProb, connAdditionProb, isAcyclic));
  }

  // Main loop.
  for (size_t gen = 0; gen < maxGen; gen++)
  {
    Genome<ActivationFunction>::mutationBuffer.clear();
    for (size_t i = 0; i < popSize; i++)
      task.Evaluate(genomeList[i]);
    Speciate();
    Reproduce();
  }

  // Find best genome.
  size_t maxIdx = -1;
  double max = -DBL_MAX;
  for (size_t i = 0; i < popSize; i++)
  {
    if (genomeList[i].getFitness() > max)
    {
      maxIdx = i;
      max = genomeList[i].getFitness();
    }
  }

  return genomeList[maxIdx];
}

template <class TaskType,
          class ActivationFunction>
void NEAT<TaskType, ActivationFunction>::Reproduce()
{
  
}

template <class TaskType,
          class ActivationFunction>
void NEAT<TaskType, ActivationFunction>::Speciate()
{

}

template <class TaskType,
          class ActivationFunction>
Genome<ActivationFunction> NEAT<TaskType, ActivationFunction>
    ::Crossover(Genome<ActivationFunction>& gen1,
                Genome<ActivationFunction>& gen2)
{
  // New genome's genes.
  std::vector<ConnectionGene> newConnGeneList;
  bool equalFitness = std::abs(gen1.getFitness() - gen2.getFitness()) < 0.01;

  if (!equalFitness)
  {
    Genome<ActivationFunction>& lessFitGenome;

    if (gen1.getFitness() >= gen2.getFitness())
    {
      newConnGeneList = gen1.connectionGeneList;
      lessFitGenome = gen2;
    }
    else
    {
      newConnGeneList = gen2.connectionGeneList;
      lessFitGenome = gen1;
    }

    // Find matching genes.
    size_t k = 0;
    for (size_t i = 0; i < lessFitGenome.connectionGeneList.size(); i++)
    {
      size_t innovID = lessFitGenome.connectionGeneList[i]
          .getGlobalInnovationID();
      for (size_t j = k; j < newConnGeneList.size(); j++)
      {
        if (innovID == newConnGeneList[j].getGlobalInnovationID())
        {
          // If either parent is disabled, preset chance that the inherited gene is disabled.
          if (!newConnGeneList[j].isEnabled() || !lessFitGenome.
                  connectionGeneList[i].isEnabled())
          {
            if (arma::randu<double>() < 0.1)// Placeholder)
              newConnGeneList[j].setEnabled() = false;
            else
              newConnGeneList[j].setEnabled() = true;
          }
          // Weights will be assigned randomly in matching genes.
          if (arma::randu<double>() < 0.5)
            newConnGeneList[j].setWeight() = lessFitGenome.connectionGeneList[i]
                .getWeight();
          k = j;
          break;
        }
      }
    }
  }
  else
  {
    size_t i = 0, j = 0;
    size_t gen1size = gen1.connectionGeneList.size();
    size_t gen2size = gen2.connectionGeneList.size();
    size_t maxSize = gen1size > gen2size ? gen1size : gen2size;
    size_t minSize = gen1size < gen2size ? gen1size : gen2size;
    Genome<ActivationFunction>& maxGenome = gen1size > gen2size ? gen1 : gen2;
    Genome<ActivationFunction>& minGenome = gen1size < gen2size ? gen1 : gen2;
    while (j < minSize)
    {
      size_t innovID1 = maxGenome.connectionGeneList[i].getGlobalInnovationID();
      size_t innovID2 = minGenome.connectionGeneList[i].getGlobalInnovationID();
      if (innovID2 < innovID1)
      {
        if (arma::randu<double>() < 0.5)
          newConnGeneList.push_back(minGenome.connectionGeneList[j++]);
      }
      else if (innovID2 == innovID1)
      {
        if (arma::randu<double>() < 0.5)
          newConnGeneList.push_back(minGenome.connectionGeneList[j]);
        else
          newConnGeneList.push_back(maxGenome.connectionGeneList[i]);
        i++;
        j++;
      }
      else
      {
        if (arma::randu<double>() < 0.5)
          newConnGeneList.push_back(maxGenome.connectionGeneList[i++]);
      }
    }
    while (i < maxSize)
    {
      if (arma::randu<double>() < 0.5)
        newConnGeneList.push_back(maxGenome.connectionGeneList[i++]);
    }
  }

  size_t nextNodeID = gen1.nextNodeID > gen2.nextNodeID ? gen1.nextNodeID : gen2.nextNodeID;

  return Genome<ActivationFunction>(inputNodeCount, outputNodeCount,
        newConnGeneList, nextNodeID, actFn, bias, weightMutationProb,
        weightMutationSize, biasMutationProb, biasMutationSize, 
        nodeAdditionProb, connAdditionProb, isAcyclic);
}

} // namespace neat
} // namespace mlpack

#endif
