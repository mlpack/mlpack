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
          class ActivationFunction,
          class SelectionPolicy>
NEAT<TaskType, ActivationFunction, SelectionPolicy>::
    NEAT(TaskType& task,
         const size_t inputNodeCount,
         const size_t outputNodeCount,
         const size_t popSize,
         const size_t maxGen,
         const size_t numSpecies,
         const double bias,
         const double weightMutationProb,
         const double weightMutationSize,
         const double biasMutationProb,
         const double biasMutationSize,
         const double nodeAdditionProb,
         const double connAdditionProb,
         const double connDeletionProb,
         const double disableProb,
         const double elitismProp,
         const bool isAcyclic):
    task(task),
    inputNodeCount(inputNodeCount),
    outputNodeCount(outputNodeCount),
    popSize(popSize),
    maxGen(maxGen),
    numSpecies(numSpecies),
    bias(bias),
    weightMutationProb(weightMutationProb),
    weightMutationSize(weightMutationSize),
    biasMutationProb(biasMutationProb),
    biasMutationSize(biasMutationSize),
    nodeAdditionProb(nodeAdditionProb),
    connAdditionProb(connAdditionProb),
    connDeletionProb(connDeletionProb),
    disableProb(disableProb),
    elitismProp(elitismProp),
    isAcyclic(isAcyclic)
{ /* Nothing to do here yet */ }

// The main loop of the NEAT algorithm. Returns the best genome.
template <class TaskType,
          class ActivationFunction,
          class SelectionPolicy>
Genome<ActivationFunction> NEAT<TaskType, ActivationFunction, SelectionPolicy>
    ::Train()
{
  Genome<ActivationFunction>::nextInnovID = 0;

  // Initialize.
  for (size_t i = 0; i < popSize; i++)
  {
    genomeList.emplace_back(Genome<ActivationFunction>(inputNodeCount,
        outputNodeCount, bias, weightMutationProb,
        weightMutationSize, biasMutationProb, biasMutationSize,
        nodeAdditionProb, connAdditionProb, connDeletionProb, isAcyclic));
  }
  speciesList = std::vector<std::vector<Genome<ActivationFunction>>>(numSpecies);
  Speciate(true);

  arma::vec fitnesses(popSize);

  // Main loop.
  for (size_t gen = 0; gen < maxGen; gen++)
  {
    Genome<ActivationFunction>::mutationBuffer.clear();
    std::cout << "Evaluating" << std::endl;
    arma::vec fitnesses(popSize);
    #pragma omp parallel for
    for (size_t i = 0; i < popSize; i++)
    {
      genomeList[i].Fitness() = task.Evaluate(genomeList[i]);
      fitnesses[i] = genomeList[i].Fitness();
    }
    std::cout << "Max fitness in generation " << gen << "is " << arma::max(fitnesses) << std::endl;
    if (gen == maxGen - 1) break;
    std::cout << "Reproducing" << std::endl;
    Reproduce();
    std::cout << "Speciating" << std::endl;
    Speciate(false); 
  }

  // Find best genome.
  size_t maxIdx = -1;
  double max = -DBL_MAX;
  for (size_t i = 0; i < popSize; i++)
  {
    if (genomeList[i].Fitness() > max)
    {
      maxIdx = i;
      max = genomeList[i].Fitness();
    }
  }

  return genomeList[maxIdx];
}

// Creates the next generation through reproduction.
template <class TaskType,
          class ActivationFunction,
          class SelectionPolicy>
void NEAT<TaskType, ActivationFunction, SelectionPolicy>::Reproduce()
{
  // The mean fitnesses of the species.
  arma::vec meanFitnesses(numSpecies, arma::fill::zeros);
  // The next generations sizes of the species.
  arma::vec speciesSize(numSpecies, arma::fill::zeros);
  // The number of elite membes in each species.
  arma::vec numElite(numSpecies, arma::fill::zeros);

  // Find the mean fitnesses.
  for (size_t i = 0; i < numSpecies; i++)
  {
    if (speciesList[i].size() == 0)
      continue;
    for (size_t j = 0; j < speciesList[i].size(); j++)
      meanFitnesses[i] += speciesList[i][j].Fitness();
    meanFitnesses[i] /= speciesList[i].size();
  }
  double totalMeanFitness = arma::accu(meanFitnesses);

  // Allot the sizes of the species.
  for (size_t i = 0; i < numSpecies; i++)
  {
    speciesSize[i] = std::round(meanFitnesses[i] / totalMeanFitness * popSize);
    // Handle special case where fitnesses are negative.
    if (speciesSize[i] < 0)
      speciesSize[i] = 0;
  }

  // If the total allotted size is less than the population, fill it.
  int delta = (int)popSize - (int)arma::accu(speciesSize);
  if (delta > 0)
  {
    size_t i = 0;
    while (delta > 0)
    {
      speciesSize[i++]++;
      delta--;
      if(i >= numSpecies)
        i = 0;
    }
  }
  // If the total allotted size is more than the population, remove the excess.
  else if (delta < 0)
  {
    size_t i = 0;
    while (delta < 0)
    {
      speciesSize[i++]--;
      delta++;
      if(i >= numSpecies)
        i = 0;
    }
  }

  // Find the number of elite members in each species.
  for (size_t i = 0; i < numSpecies; i++)
  {
    numElite[i] = std::round(elitismProp * speciesSize[i]);
    // This ensures that the best genome from every species will be saved.
    if (numElite[i] == 0)
      numElite[i] = 1;
    if (numElite[i] > speciesList[i].size())
      numElite[i] = speciesList[i].size();
  }

  // Prepare the centroids matrix for recalculation.
  centroids.fill(0);

  // Crossover and add the new genomes.
  genomeList.clear();
  for (size_t i = 0; i < numSpecies; i++)
  {
    // Handle the case where the species should be empty.
    if (speciesSize[i] == 0)
      continue;

    size_t currentSize = genomeList.size();
    std::sort(speciesList[i].begin(), speciesList[i].begin(), compareGenome);
    arma::vec fitnesses(speciesList[i].size());

    // Let us update the species' centroids.
    if (speciesSize[i] == 1)
    {
      for (size_t j = 0; j < speciesList[i][0].connectionGeneList.size(); j++)
      {
        size_t innovID = speciesList[i][0].connectionGeneList[j].InnovationID();
        centroids(innovID, i) = speciesList[i][0].connectionGeneList[j].Weight();
      }
    }
    else
    {
      for (size_t k = 0; k < numElite[i]; k++)
      {
        for (size_t j = 0; j < speciesList[i][k].connectionGeneList.size(); j++)
        {
          size_t innovID = speciesList[i][k].connectionGeneList[j].InnovationID();
          centroids(innovID, i) += (speciesList[i][k].connectionGeneList[j].Weight()) / numElite[i];
        }
      }
    }

    // Handle the case where the species has only one member.
    if (speciesList[i].size() == 1)
    {
      genomeList.push_back(speciesList[i][0]);

      // Fill the species with mutated members of the population.
      while (genomeList.size() - currentSize < speciesSize[i])
      {
        genomeList.push_back(speciesList[i][0]);
        genomeList[genomeList.size() - 1].Mutate();
      }
      continue;
    }
    
    // Fill the elite members.
    for (size_t j = 0; j < numElite[i]; j++)
      genomeList.push_back(speciesList[i][j]);

    // Fill the fitnesses.
    for (size_t j = 0; j < fitnesses.n_elem; j++)
      fitnesses[j] = speciesList[i][j].Fitness();

    while (genomeList.size() < speciesSize[i] + currentSize)
    {
      arma::uvec selection(2);
      SelectionPolicy::Select(fitnesses, selection);
      Genome<ActivationFunction> child = Crossover(speciesList[i][selection[0]],
          speciesList[i][selection[1]]);
      genomeList.push_back(child);
      genomeList[genomeList.size() - 1].Mutate();
    }
  }
}

// Speciates the genomes.
template <class TaskType,
          class ActivationFunction,
          class SelectionPolicy>
void NEAT<TaskType, ActivationFunction, SelectionPolicy>::Speciate(bool init)
{
  // Translate the genome into points in space.
  arma::mat data(Genome<ActivationFunction>::nextInnovID, popSize, arma::fill::zeros);
  for (size_t i = 0; i < popSize; i++)
  {
    for (size_t j = 0; j < genomeList[i].connectionGeneList.size(); j++)
    {
      size_t innovID = genomeList[i].connectionGeneList[j].InnovationID();
      data(innovID, i) = genomeList[i].connectionGeneList[j].Weight();
    }
  }

  arma::Row<size_t> assignments(popSize, arma::fill::zeros);
  kmeans::KMeans<metric::EuclideanDistance, kmeans::SampleInitialization,
      kmeans::MaxVarianceNewCluster, kmeans::CoverTreeDualTreeKMeans> k;
  if (init)
  {
    centroids = arma::mat(data.n_rows, numSpecies, arma::fill::zeros);
    k.Cluster(data, numSpecies, assignments, centroids);
  }
  else
  {
    centroids.resize(data.n_rows, numSpecies);
    k.Cluster(data, numSpecies, assignments, centroids, false, true);
  }

  // Clear the old species list to make space for a new one.
  for (size_t i = 0; i < numSpecies; i++)
    speciesList[i].clear();

  // Assign the genomes to their species.
  for (size_t i = 0; i < assignments.n_elem; i++)
    speciesList[assignments[i]].push_back(genomeList[i]);
}

// Crosses over two genomes and creates a child.
template <class TaskType,
          class ActivationFunction,
          class SelectionPolicy>
Genome<ActivationFunction> NEAT<TaskType, ActivationFunction, SelectionPolicy>
    ::Crossover(Genome<ActivationFunction>& gen1,
                Genome<ActivationFunction>& gen2)
{
  // New genome's genes.
  std::vector<ConnectionGene> newConnGeneList;
  bool equalFitness = std::abs(gen1.Fitness() - gen2.Fitness()) < 0.001;
  if (!equalFitness || isAcyclic)
  {
    Genome<ActivationFunction> lessFitGenome = gen1;
    std::vector<size_t> nodeDepths;
    size_t nextNodeID;

    if (equalFitness)
    {
      if (arma::randu<double>() < 0.5)
      {
        newConnGeneList = gen1.connectionGeneList;
        nextNodeID = gen1.getNodeCount();
        nodeDepths = gen1.nodeDepths;
        lessFitGenome = gen2;
      }
      else
      {
        newConnGeneList = gen2.connectionGeneList;
        nextNodeID = gen2.getNodeCount();
        nodeDepths = gen2.nodeDepths;
        lessFitGenome = gen1;
      }
    }
    else if (gen1.Fitness() > gen2.Fitness())
    {
      newConnGeneList = gen1.connectionGeneList;
      nextNodeID = gen1.getNodeCount();
      nodeDepths = gen1.nodeDepths;
      lessFitGenome = gen2;
    }
    else
    {
      newConnGeneList = gen2.connectionGeneList;
      nextNodeID = gen2.getNodeCount();
      nodeDepths = gen2.nodeDepths;
      lessFitGenome = gen1;
    }

    // Find matching genes.
    for (size_t i = 0; i < lessFitGenome.connectionGeneList.size(); i++)
    {
      size_t innovID = lessFitGenome.connectionGeneList[i]
          .InnovationID();
      for (size_t j = 0; j < newConnGeneList.size(); j++)
      {
        if (innovID == newConnGeneList[j].InnovationID())
        {
          // If either parent is disabled, preset chance that the inherited gene is disabled.
          if (!isAcyclic && !newConnGeneList[j].Enabled())
          {
            if (arma::randu<double>() < disableProb)
              newConnGeneList[j].Enabled() = true;
            else
              newConnGeneList[j].Enabled() = false;
          }
          // Weights will be assigned randomly in matching genes.
          if (arma::randu<double>() < 0.5)
            newConnGeneList[j].Weight() = lessFitGenome.connectionGeneList[i]
                .Weight();
          break;
        }
      }
    }

    if (isAcyclic)
    {
      return Genome<ActivationFunction>(newConnGeneList, nodeDepths, 
        inputNodeCount, outputNodeCount, nextNodeID, bias, weightMutationProb,
        weightMutationSize, biasMutationProb, biasMutationSize,
        nodeAdditionProb, connAdditionProb, connDeletionProb, isAcyclic);
    }
    else
    {
      return Genome<ActivationFunction>(newConnGeneList, inputNodeCount,
        outputNodeCount, nextNodeID, bias, weightMutationProb,
        weightMutationSize, biasMutationProb, biasMutationSize,
        nodeAdditionProb, connAdditionProb, connDeletionProb, isAcyclic);
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
      size_t innovID1 = maxGenome.connectionGeneList[i].InnovationID();
      size_t innovID2 = minGenome.connectionGeneList[j].InnovationID();
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

    size_t nextNodeID = gen1.getNodeCount() > gen2.getNodeCount() ? gen1.getNodeCount() : gen2.getNodeCount();

    return Genome<ActivationFunction>(newConnGeneList, inputNodeCount,
          outputNodeCount, nextNodeID, bias, weightMutationProb,
          weightMutationSize, biasMutationProb, biasMutationSize,
          nodeAdditionProb, connAdditionProb, connDeletionProb, isAcyclic);
  }
}

template <class TaskType,
          class ActivationFunction,
          class SelectionPolicy>
bool NEAT<TaskType, ActivationFunction, SelectionPolicy>::
    compareGenome(Genome<ActivationFunction>& gen1,
                  Genome<ActivationFunction>& gen2)
{
  return gen1.Fitness() > gen2.Fitness();
}

} // namespace neat
} // namespace mlpack

#endif
