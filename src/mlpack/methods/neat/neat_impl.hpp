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
         const double initialBias,
         const double initialWeight,
         const double weightMutationProb,
         const double weightMutationSize,
         const double biasMutationProb,
         const double biasMutationSize,
         const double nodeAdditionProb,
         const double connAdditionProb,
         const double connDeletionProb,
         const double disableProb,
         const double elitismProp,
         const double finalFitness,
         const size_t complexityThreshold,
         const size_t maxSimplifyGen,
         const bool isAcyclic):
    task(task),
    inputNodeCount(inputNodeCount),
    outputNodeCount(outputNodeCount),
    popSize(popSize),
    maxGen(maxGen),
    numSpecies(numSpecies),
    initialBias(initialBias),
    initialWeight(initialWeight),
    weightMutationProb(weightMutationProb),
    weightMutationSize(weightMutationSize),
    biasMutationProb(biasMutationProb),
    biasMutationSize(biasMutationSize),
    nodeAdditionProb(nodeAdditionProb),
    connAdditionProb(connAdditionProb),
    connDeletionProb(connDeletionProb),
    disableProb(disableProb),
    elitismProp(elitismProp),
    finalFitness(finalFitness),
    complexityThreshold(complexityThreshold),
    currentComplexityCeiling(complexityThreshold),
    meanComplexity(0),
    lastTransitionGen(0),
    maxSimplifyGen(maxSimplifyGen),
    searchMode(0),
    isAcyclic(isAcyclic),
    contenderNum(0),
    tournamentSelectProb(0)
{ /* Nothing to do here yet */ }

// The main loop of the NEAT algorithm. Returns the best genome.
template <class TaskType,
          class ActivationFunction,
          class SelectionPolicy>
void NEAT<TaskType, ActivationFunction, SelectionPolicy>
    ::Train(Genome<ActivationFunction>& genome)
{
  Genome<ActivationFunction>::nextInnovID = 0;
  Genome<ActivationFunction>::mutationBuffer.clear();

  // Initialize.
  Initialize();
  speciesList = std::vector<std::vector<Genome<ActivationFunction>>>
      (numSpecies);
  Speciate(true);

  // Main loop.
  for (size_t gen = 0; gen < maxGen; gen++)
  {
    meanComplexity = 0;
    arma::vec fitnesses(popSize);

    #pragma omp parallel for
    for (size_t i = 0; i < popSize; i++)
    {
      if (complexityThreshold != 0)
        meanComplexity += (double)genomeList[i].Complexity() / popSize;
      genomeList[i].Fitness() = task.Evaluate(genomeList[i]);
      fitnesses[i] = genomeList[i].Fitness();
    }

    // Check termination criteria.
    if (finalFitness != 0 && fitnesses.max() >= finalFitness)
      break;
    else if (gen == maxGen - 1)
      break;

    // Set search mode.
    if (meanComplexity > currentComplexityCeiling)
    {
      if (searchMode == 0)
        lastTransitionGen = gen;
      searchMode = 1;
    }
    else
    {
      if (searchMode == 1)
        lastTransitionGen = gen;
      searchMode = 0;
    }

    if (lastTransitionGen + maxSimplifyGen == gen && searchMode == 1)
    {
      currentComplexityCeiling += complexityThreshold;
      searchMode = 0;
    }

    Log::Info << "The maximum fitness in generation " << gen << " is " <<
        fitnesses.max() << std::endl;

    Speciate(false);
    Reproduce();
  }

  // Find best genome.
  size_t maxIdx = 0;
  double maxFitness = genomeList[0].Fitness();
  for (size_t i = 1; i < popSize; i++)
  {
    if (genomeList[i].Fitness() > maxFitness)
    {
      maxIdx = i;
      maxFitness = genomeList[i].Fitness();
    }
  }

  genome = genomeList[maxIdx];
}

template <class TaskType,
          class ActivationFunction,
          class SelectionPolicy>
void NEAT<TaskType, ActivationFunction, SelectionPolicy>::
    Step(Genome<ActivationFunction>& genome)
{
  // If this is true, the population has not been initialized.
  if (genomeList.size() == 0)
  {
    Initialize();
    speciesList = std::vector<std::vector<Genome<ActivationFunction>>>
        (numSpecies);
    Speciate(true);
  }

  #pragma omp parallel for
  for (size_t i = 0; i < popSize; i++)
    genomeList[i].Fitness() = task.Evaluate(genomeList[i]);
  Speciate(false);
  Reproduce();

  // Find best genome.
  size_t maxIdx = 0;
  double maxFitness = genomeList[0].Fitness();
  for (size_t i = 1; i < popSize; i++)
  {
    if (genomeList[i].Fitness() > maxFitness)
    {
      maxIdx = i;
      maxFitness = genomeList[i].Fitness();
    }
  }

  genome = genomeList[maxIdx];
}

// Speciates the genomes.
template <class TaskType,
          class ActivationFunction,
          class SelectionPolicy>
void NEAT<TaskType, ActivationFunction, SelectionPolicy>::Speciate(bool init)
{
  // Translate the genome into points in space.
  arma::mat data(Genome<ActivationFunction>::nextInnovID, popSize,
      arma::fill::zeros);
  for (size_t i = 0; i < popSize; i++)
  {
    for (size_t j = 0; j < genomeList[i].connectionGeneList.size(); j++)
    {
      if (genomeList[i].connectionGeneList[j].Enabled())
      {
        size_t innovID = genomeList[i].connectionGeneList[j].InnovationID();
        data(innovID, i) = genomeList[i].connectionGeneList[j].Weight();
      }
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
      if (i >= numSpecies)
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
      if (i >= numSpecies)
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
    else if (numElite[i] > speciesList[i].size())
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
    arma::vec fitnesses(speciesList[i].size());
    std::sort(speciesList[i].begin(), speciesList[i].end(),  compareGenome);

    // Let us update the species' centroids.
    if (speciesSize[i] == 1)
    {
      for (size_t j = 0; j < speciesList[i][0].connectionGeneList.size(); j++)
      {
        size_t innovID = speciesList[i][0].connectionGeneList[j].InnovationID();
        centroids(innovID, i) = speciesList[i][0].connectionGeneList[j]
            .Weight();
      }
    }
    else
    {
      for (size_t k = 0; k < numElite[i]; k++)
      {
        for (size_t j = 0; j < speciesList[i][k].connectionGeneList.size(); j++)
        {
          size_t innovID = speciesList[i][k].connectionGeneList[j].
              InnovationID();
          centroids(innovID, i) += (speciesList[i][k].connectionGeneList[j]
              .Weight()) / numElite[i];
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
      // Complexifying.
      if (searchMode == 0)
      {
        arma::uvec selection(2);
        Select(fitnesses, selection, contenderNum, tournamentSelectProb);
        if (selection[0] == selection[1])
        {
          genomeList.push_back(speciesList[i][selection[0]]);
          genomeList[genomeList.size() - 1].NodeAdditionProb() =
              nodeAdditionProb;
          genomeList[genomeList.size() - 1].ConnAdditionProb() =
              connAdditionProb;
          genomeList[genomeList.size() - 1].ConnDeletionProb() =
              connDeletionProb;
        }
        else
        {
          Genome<ActivationFunction> child = Crossover(speciesList[i]
              [selection[0]], speciesList[i][selection[1]]);
          genomeList.push_back(child);
        }
        genomeList[genomeList.size() - 1].Mutate();
      }
      // Simplifying.
      else
      {
        arma::uvec selection(1);
        Select(fitnesses, selection, contenderNum, tournamentSelectProb);

        genomeList.push_back(speciesList[i][selection[0]]);

        // Change mutation parameters for simplification.
        genomeList[genomeList.size() - 1].NodeAdditionProb() = 0;
        genomeList[genomeList.size() - 1].ConnAdditionProb() = 0;
        genomeList[genomeList.size() - 1].ConnDeletionProb() = 0.8;

        genomeList[genomeList.size() - 1].Mutate();
      }
    }
  }
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
  Genome<ActivationFunction> lessFitGenome = gen1;
  std::vector<size_t> nodeDepths;
  size_t nextNodeID;
  double bias;

  if (equalFitness)
  {
    if (arma::randu() < 0.5)
    {
      newConnGeneList = gen1.connectionGeneList;
      nextNodeID = gen1.NodeCount();
      nodeDepths = gen1.nodeDepths;
      bias = gen1.Bias();
      lessFitGenome = gen2;
    }
    else
    {
      newConnGeneList = gen2.connectionGeneList;
      nextNodeID = gen2.NodeCount();
      nodeDepths = gen2.nodeDepths;
      bias = gen2.Bias();
      lessFitGenome = gen1;
    }
  }
  else if (gen1.Fitness() > gen2.Fitness())
  {
    newConnGeneList = gen1.connectionGeneList;
    nextNodeID = gen1.NodeCount();
    nodeDepths = gen1.nodeDepths;
    bias = gen1.Bias();
    lessFitGenome = gen2;
  }
  else
  {
    newConnGeneList = gen2.connectionGeneList;
    nextNodeID = gen2.NodeCount();
    nodeDepths = gen2.nodeDepths;
    bias = gen2.Bias();
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
        if (!isAcyclic && !newConnGeneList[j].Enabled())
        {
          if (arma::randu() < disableProb)
            newConnGeneList[j].Enabled() = true;
          else
            newConnGeneList[j].Enabled() = false;
        }

        // Weights will be assigned randomly in matching genes.
        if (arma::randu() < 0.5)
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

template <class TaskType,
          class ActivationFunction,
          class SelectionPolicy>
bool NEAT<TaskType, ActivationFunction, SelectionPolicy>::
    compareGenome(Genome<ActivationFunction>& gen1,
                  Genome<ActivationFunction>& gen2)
{
  return gen1.Fitness() > gen2.Fitness();
}

template <class TaskType,
          class ActivationFunction,
          class SelectionPolicy>
template <typename Task>
typename std::enable_if<
    HasStartingGenome<Task, std::vector<ConnectionGene>(Task::*)()>::value,
    void>::type
NEAT<TaskType, ActivationFunction, SelectionPolicy>::Initialize()
{
  std::vector<ConnectionGene> connGeneList = task.startingGenome();
  size_t maxInnovID = 0, maxNodeID = 0;
  for (size_t i = 0; i < connGeneList.size(); i++)
  {
    if (connGeneList[i].InnovationID() > maxInnovID)
      maxInnovID = connGeneList[i].InnovationID();
    if (connGeneList[i].Source() > maxNodeID)
      maxNodeID = connGeneList[i].Source();
    if (connGeneList[i].Target() > maxNodeID)
      maxNodeID = connGeneList[i].Target();
  }
  Genome<ActivationFunction>::nextInnovID = maxInnovID + 1;

  for (size_t i = 0; i < popSize; i++)
  {
    if (isAcyclic)
    {
      double bias = startingGenome.Bias() + arma::randn();
      genomeList.emplace_back(Genome<ActivationFunction>(connGeneList,
      inputNodeCount, outputNodeCount, maxNodeID + 1, bias, weightMutationProb,
      weightMutationSize, biasMutationProb, biasMutationSize, nodeAdditionProb,
      connAdditionProb, connDeletionProb, isAcyclic));

      // Let's find the node depths.
      genomeList[i].nodeDepths.resize(genomeList[i].NodeCount(), 0);
      for (size_t j = 0; j <= inputNodeCount; j++)
        genomeList[i].Traverse(j);

      genomeList[i].MutateWeights();
    }
    else
    {
      double bias = startingGenome.Bias() + arma::randn();
      genomeList.emplace_back(Genome<ActivationFunction>(connGeneList,
      inputNodeCount, outputNodeCount, maxNodeID + 1, bias, weightMutationProb,
      weightMutationSize, biasMutationProb, biasMutationSize, nodeAdditionProb,
      connAdditionProb, connDeletionProb, isAcyclic));

      genomeList[i].MutateWeights();
    }
  }
}

template <class TaskType,
          class ActivationFunction,
          class SelectionPolicy>
template <typename Task>
typename std::enable_if<
    !HasStartingGenome<Task, std::vector<ConnectionGene>(Task::*)()>::value,
    void>::type
NEAT<TaskType, ActivationFunction, SelectionPolicy>::Initialize()
{
  if (startingGenome.connectionGeneList.size() == 0)
  {
    for (size_t i = 0; i < popSize; i++)
    {
      double bias = initialBias + arma::randn();
      genomeList.emplace_back(Genome<ActivationFunction>(inputNodeCount,
          outputNodeCount, bias, initialWeight, weightMutationProb,
          weightMutationSize, biasMutationProb, biasMutationSize,
          nodeAdditionProb, connAdditionProb, connDeletionProb, isAcyclic));
      genomeList[i].MutateWeights();
    }
  }
  else
  {
    for (size_t i = 0; i < popSize; i++)
    {
      genomeList.push_back(startingGenome);
      genomeList[i].MutateWeights();
    }
  }
}

template <class TaskType,
          class ActivationFunction,
          class SelectionPolicy>
template <typename Archive>
void NEAT<TaskType, ActivationFunction, SelectionPolicy>::
    serialize(Archive& ar,
              const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(task);
  ar & BOOST_SERIALIZATION_NVP(inputNodeCount);
  ar & BOOST_SERIALIZATION_NVP(outputNodeCount);
  ar & BOOST_SERIALIZATION_NVP(popSize);
  ar & BOOST_SERIALIZATION_NVP(maxGen);
  ar & BOOST_SERIALIZATION_NVP(numSpecies);
  ar & BOOST_SERIALIZATION_NVP(initialBias);
  ar & BOOST_SERIALIZATION_NVP(initialWeight);
  ar & BOOST_SERIALIZATION_NVP(weightMutationProb);
  ar & BOOST_SERIALIZATION_NVP(weightMutationSize);
  ar & BOOST_SERIALIZATION_NVP(biasMutationProb);
  ar & BOOST_SERIALIZATION_NVP(biasMutationSize);
  ar & BOOST_SERIALIZATION_NVP(nodeAdditionProb);
  ar & BOOST_SERIALIZATION_NVP(connAdditionProb);
  ar & BOOST_SERIALIZATION_NVP(connDeletionProb);
  ar & BOOST_SERIALIZATION_NVP(disableProb);
  ar & BOOST_SERIALIZATION_NVP(elitismProp);
  ar & BOOST_SERIALIZATION_NVP(finalFitness);
  ar & BOOST_SERIALIZATION_NVP(isAcyclic);

  if (genomeList.size() != 0)
  {
    ar & BOOST_SERIALIZATION_NVP(genomeList);
    ar & BOOST_SERIALIZATION_NVP(speciesList);
    ar & BOOST_SERIALIZATION_NVP(centroids);
  }
}

template <class TaskType,
          class ActivationFunction,
          class SelectionPolicy>
template <typename Policy>
typename std::enable_if<
    std::is_same<Policy, TournamentSelection>::value, void>::type
NEAT<TaskType, ActivationFunction, SelectionPolicy>::
    Select(arma::vec& fitnesses,
           arma::uvec& selection,
           const size_t contenderNum,
           const double prob)
{
  if (contenderNum == 0)
    Log::Fatal << "contenderNum must be nonzero.";

  Policy::Select(fitnesses, selection, contenderNum, prob);
}

template <class TaskType,
          class ActivationFunction,
          class SelectionPolicy>
template <typename Policy>
typename std::enable_if<
    !std::is_same<Policy, TournamentSelection>::value, void>::type
NEAT<TaskType, ActivationFunction, SelectionPolicy>::
    Select(arma::vec& fitnesses,
           arma::uvec& selection,
           const size_t /* Unused */,
           const double /* Unused */)
{
  Policy::Select(fitnesses, selection);
}

} // namespace neat
} // namespace mlpack

#endif
