/**
 * @file cne_impl.hpp
 * @author Marcus Edel
 * @author Kartik Nighania
 *
 * Conventional Neural Evolution
 * An optimizer that works like biological evolution which selects best
 * candidates based on their fitness scores and creates new generation by
 * mutation and crossover of population.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_CNE_CNE_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_CNE_CNE_IMPL_HPP

#include "cne.hpp"

namespace mlpack {
namespace optimization {

CNE::CNE(const size_t populationSize,
         const size_t maxGenerations,
         const double mutationProb,
         const double mutationSize,
         const double selectPercent,
         const double tolerance,
         const double objectiveChange) :
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    mutationProb(mutationProb),
    mutationSize(mutationSize),
    selectPercent(selectPercent),
    tolerance(tolerance),
    objectiveChange(objectiveChange),
    numElite(0),
    elements(0)
{ /* Nothing to do here. */ }

//! Optimize the function.
template<typename DecomposableFunctionType>
double CNE::Optimize(DecomposableFunctionType& function, arma::mat& iterate)
{
  // Make sure for evolution to work at least four candidates are present.
  if (populationSize < 4)
  {
    throw std::logic_error("CNE::Optimize(): population size should be at least"
        " 4!");
  }

  // Find the number of elite canditates from population.
  numElite = floor(selectPercent * populationSize);

  // Making sure we have even number of candidates to remove and create.
  if ((populationSize - numElite) % 2 != 0)
    numElite--;

  // Terminate if two parents can not be created.
  if (numElite < 2)
  {
    throw std::logic_error("CNE::Optimize(): unable to select two parents. "
        "Increase selection percentage.");
  }

  // Terminate if at least two childs are not possible.
  if ((populationSize - numElite) < 2)
  {
    throw std::logic_error("CNE::Optimize(): no space to accomodate even 2 "
        "children. Increase population size.");
  }

  // Set the population size and fill random values [0,1].
  population = arma::randu(iterate.n_rows, iterate.n_cols, populationSize);

  // Store the number of elements in a cube slice or a matrix column.
  elements = population.n_rows * population.n_cols;

  // initializing helper variables.
  fitnessValues.set_size(populationSize);

  Log::Info << "CNE initialized successfully. Optimization started."
      << std::endl;

  // Find the fitness before optimization using given iterate parameters.
  size_t lastBestFitness = function.Evaluate(iterate);

  // Iterate until maximum number of generations is obtained.
  for (size_t gen = 1; gen <= maxGenerations; gen++)
  {
    // Calculating fitness values of all candidates.
    for (size_t i = 0; i < populationSize; i++)
    {
       // Select a candidate and insert the parameters in the function.
       iterate = population.slice(i);

       // Find fitness of candidate.
       fitnessValues[i] = function.Evaluate(iterate);
    }

    Log::Info << "Generation number: " << gen << " best fitness = "
        << fitnessValues.min() << std::endl;

    // Create next generation of species.
    Reproduce();

    // Check for termination criteria.
    if (tolerance >= fitnessValues.min())
    {
      Log::Info << "CNE::Optimize(): terminating. Given fitness criteria "
          << tolerance << " > " << fitnessValues.min() << "." << std::endl;
      break;
    }

    // Check for termination criteria.
    if (lastBestFitness - fitnessValues.min() < objectiveChange)
    {
      Log::Info << "CNE::Optimize(): terminating. Fitness history change "
          << (lastBestFitness - fitnessValues.min())
          << " < " << objectiveChange << "." << std::endl;
      break;
    }

    // Store the best fitness of present generation.
    lastBestFitness = fitnessValues.min();
  }

  // Set the best candidate into the network parameters.
  iterate = population.slice(index(0));

  return function.Evaluate(iterate);
}

//! Reproduce candidates to create the next generation.
void CNE::Reproduce()
{
  // Sort fitness values. Smaller fitness value means better performance.
  index = arma::sort_index(fitnessValues);

  // First parent.
  size_t mom;

  // Second parent.
  size_t dad;

  for (size_t i = numElite; i < populationSize - 1; i++)
  {
    // Select 2 different parents from elite group randomly [0, numElite).
    mom = mlpack::math::RandInt(0, numElite);
    dad = mlpack::math::RandInt(0, numElite);

    // Making sure both parents are not the same.
    if (mom == dad)
    {
      if (dad != numElite - 1)
      {
        dad++;
      }
      else
      {
        dad--;
      }
    }

    // Parents generate 2 children replacing the dropped-out candidates.
    // Also finding the index of these candidates in the population matrix.
    Crossover(index[mom], index[dad], index[i], index[i + 1]);
  }

  // Mutating the weights with small noise values.
  // This is done to bring change in the next generation.
  Mutate();
}

//! Crossover parents to create new children.
void CNE::Crossover(const size_t mom,
                    const size_t dad,
                    const size_t child1,
                    const size_t child2)
{
  // Replace the candidates with parents at their place.
  population.slice(child1) = population.slice(mom);
  population.slice(child2) = population.slice(dad);

  // Preallocate random selection vector (values between 0 and 1).
  arma::vec selection = arma::randu(elements);

  // Randomly alter mom and dad genome weights to get two different children.
  for (size_t i = 0; i < elements; i++)
  {
    // Using it to alter the weights of the children.
    if (selection(i) > 0.5)
    {
      population.slice(child1)(i) = population.slice(mom)(i);
      population.slice(child2)(i) = population.slice(dad)(i);
    }
    else
    {
      population.slice(child1)(i) = population.slice(dad)(i);
      population.slice(child2)(i) = population.slice(mom)(i);
    }
  }
}

//! Modify weights with some noise for the evolution of next generation.
void CNE::Mutate()
{
  // Mutate the whole matrix with the given rate and probability.
  // The best candidate is not altered.
  for (size_t i = 1; i < populationSize; i++)
  {
    population.slice(index(i)) += (arma::randu(
        population.n_rows, population.n_cols) < mutationProb) %
        (mutationSize * arma::randn(population.n_rows, population.n_cols));
  }
}

} // namespace optimization
} // namespace mlpack

#endif
