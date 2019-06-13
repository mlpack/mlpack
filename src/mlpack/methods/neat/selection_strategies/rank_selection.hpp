/**
 * @file neat.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Definition of the Rank Selection class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NEAT_SELECTION_RANK_HPP
#define MLPACK_METHODS_NEAT_SELECTION_RANK_HPP

#include <mlpack/prereqs.hpp>

/**
 * Implementation of rank selection. The candidates are assigned ranks based
 * on their fitness, i.e. the candidate with the highest fitness is given rank
 * 1, the next is given rank 2, and so on. Each candidate is then chosen with
 * probability (N - rank) * 2 / (N * (N + 1)).
 */
class RankSelection
{
 public:
  /**
   * The method that selects two parents out of the population. It returns a
   * pair of indices of the parents.
   * 
   * @param fitnesses A sorted Armadillo vector of fitnesses in ascending
   *    order.
   */
  static std::pair<size_t, size_t> Select(arma::vec& fitnesses)
  {
    size_t parent1 = fitnesses.n_elem, parent2 = fitnesses.n_elem;
    size_t pos = 0, size = fitnesses.n_elem;
    // Choose first genome.
    while (parent1 == fitnesses.n_elem)
    {
      if (pos >= size)
        pos = 0;
      double prob = (size - pos) * 2 / (size * (size + 1));
      if (arma::randu<double>() < prob)
        parent1 = pos;
      pos++;
    }

    // Choose second genome.
    pos = 0;
    while (parent2 == fitnesses.n_elem && parent1 != parent2)
    {
      if (pos >= size)
        pos = 0;
      double prob = (size - pos) * 2 / (size * (size + 1));
      if (arma::randu<double>() < prob)
        parent2 = pos;
      pos++;
    }
    return std::make_pair(parent1, parent2);
  }
};

#endif
