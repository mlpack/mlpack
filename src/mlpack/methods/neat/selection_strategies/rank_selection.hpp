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
   * The method that selects parents out of the population. It returns 
   * indices of the parents.
   * 
   * @param fitnesses A sorted Armadillo vector of fitnesses in ascending
   *    order.
   * @param selection The selected indices.
   */
  static void Select(const arma::vec& fitnesses, arma::uvec& selection)
  {
    selection.fill(0);
    size_t size = fitnesses.n_elem;
    double denom = std::pow(size, 2);
    for (size_t i = 0; i < selection.n_elem; i++)
    {
      size_t pos = 0;
      while (pos < size)
      {
        double prob = (double)(size - pos) / denom;
        if (arma::randu() < prob)
        {
          selection[i] = pos;
          break;
        }
        pos++;
      }
    }
  }
};

#endif
