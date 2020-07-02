/**
 * @file roulette_selection.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Definition of the Roulette Selection class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NEAT_SELECTION_ROULETTE_HPP
#define MLPACK_METHODS_NEAT_SELECTION_ROULETTE_HPP

#include <mlpack/prereqs.hpp>

class RouletteSelection
{
 public:
  /**
   * The method that selects parents out of the population. It returns a
   * pair of indices of the parents.
   * 
   * @param fitnesses A sorted Armadillo vector of fitnesses in descending
   *    order.
   * @param selection The selected indices.
   */
  static void Select(const arma::vec& fitnesses, arma::uvec& selection)
  {
    selection.fill(0);
    double totalFitness = arma::accu(fitnesses);
    for (size_t i = 0; i < selection.n_elem; i++)
    {
      size_t k = fitnesses.n_elem - 1;
      double prob = 0, randNum = arma::randu();
      while (k != 0)
      {
        prob += fitnesses[k] / totalFitness;
        if (randNum < prob)
        {
          selection[i] = k;
          prob = 0;
          break;
        }
        k--;
      }
    }
  }
};

#endif
