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
   * The method that selects two parents out of the population. It returns a
   * pair of indices of the parents.
   * 
   * @param fitnesses A sorted Armadillo vector of fitnesses in descending
   *    order.
   */
  static void Select(arma::vec& fitnesses, arma::uvec& selection)
  {
    selection[0] = fitnesses.n_elem;
    selection[1] = fitnesses.n_elem;
    double totalFitness = arma::accu(fitnesses);
    double prob = 0;
    for (size_t k = 0; k < fitnesses.n_elem; k++)
    {
      prob += fitnesses[k] / totalFitness;
      if (arma::randu<double>() < prob)
      {
        if (selection[0] == fitnesses.n_elem)
          selection[0] = k;
        else if (selection[1] == fitnesses.n_elem && selection[0] != selection[1])
          selection[1] = k;
        else if (selection[1] != fitnesses.n_elem && selection[0] != fitnesses.n_elem)
          break;
      }
      if (k == fitnesses.n_elem - 1)
      {
        k = 0;
        prob = 0;
      }
    }
  }
};

#endif
