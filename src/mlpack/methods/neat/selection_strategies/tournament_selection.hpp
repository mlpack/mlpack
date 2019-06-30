/**
 * @file tournament_selection.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Definition of the Tournament Selection class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NEAT_SELECTION_TOURNAMENT_HPP
#define MLPACK_METHODS_NEAT_SELECTION_TOURNAMENT_HPP

#include <mlpack/prereqs.hpp>

/**
 * Implementation of tournament selection. k random candidates are selected,
 * and then sorted in decreasing order of fitness. Two candidates are then
 * chosen with probabilities depending on their ranking among the contenders.
 */
class TournamentSelection
{
 public:
  /**
   * The method that selects two parents out of the population. It returns a
   * pair of indices of the parents.
   * 
   * @param fitnesses A sorted Armadillo vector of fitnesses in descending
   *    order.
   * @param contenderNum The number of contenders.
   * @param prob The probability of the fittest candidate being chosen.
   */
  static void Select(arma::vec& fitnesses,
                     arma::uvec& selection,
                     const size_t contenderNum,
                     const double prob)
  {
    selection[0] = fitnesses.n_elem;
    selection[1] = fitnesses.n_elem;
    arma::uvec contenders = arma::randi<arma::uvec>(contenderNum,
        arma::distr_param(0, fitnesses.n_elem));
    for (size_t k = 0; k < contenderNum; k++)
    {
      double contenderProb = prob * std::pow(1 - prob, k);
      if (arma::randu<double>() < contenderProb)
      {
        if (selection[0] == fitnesses.n_elem)
          selection[0] = contenders[k];
        else if (selection[1] == fitnesses.n_elem && selection[1] != selection[0])
          selection[1] = contenders[k];
        else if (selection[0] != fitnesses.n_elem && selection[1] != fitnesses.n_elem)
          break;
      }
      if (k == contenderNum - 1)
        k = 0;
    }
  }
};

#endif
