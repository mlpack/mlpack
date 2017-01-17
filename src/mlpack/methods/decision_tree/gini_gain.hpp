/**
 * @file gini_gain.hpp
 * @author Ryan Curtin
 *
 * The GiniImpurity class, which is a fitness function (FitnessFunction) for
 * decision trees.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_GINI_GAIN_HPP
#define MLPACK_METHODS_DECISION_TREE_GINI_GAIN_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree {

/**
 * The Gini gain, a measure of set purity usable as a fitness function
 * (FitnessFunction) for decision trees.  This is the exact same thing as the
 * well-known Gini impurity, but negated---since the decision tree will be
 * trying to maximize gain (and the Gini impurity would need to be minimized).
 */
class GiniGain
{
 public:
  /**
   * Evaluate the Gini impurity on the given set of labels.  RowType should be
   * an Armadillo vector that holds size_t objects.
   *
   * @param labels Set of labels to evaluate Gini impurity on.
   */
  template<typename RowType>
  static double Evaluate(const RowType& labels,
                         const size_t numClasses)
  {
    // Corner case: if there are no elements, the impurity is zero.
    if (labels.n_elem == 0)
      return 0.0;

    arma::Col<size_t> counts(numClasses);
    counts.zeros();
    for (size_t i = 0; i < labels.n_elem; ++i)
      counts[labels[i]]++;

    // Calculate the Gini impurity of the un-split node.
    double impurity = 0.0;
    for (size_t i = 0; i < numClasses; ++i)
    {
      const double f = ((double) counts[i] / (double) labels.n_elem);
      impurity += f * (1.0 - f);
    }

    return -impurity;
  }

  /**
   * Return the range of the Gini impurity for the given number of classes.
   * (That is, the difference between the maximum possible value and the minimum
   * possible value.)
   */
  static double Range(const size_t numClasses)
  {
    // The best possible case is that only one class exists, which gives a Gini
    // impurity of 0.  The worst possible case is that the classes are evenly
    // distributed, which gives n * (1/n * (1 - 1/n)) = 1 - 1/n.
    return 1.0 - (1.0 / double(numClasses));
  }
};

} // namespace tree
} // namespace mlpack

#endif
