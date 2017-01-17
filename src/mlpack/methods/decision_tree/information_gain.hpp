/**
 * @file information_gain.hpp
 * @author Ryan Curtin
 *
 * An implementation of information gain, which can be used in place of Gini
 * impurity.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_INFORMATION_GAIN_HPP
#define MLPACK_METHODS_DECISION_TREE_INFORMATION_GAIN_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace tree {

class InformationGain
{
 public:
  /**
   * Given the sufficient statistics of a proposed split, calculate the
   * information gain if that split was to be used.  The 'counts' matrix should
   * contain the number of points in each class in each column, so the size of
   * 'counts' is children x classes, where 'children' is the number of child
   * nodes in the proposed split.
   *
   * @param counts Matrix of sufficient statistics.
   */
  static double Evaluate(const arma::Row<size_t>& labels,
                         const size_t numClasses)
  {
    // Edge case: if there are no elements, the gain is zero.
    if (labels.n_elem == 0)
      return 0.0;

    // Count the number of elements in each class.
    arma::Col<size_t> counts(numClasses);
    counts.zeros();
    for (size_t i = 0; i < labels.n_elem; ++i)
      counts[labels[i]]++;

    // Calculate the information gain.
    double gain = 0.0;
    for (size_t i = 0; i < numClasses; ++i)
    {
      const double f = ((double) counts[i] / (double) labels.n_elem);
      if (f > 0.0)
        gain += f * std::log2(f);
    }

    return gain;
  }

  /**
   * Return the range of the information gain for the given number of classes.
   * (That is, the difference between the maximum possible value and the minimum
   * possible value.)
   */
  static double Range(const size_t numClasses)
  {
    // The best possible case gives an information gain of 0.  The worst
    // possible case is even distribution, which gives n * (1/n * log2(1/n)) =
    // log2(1/n) = -log2(n).  So, the range is log2(n).
    return std::log2(numClasses);
  }
};

} // namespace tree
} // namespace mlpack

#endif
