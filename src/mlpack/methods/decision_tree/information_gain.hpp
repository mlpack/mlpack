/**
 * @file information_gain.hpp
 * @author Ryan Curtin
 *
 * An implementation of information gain, which can be used in place of Gini
 * gain.
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

/**
 * The standard information gain criterion, used for calculating gain in
 * decision trees.
 */
class InformationGain
{
 public:
  /**
   * Given a set of labels, calculate the information gain of those labels.
   *
   * @param labels Labels of the dataset.
   * @param numClasses Number of classes in the dataset.
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
   *
   * @param numClasses Number of classes in the dataset.
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
