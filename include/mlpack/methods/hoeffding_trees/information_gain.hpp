/**
 * @file methods/hoeffding_trees/information_gain.hpp
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
#ifndef MLPACK_METHODS_HOEFFDING_TREES_INFORMATION_GAIN_HPP
#define MLPACK_METHODS_HOEFFDING_TREES_INFORMATION_GAIN_HPP

namespace mlpack {

class HoeffdingInformationGain
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
  static double Evaluate(const arma::Mat<size_t>& counts)
  {
    // Calculate the number of elements in the unsplit node and also in each
    // proposed child.
    size_t numElem = 0;
    arma::vec splitCounts(counts.n_elem);
    for (size_t i = 0; i < counts.n_cols; ++i)
    {
      splitCounts[i] = accu(counts.col(i));
      numElem += splitCounts[i];
    }

    // Corner case: if there are no elements, the gain is zero.
    if (numElem == 0)
      return 0.0;

    arma::Col<size_t> classCounts = sum(counts, 1);

    // Calculate the gain of the unsplit node.
    double gain = 0.0;
    for (size_t i = 0; i < classCounts.n_elem; ++i)
    {
      const double f = ((double) classCounts[i] / (double) numElem);
      if (f > 0.0)
        gain -= f * std::log2(f);
    }

    // Now calculate the impurity of the split nodes and subtract them from the
    // overall gain.
    for (size_t i = 0; i < counts.n_cols; ++i)
    {
      if (splitCounts[i] > 0)
      {
        double splitGain = 0.0;
        for (size_t j = 0; j < counts.n_rows; ++j)
        {
          const double f = ((double) counts(j, i) / (double) splitCounts[i]);
          if (f > 0.0)
            splitGain += f * std::log2(f);
        }

        gain += ((double) splitCounts[i] / (double) numElem) * splitGain;
      }
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

} // namespace mlpack

#endif
