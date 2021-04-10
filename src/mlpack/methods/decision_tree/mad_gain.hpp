/**
 * @file methods/decision_tree/mad_gain.hpp
 * @author Rishabh Garg
 *
 * The mean absolute deviation gain class, a fitness funtion for regression
 * based decision trees.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more informatio
n.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_MAD_GAIN_HPP
#define MLPACK_METHODS_DECISION_TREE_MAD_GAIN_HPP

#include <mlpack/prereqs.hpp>
#include "utils.hpp"

namespace mlpack {
namespace tree {

/**
 * The MAD (Mean absolute deviation) gain, is a measure of set purity based on
 * the deviation of dependent values present in the node. This is same thing as
 * negation of deviation of dependent variable from the mean in the node as we
 * will try to maximize this quantity to maximize gain (and thus reduce
 * absolute deviation of a set).
*/
class MADGain
{
 public:
  /**
   * Evaluate the mean absolute deviation gain from begin to end index. Note
   * that gain can be slightly greater than 0 due to floating-point
   * representation issues. Thus if you are checking for perfect fit, be sure
   * to use 'gain >= 0.0'. Not 'gain == 0.0'. The labels should always be of
   * type arma::Row<double> or arma::rowvec.
   *
   * @param labels Set of labels to evaluate MAD gain on.
   * @param weights Weight of labels.
   * @param begin Start index.
   * @param end End index.
   */
  template<bool UseWeights, typename WeightVecType>
  static double Evaluate(const arma::rowvec& labels,
                         const WeightVecType& weights,
                         const size_t begin,
                         const size_t end)
  {
    double mad = 0.0;

    if (UseWeights)
    {
      double accWeights = 0.0;
      double weightedMean = 0.0;

      WeightedSum(labels, weights, begin, end, accWeights, weightedMean);

      // Catch edge case: if there are no weights, the impurity is zero.
      if (accWeights == 0.0)
        return 0.0;

      weightedMean /= accWeights;

      for (size_t i = begin; i < end; ++i)
      {
        mad += weights[i] * (std::abs(labels[i] - weightedMean));
      }
      mad /= accWeights;
      }
    else
    {
      double mean = 0.0;
      Sum(labels, begin, end, mean);
      mean /= (double) (end - begin);

      for (size_t i = begin; i < end; ++i)
        mad += std::abs(labels[i] - mean);

      mad /= (double) (end - begin);
    }

    return -mad;
  }

  /**
   * Evaluate the MAD gain on the complete vector.
   *
   * @param labels Set of labels to evaluate MAD gain on.
   * @param weights Weights associated to each label.
   */
  template<bool UseWeights, typename WeightVecType>
  static double Evaluate(const arma::rowvec& labels,
                         const size_t /* numClasses */,
                         const WeightVecType& weights)
  {
    // Corner case: if there are no elements, the impurity is zero.
    if (labels.n_elem == 0)
      return 0.0;

    return Evaluate<UseWeights>(labels, weights, 0, labels.n_elem);
  }

};

} // namespace tree
} // namespace mlpack

#endif
