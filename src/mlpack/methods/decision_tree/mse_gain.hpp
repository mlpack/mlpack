/**
 * @file methods/decision_tree/mse_gain.hpp
 * @author Rishabh Garg
 *
 * The mean squared error gain class, which is a fitness funtion for
 * regression based decision trees.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_MSE_GAIN_HPP
#define MLPACK_METHODS_DECISION_TREE_MSE_GAIN_HPP

#include <mlpack/prereqs.hpp>
#include "utils.hpp"

namespace mlpack {
namespace tree {

/**
 * The MSE (Mean squared error) gain, is a measure of set purity based on the
 * variance of response values present in the node. This is same thing as
 * negation of variance of dependent variable in the node as we will try to
 * maximize this quantity to maximize gain (and thus reduce variance of a set).
 */
class MSEGain
{
 public:
  /**
   * Evaluate the mean squared error gain of labls from begin to end index.
   * Note that gain can be slightly greater than 0 due to floating-point
   * representation issues. Thus if you are checking for perfect fit, be
   * sure to use 'gain >= 0.0' and not 'gain == 0.0'. The labels vector should
   * always be of type arma::Row<double> or arma::rowvec.
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
    double mse = 0.0;

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
        mse += weights[i] * std::pow(labels[i] - weightedMean, 2);

      mse /= accWeights;
    }
    else
    {
      double mean = 0.0;
      Sum(labels, begin, end, mean);
      mean /= (double) (end - begin);

      for (size_t i = begin; i < end; ++i)
        mse += std::pow(labels[i] - mean, 2);

      mse /= (double) (end - begin);
    }

    return -mse;
  }

  /**
   * Evaluate the MSE gain on the complete vector.
   *
   * @param labels Set of labels to evaluate MAD gain on.
   * @param weights Weights associated to each label.
   */
  template<bool UseWeights, typename WeightVecType>
  static double Evaluate(const arma::rowvec& labels,
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
