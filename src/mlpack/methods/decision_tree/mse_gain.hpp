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
   * Evaluate the mean squared error gain of values from begin to end index.
   * Note that gain can be slightly greater than 0 due to floating-point
   * representation issues. Thus if you are checking for perfect fit, be
   * sure to use 'gain >= 0.0' and not 'gain == 0.0'. The values vector
   * should always be of type arma::Row<double> or arma::rowvec.
   *
   * @param values Set of values to evaluate MAD gain on.
   * @param weights Weights associated to each value.
   * @param begin Start index.
   * @param end End index.
   */
  template<bool UseWeights, typename WeightVecType>
  static double Evaluate(const arma::rowvec& values,
                         const WeightVecType& weights,
                         const size_t begin,
                         const size_t end)
  {
    double mse = 0.0;

    if (UseWeights)
    {
      double accWeights = 0.0;
      double weightedMean = 0.0;
      WeightedSum(values, weights, begin, end, accWeights, weightedMean);

      // Catch edge case: if there are no weights, the impurity is zero.
      if (accWeights == 0.0)
        return 0.0;

      weightedMean /= accWeights;

      for (size_t i = begin; i < end; ++i)
        mse += weights[i] * std::pow(values[i] - weightedMean, 2);

      mse /= accWeights;
    }
    else
    {
      double mean = 0.0;
      Sum(values, begin, end, mean);
      mean /= (double) (end - begin);

      mse = arma::accu(arma::square(values.subvec(begin, end - 1) - mean));
      mse /= (double) (end - begin);
    }

    return -mse;
  }

  /**
   * Evaluate the MSE gain on the complete vector.
   *
   * @param values Set of values to evaluate MSE gain on.
   * @param weights Weights associated to each value.
   */
  template<bool UseWeights, typename WeightVecType>
  static double Evaluate(const arma::rowvec& values,
                         const size_t /* numClasses */,
                         const WeightVecType& weights)
  {
    // Corner case: if there are no elements, the impurity is zero.
    if (values.n_elem == 0)
      return 0.0;

    return Evaluate<UseWeights>(values, weights, 0, values.n_elem);
  }

  /**
   * Calculates the weighted mean squared error gain given the sum of squares
   * and mean.
   *
   * X = array of values of size n.
   * W = array of weights of size n.
   *
   * @f{eqnarray*}{
   *    MSE = \sum\limits_{i=1}^n {W_i * {X_i}^2} -
   *        {\dfrac{\sum\limits_{j=1}^n W_j * X_j}
   *        {\sum\limits_{j=1}^n W_i}}^2
   * @f}
   *
   * @param weightedSumSquares Precomputed weighted sum of square
   *    (sum(Wi * Xi^2)) of values.
   * @param weightedMean Precomputed weighted mean (sum(Wi * Xi) / sum(Wi)) of
   *    values.
   * @param totalChildWeight Total weight of all the samples in that child.
   */
  static double Evaluate(const double weightedSumSquares,
                         const double weightedMean,
                         const double totalChildWeight)
  {
    double mse = weightedSumSquares / totalChildWeight -
        weightedMean * weightedMean;
    return -mse;
  }

  /**
   * Calculates the  mean squared error gain given the sum of squares and mean.
   *
   * X = array of values of size n.
   *
   * @f{eqnarray*}{
   *   MSE = \sum\limits_{i=1}^n {X_i}^2 -
   *       {\dfrac{\sum\limits_{j=1}^n X_j}{n}}^2
   * @f}
   *
   * @param sumSquares Precomputed sum of square (sum(Xi^2)) of values.
   * @param mean Precomputed mean (sum(Xi) / n) of values.
   * @param childSize The total number of samples in that child.
   */
  static double Evaluate(const double sumSquares,
                         const double mean,
                         const size_t childSize)
  {
    double mse = sumSquares / (double) childSize - mean * mean;
    return -mse;
  }
};

} // namespace tree
} // namespace mlpack

#endif
