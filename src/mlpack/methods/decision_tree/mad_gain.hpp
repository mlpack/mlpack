/**
 * @file methods/decision_tree/mse_gain.hpp
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
      double accWeights[4] = { 0.0, 0.0, 0.0, 0.0 };
      double weightedMean[4] = { 0.0, 0.0, 0.0, 0.0 };

      // SIMD loop: sums four elements simultaneously (if the compiler manages
      // to vectorize the loop).
      for (size_t i = begin + 3; i < end; i += 4)
      {
        const double weight1 = weights[i - 3];
        const double weight2 = weights[i - 2];
        const double weight3 = weights[i - 1];
        const double weight4 = weights[i];

        weightedMean[0] += weight1 * labels[i - 3];
        weightedMean[1] += weight2 * labels[i - 2];
        weightedMean[2] += weight3 * labels[i - 1];
        weightedMean[3] += weight4 * labels[i];

        accWeights[0] += weight1;
        accWeights[1] += weight2;
        accWeights[2] += weight3;
        accWeights[3] += weight4;
      }

      // Handle leftovers.
      if ((end - begin) % 4 == 1)
      {
        const double weight1 = weights[end - 1];
        weightedMean[0] += weight1 * labels[end - 1];
        accWeights[0] += weight1;
      }
      else if ((end - begin) % 4 == 2)
      {
        const double weight1 = weights[end - 2];
        const double weight2 = weights[end - 1];

        weightedMean[0] += weight1 * labels[end - 2];
        weightedMean[1] += weight2 * labels[end - 1];

        accWeights[0] += weight1;
        accWeights[1] += weight2;
      }
      else if ((end - begin) % 4 == 3)
      {
        const double weight1 = weights[end - 3];
        const double weight2 = weights[end - 2];
        const double weight3 = weights[end - 1];

        weightedMean[0] += weight1 * labels[end - 3];
        weightedMean[1] += weight2 * labels[end - 2];
        weightedMean[2] += weight1 * labels[end - 1];

        accWeights[0] += weight1;
        accWeights[1] += weight2;
        accWeights[2] += weight3;
      }

      accWeights[0] += accWeights[1] + accWeights[2] + accWeights[3];
      weightedMean[0] += weightedMean[1] + weightedMean[2] + weightedMean[3];

      // Catch edge case: if there are no weights, the impurity is zero.
      if (accWeights[0] == 0.0)
        return 0.0;

      for (size_t i = begin; i < end; ++i)
      {
        const double f = weights[i] * (std::abs(labels[i] - weightedMean[0]));
        mad += f / accWeights[0];
      }
    }
    else
    {
      double mean[4] = { 0.0, 0.0, 0.0, 0.0 };

      // SIMD loop: add counts for four elements simultaneously (if the compiler
      // manages to vectorize the loop).
      for (size_t i = begin + 3; i < end; i += 4)
      {
        mean[0] += labels[i - 3];
        mean[1] += labels[i - 2];
        mean[2] += labels[i - 1];
        mean[3] += labels[i];
      }

      // Handle leftovers.
      if (labels.n_elem % 4 == 1)
      {
        mean[0] += labels[end - 1];
      }
      else if (labels.n_elem % 4 == 2)
      {
        mean[0] += labels[end - 2];
        mean[1] += labels[end - 1];
      }
      else if (labels.n_elem % 4 == 3)
      {
        mean[0] += labels[end - 3];
        mean[1] += labels[end - 2];
        mean[2] += labels[end - 1];
      }

      mean[0] += mean[1] + mean[2] + mean[3];

      for (size_t i = begin; i < end; ++i)
        mad += std::abs(labels[i] - mean[0]);

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
