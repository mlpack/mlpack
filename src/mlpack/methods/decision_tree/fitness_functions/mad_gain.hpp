/**
 * @file methods/decision_tree/fitness_functions/mad_gain.hpp
 * @author Rishabh Garg
 *
 * The mean absolute deviation gain class, a fitness function for regression
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
#include "mlpack/methods/decision_tree/utils.hpp"

namespace mlpack {

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
   * to use 'gain >= 0.0'. Not 'gain == 0.0'. The values should always be of
   * type arma::Row<double> or arma::rowvec.
   *
   * @param values Set of values to evaluate MAD gain on.
   * @param weights Weights associated to each value.
   * @param begin Start index.
   * @param end End index.
   */
  template<bool UseWeights, typename VecType, typename WeightVecType>
  static double Evaluate(const VecType& values,
                         const WeightVecType& weights,
                         const size_t begin,
                         const size_t end)
  {
    double mad = 0.0;

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
      {
        mad += weights[i] * (std::abs(values[i] - weightedMean));
      }
      mad /= accWeights;
      }
    else
    {
      double mean = 0.0;
      Sum(values, begin, end, mean);
      mean /= (double) (end - begin);

      mad = accu(arma::abs(values.subvec(begin, end - 1) - mean));
      mad /= (double) (end - begin);
    }

    return -mad;
  }

  /**
   * Evaluate the MAD gain on the complete vector.
   *
   * @param values Set of values to evaluate MAD gain on.
   * @param weights Weights associated to each value.
   */
  template<bool UseWeights, typename VecType, typename WeightVecType>
  static double Evaluate(const VecType& values,
                         const WeightVecType& weights)
  {
    // Corner case: if there are no elements, the impurity is zero.
    if (values.n_elem == 0)
      return 0.0;

    return Evaluate<UseWeights>(values, weights, 0, values.n_elem);
  }

  /**
   * Returns the output value for each leaf node for prediction. The output
   * value is calculated as the average of all the points in that leaf node.
   * This calculation is specific to regression trees only.
   */
  template<bool UseWeights, typename ResponsesType, typename WeightsType>
  double OutputLeafValue(const ResponsesType& responses,
                         const WeightsType& weights)
  {
    if (UseWeights)
    {
      double accWeights, weightedSum;
      WeightedSum(responses, weights, 0, responses.n_elem, accWeights,
          weightedSum);
      return weightedSum / accWeights;
    }
    else
    {
      double sum;
      Sum(responses, 0, responses.n_elem, sum);
      return sum / responses.n_elem;
    }
  }
};

} // namespace mlpack

#endif
