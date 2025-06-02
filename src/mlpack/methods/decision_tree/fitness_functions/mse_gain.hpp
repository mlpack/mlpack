/**
 * @file methods/decision_tree/fitness_functions/mse_gain.hpp
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
#include <mlpack/methods/decision_tree/utils.hpp>

namespace mlpack {

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
  template<bool UseWeights, typename VecType, typename WeightVecType>
  static double Evaluate(const VecType& values,
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

      mse = accu(square(values.subvec(begin, end - 1) - mean));
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

  /**
   * Calculates the  mean squared error gain for the left and right children
   * for the current index.
   *
   * X = array of values of size n.
   *
   * @f{eqnarray*}{
   *   MSE = \sum\limits_{i=1}^n {X_i}^2 -
   *       {\dfrac{\sum\limits_{j=1}^n X_j}{n}}^2
   * @f}
   */
  std::tuple<double, double> BinaryGains()
  {
    double mseLeft = leftSumSquares / leftSize - leftMean * leftMean;
    double mseRight = (totalSumSquares - leftSumSquares) / rightSize
          - rightMean * rightMean;

    return std::make_tuple(-mseLeft, -mseRight);
  }

  /**
   * Caches the prefix sum of squares to efficiently compute gain value for
   * each split. It also computes the initial mean for left and right child.
   *
   * @param responses The set of responses on which statistics are computed.
   * @param weights The set of weights associated to each response.
   * @param minimum The minimum number of elements in a leaf.
   */
  template<bool UseWeights, typename ResponsesType, typename WeightVecType>
  void BinaryScanInitialize(const ResponsesType& responses,
                            const WeightVecType& weights,
                            const size_t minimum)
  {
    using RType = typename ResponsesType::elem_type;
    using WType = typename WeightVecType::elem_type;

    // Initializing data members to cache statistics.
    leftMean = 0.0;
    rightMean = 0.0;
    leftSize = 0.0;
    rightSize = 0.0;
    leftSumSquares = 0.0;
    totalSumSquares = 0.0;

    if (UseWeights)
    {
      totalSumSquares = accu(weights % square(responses));
      for (size_t i = 0; i < minimum - 1; ++i)
      {
        const WType w = weights[i];
        const RType x = responses[i];

        // Calculating initial weighted mean of responses for the left child.
        leftSize += w;
        leftMean += w * x;
        leftSumSquares += w * x * x;
      }
      if (leftSize > 1e-9)
        leftMean /= leftSize;

      for (size_t i = minimum - 1; i < responses.n_elem; ++i)
      {
        const WType w = weights[i];
        const RType x = responses[i];

        // Calculating initial weighted mean of responses for the right child.
        rightSize += w;
        rightMean += w * x;
      }
      if (rightSize > 1e-9)
        rightMean /= rightSize;
    }
    else
    {
      totalSumSquares = accu(square(responses));
      for (size_t i = 0; i < minimum - 1; ++i)
      {
        const RType x = responses[i];

        // Calculating the initial mean of responses for the left child.
        ++leftSize;
        leftMean += x;
        leftSumSquares += x * x;
      }
      if (leftSize > 1e-9)
        leftMean /= leftSize;

      for (size_t i = minimum - 1; i < responses.n_elem; ++i)
      {
        const RType x = responses[i];

        // Calculating the initial mean of responses for the right child.
        ++rightSize;
        rightMean += x;
      }
      if (rightSize > 1e-9)
        rightMean /= rightSize;
    }
  }

  /**
   * Updates the statistics for the given index.
   *
   * @param responses The set of responses on which statistics are computed.
   * @param weights The set of weights associated to each response.
   * @param index The current index.
   */
  template<bool UseWeights, typename ResponsesType, typename WeightVecType>
  void BinaryStep(const ResponsesType& responses,
                  const WeightVecType& weights,
                  const size_t index)
  {
    using RType = typename ResponsesType::elem_type;
    using WType = typename WeightVecType::elem_type;

    if (UseWeights)
    {
      const WType w = weights[index];
      const RType x = responses[index];

      // Update weighted sum of squares for left child.
      leftSumSquares += w * x * x;

      // Update weighted mean for both childs.
      leftMean = (leftMean * leftSize + w * x) / (leftSize + w);
      leftSize += w;

      rightMean = (rightMean * rightSize - w * x) / (rightSize - w);
      rightSize -= w;
    }
    else
    {
      const RType x = responses[index];

      // Update sum of squares for left child.
      leftSumSquares += x * x;

      // Update mean for both childs.
      leftMean = (leftMean * leftSize + x) / (leftSize + 1);
      ++leftSize;

      rightMean = (rightMean * rightSize - x) / (rightSize - 1);
      --rightSize;
    }
  }

 private:
  /**
   * The following data members cache statistics for weighted data when
   * `UseWeights` is true, else it will calculate unweighted statistics.
   */
  // Stores the sum of squares / weighted sum of squares for the left child.
  double leftSumSquares;
  // For unweighted data, stores the number of elements in each child.
  // For weighted data, stores the sum of weights of elements in each
  // child.
  double leftSize;
  double rightSize;
  // Stores the mean / weighted mean.
  double leftMean;
  double rightMean;
  // Stores the total sum of squares / total weighted sum of squares.
  double totalSumSquares;
};

} // namespace mlpack

#endif
