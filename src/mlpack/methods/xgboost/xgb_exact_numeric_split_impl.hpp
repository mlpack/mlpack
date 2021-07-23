/**
 * @file methods/xgboost/xgb_exact_numeric_split_impl.hpp
 * @author Rishabh Garg
 *
 * Implementation of exact numeric splitter for XGBoost.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_XGBOOST_XGB_EXACT_NUMERIC_SPLIT_IMPL_HPP
#define MLPACK_METHODS_XGBOOST_XGB_EXACT_NUMERIC_SPLIT_IMPL_HPP

namespace mlpack {
namespace ensemble {

template<typename LossFunction>
template<bool UseWeights, typename VecType, typename MatType,
          typename WeightVecType>
double XGBExactNumericSplit<LossFunction>::SplitIfBetter(
    const double bestGain,
    const VecType& data,
    const MatType& input,
    const WeightVecType& weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    double& splitInfo,
    AuxiliarySplitInfo& aux,
    LossFunction lossFunction)
{
  // It can't be outperformed.
  if (bestGain == 0.0)
    return DBL_MAX;

  // TODO: Filter out and separate the missing data before sorting to decrease
  // sensitivity towards missing data.

  // Next, sort the data.
  arma::uvec sortedIndices = arma::sort_index(data);
  ResponsesType sortedResponses(responses.n_elem);
  for (size_t i = 0; i < sortedResponses.n_elem; ++i)
    sortedResponses[i] = responses[sortedIndices[i]];
  lossFunction.sortGradAndHess(sortedIndices);

  // Sanity check: if the first element is the same as the last, we can't split
  // in this dimension.
  if (data[sortedIndices[0]] == data[sortedIndices[sortedIndices.n_elem - 1]])
    return DBL_MAX;

  // Initialize the binary scan.
  // In XGBoost, minimum number of points in a node is determined by a parameter
  // called min_child_weight. This parameter will be stored within the loss function
  // class and thus the begin index will be calculated inside BinaryScanInitialize().
  size_t index = 0;
  lossFunction.BinaryScanInitialize(index);

  bool improved = false;
  bool endLoop = false;
  for (; index < data.n_elem - 1; ++index)
  {
    lossFunction.BinaryStep(index, endLoop);

    // We have to ensure that the min_child_weight condition is held in the right
    // child too. So, if at any index, that condition will be false, then we will
    // end the loop.
    if (endLoop) break;

    // Make sure that the value has changed.
    if (data[sortedIndices[index]] == data[sortedIndices[index - 1]])
      continue;

    // Calculate the gain for the left and right child.
    std::tuple<double, double> binaryGains = fitnessFunction.BinaryGains();
    const double leftGain = std::get<0>(binaryGains);
    const double rightGain = std::get<1>(binaryGains);

    // Corner case: is this the best possible split?
    if (gain >= 0.0)
    {
      // We can take a shortcut: no split will be better than this, so just
      // take this one. The actual split value will be halfway between the
      // value at index - 1 and index.
      splitInfo = (data[sortedIndices[index - 1]] +
          data[sortedIndices[index]]) / 2.0;

      return gain;
    }
    if (gain > bestFoundGain)
    {
      // We still have a better split.
      bestFoundGain = gain;
      splitInfo = (data[sortedIndices[index - 1]] +
          data[sortedIndices[index]]) / 2.0;
      improved = true;
    }
  }
  // If we didn't improve, return the original gain exactly as we got it
  // (without introducing floating point errors).
  if (!improved)
    return DBL_MAX;

  return bestFoundGain;
}

} // namespace ensemble
} // namespace mlpack

#endif
