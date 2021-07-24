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
    LossFunction& lossFunction)
{
  // It can't be outperformed.
  if (bestGain == 0.0)
    return DBL_MAX;

  // TODO: Filter out and separate the missing data before sorting to decrease
  // sensitivity towards missing data.

  // Next, sort the data.
  arma::uvec sortedIndices = arma::sort_index(data);
  arma::vec sortedGradients(sortedIndices.n_elem);
  arma::vec sortedHessians(sortedIndices.n_elem);
  lossFunction.sortGradAndHess(sortedIndices, sortedGradients,
      sortedHessians);

  // Sanity check: if the first element is the same as the last, we can't split
  // in this dimension.
  if (data[sortedIndices[0]] == data[sortedIndices[sortedIndices.n_elem - 1]])
    return DBL_MAX;

  // Initialize the binary scan. It initialises the index such that the weight
  // of the left child is higher than the minChildWeight.
  size_t index = lossFunction.BinaryScanInitialize(sortedGradients,
      sortedHessians);

  bool improved = false;
  bool endLoop = false;
  for (; index < data.n_elem - 1; ++index)
  {
    lossFunction.BinaryStep(sortedGradients, sortedHessians, index, endLoop);

    // We have to ensure that the minChildWeight condition is held in the right
    // child too. So, if at any index, that condition doesn't hold true, then
    // terminate the loop.
    if (endLoop) break;

    // Make sure that the value has changed.
    if (data[sortedIndices[index]] == data[sortedIndices[index - 1]])
      continue;

    // Evaluate the gain for the split.
    const double gain = lossFunction.Evaluate();

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
