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
  lossFunction.BinaryScanInitialize();

  double bestFoundGain = std::min(bestGain + minimumGainSplit, 0.0);
  bool improved = false;
  // Force a minimum leaf size of 1 (empty children don't make sense).
  const size_t minimum = std::max(minimumLeafSize, (size_t) 1);
}

} // namespace ensemble
} // namespace mlpack

#endif
