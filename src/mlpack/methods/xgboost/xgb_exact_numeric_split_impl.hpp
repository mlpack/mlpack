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
  
}

} // namespace ensemble
} // namespace mlpack

#endif
