/**
 * @file methods/xgboost/xgb_exact_numeric_split.hpp
 * @author Rishabh Garg
 *
 * The exact numeric splitter for gradient boosted decision tree (GBDT) in
 * XGBoost.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_XGBOOST_XGB_EXACT_NUMERIC_SPLIT_HPP
#define MLPACK_METHODS_XGBOOST_XGB_EXACT_NUMERIC_SPLIT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ensemble {

/**
 * The XGBExactNumericSplit is a splitting function for GBDT that will
 * exhaustively search a numeric dimension for the best binary split.
 *
 * @tparam LossFunction Loss function to use to evaluate split.
 */
template<typename LossFunction>
class XGBExactNumericSplit
{
 public:
  // No extra info needed for split.
  class AuxiliarySplitInfo { };

 /**
   * Check if we can split a node.  If we can split a node in a way that
   * improves on 'bestGain', then we return the improved gain.  Otherwise we
   * return the value 'bestGain'.  If a split is made, then splitInfo and aux
   * may be modified.
   *
   * @param bestGain Best gain seen so far (we'll only split if we find gain
   *      better than this).
   * @param data The dimension of data points to check for a split in.
   * @param input This is a 2D matrix. The first row stores the true observed
   *    values and the second row stores the prediction at the current step
   *    of boosting.
   * @param weights Weights associated with responses.
   * @param minimumLeafSize Minimum number of points in a leaf node for
   *      splitting.
   * @param minimumGainSplit Minimum gain split.
   * @param splitInfo Stores split information on a successful split.
   * @param aux Auxiliary split information, which may be modified on a
   *      successful split.
   */
  template<bool UseWeights, typename VecType, typename MatType,
           typename WeightVecType>
  static double SplitIfBetter(
      const double bestGain,
      const VecType& data,
      const MatType& input,
      const WeightVecType& weights,
      const size_t minimumLeafSize,
      const double minimumGainSplit,
      double& splitInfo,
      AuxiliarySplitInfo& aux,
      LossFunction lossFunction);

  /**
   * Returns 2, since the binary split always has two children.
   */
  static size_t NumChildren(const double& /* splitInfo */,
                            const AuxiliarySplitInfo& /* aux */)
  {
    return 2;
  }

  /**
   * Given a point, calculate which child it should go to (left or right).
   *
   * @param point Point to calculate direction of.
   * @param splitInfo Auxiliary information for the split.
   * @param * (aux) Auxiliary information for the split (Unused).
   */
  template<typename ElemType>
  static size_t CalculateDirection(
      const ElemType& point,
      const double& splitInfo,
      const AuxiliarySplitInfo& /* aux */);
};

} // namespace ensemble
} // namespace mlpack

// Include implementation.
#include "xgb_exact_numeric_split_impl.hpp"

#endif
