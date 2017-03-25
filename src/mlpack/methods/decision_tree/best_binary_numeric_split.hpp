/**
 * @file best_binary_numeric_split.hpp
 * @author Ryan Curtin
 *
 * A tree splitter that finds the best binary numeric split.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_BEST_BINARY_NUMERIC_SPLIT_HPP
#define MLPACK_METHODS_DECISION_TREE_BEST_BINARY_NUMERIC_SPLIT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace tree {

/**
 * The BestBinaryNumericSplit is a splitting function for decision trees that
 * will exhaustively search a numeric dimension for the best binary split.
 *
 * @tparam FitnessFunction Fitness function to use to calculate gain.
 */
template<typename FitnessFunction>
class BestBinaryNumericSplit
{
 public:
  // No extra info needed for split.
  template<typename ElemType>
  class AuxiliarySplitInfo { };

  /**
   * Check if we can split a node.  If we can split a node in a way that
   * improves on 'bestGain', then we return the improved gain.  Otherwise we
   * return the value 'bestGain'.  If a split is made, then classProbabilities
   * and aux may be modified.
   *
   * @param bestGain Best gain seen so far (we'll only split if we find gain
   *      better than this).
   * @param data The dimension of data points to check for a split in.
   * @param numCategories Number of categories in the categorical data.
   * @param labels Labels for each point.
   * @param numClasses Number of classes in the dataset.
   * @param minimumLeafSize Minimum number of points in a leaf node for
   *      splitting.
   * @param classProbabilities Class probabilities vector, which may be filled
   *      with split information a successful split.
   * @param aux Auxiliary split information, which may be modified on a
   *      successful split.
   */
  template<typename VecType>
  static double SplitIfBetter(
      const double bestGain,
      const VecType& data,
      const arma::Row<size_t>& labels,
      const size_t numClasses,
      const size_t minimumLeafSize,
      arma::Col<typename VecType::elem_type>& classProbabilities,
      AuxiliarySplitInfo<typename VecType::elem_type>& aux);

  /**
   * Returns 2, since the binary split always has two children.
   */
  template<typename ElemType>
  static size_t NumChildren(const arma::Col<ElemType>& /* classProbabilities */,
                            const AuxiliarySplitInfo<ElemType>& /* aux */)
  {
    return 2;
  }

  /**
   * Given a point, calculate which child it should go to (left or right).
   *
   * @param point Point to calculate direction of.
   * @param classProbabilities Auxiliary information for the split.
   * @param aux (Unused) auxiliary information for the split.
   */
  template<typename ElemType>
  static size_t CalculateDirection(
      const ElemType& point,
      const arma::Col<ElemType>& classProbabilities,
      const AuxiliarySplitInfo<ElemType>& /* aux */);
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "best_binary_numeric_split_impl.hpp"

#endif
