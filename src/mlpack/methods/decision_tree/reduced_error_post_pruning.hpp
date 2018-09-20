/**
 * @file reduced_error_post_pruning.hpp
 * @author Manthan-R-Sheth
 *
 * This file defines a pruning mechanism for decision tree.
 * It is a regularization technique.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_REDUCED_ERROR_POST_PRUNING_HPP
#define MLPACK_REDUCED_ERROR_POST_PRUNING_HPP

#include <mlpack/prereqs.hpp>
#include "decision_tree.hpp"

namespace mlpack {
namespace tree {

/**
 * The ReducedErrorPostPruning is a pruning mechanism for post tree pruning for
 * improving generalisation.
 *
 * @tparam DecisionTree Decision Tree for pruning upon.
 */
template<typename DecisionTree>
class ReducedErrorPostPruning
{
 public:
  /**
   * Prune the already built decision tree by making tree nodes as leaf nodes
   * if the score on the validation set increases.
   *
   * @param root pointer to the root node of the tree for score calculation.
   * @param labels Labels for each training point.
   * @param numClasses Number of classes in the dataset.
   * @param weights Weights of all the labels
   * @param validData Validation dataset.
   * @param validLabels labels for validation points.
   * @param bestScore best score of pruning on validation set.
   */
  template<bool UseWeights, typename MatType, typename LabelsType,
    typename WeightsType>
  static void Prune(DecisionTree* root,
                    DecisionTree* currentNode,
                    LabelsType&& labels,
                    const size_t numClasses,
                    WeightsType&& weights,
                    MatType&& validData,
                    LabelsType&& validLabels,
                    double& bestScore);

  /**
   * Utility function for validating the score on the tree whose root is given.
   *
   * @param root pointer to the root node of the tree for score calculation.
   * @param validData Validation dataset.
   * @param validLabels labels for validation points.
   */
  template<typename MatType, typename LabelsType>
  static double ValidateScore(DecisionTree* root,
                       MatType&& validData,
                       LabelsType&& validLabels);
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "reduced_error_post_pruning_impl.hpp"

#endif // MLPACK_REDUCED_ERROR_POST_PRUNING_HPP
