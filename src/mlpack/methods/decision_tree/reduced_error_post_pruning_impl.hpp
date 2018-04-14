/**
 * @file reduced_error_post_pruning_impl.hpp
 * @author Manthan-R-Sheth
 *
 * This file is the implementation of a pruning mechanism for decision tree.
 * It is a regularization technique.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_REDUCED_ERROR_POST_PRUNING_IMPL_HPP
#define MLPACK_REDUCED_ERROR_POST_PRUNING_IMPL_HPP

#include "reduced_error_post_pruning.hpp"

namespace mlpack {
namespace tree {

template<typename DecisionTree>
template<bool UseWeights, typename MatType, typename LabelsType,
  typename WeightsType>
void ReducedErrorPostPruning<DecisionTree>::Prune(
    DecisionTree* root,
    DecisionTree* currentNode,
    LabelsType&& labels,
    const size_t numClasses,
    WeightsType&& weights,
    MatType&& validData,
    LabelsType&& validLabels,
    double& bestScore)
{
  size_t numOfChildren = currentNode->NumChildren();
  for (size_t i = 0; i < numOfChildren; ++i)
  {
    DecisionTree* node = &currentNode->Child(i);
    if (node->NumChildren() == 0)
    {
      for (size_t j = 0; j < numOfChildren; ++j)
      {
        DecisionTree* siblingNode = &currentNode->Child(j);
        if (siblingNode->NumChildren() != 0)
        {
          return;
        }
      }
      std::vector<DecisionTree*> childrenbacktrack = currentNode->getChildren();
      size_t dimensionTypeOrMajorityClassbacktrack =
        currentNode->getDimensionTypeOrMajorityClass();
      arma::vec classProbabilitiesbacktrack =
        currentNode->getClassProbabilities();

      arma::vec classProbabilitiesNew(numClasses);
      // Calculate class probabilities of present node because its children are
      // all leaves.
      for (size_t j = 0; j < numOfChildren; ++j)
      {
        for (size_t k = 0; k < numClasses; ++k)
          classProbabilitiesNew[k] +=
            childrenbacktrack[j]->getClassProbabilities()[k];
      }

      // Pruning the node's child and making the present node as leaf.
      currentNode->setChildren().clear();
      double newScore = ValidateScore(root, validData, validLabels);
      // Pruning the tree if the pruned tree gives better accuracy.
      if (newScore > bestScore)
      {
        bestScore = newScore;
        return;
      }
      else
      {
        // backtracking in case accuracy is not increased.
        currentNode->setChildren() = childrenbacktrack;
        currentNode->setDimensionTypeOrMajorityClass() =
          dimensionTypeOrMajorityClassbacktrack;
        currentNode->setClassProbabilities() = classProbabilitiesbacktrack;
      }
    }
    else
    {
      Prune<true>(root, node, labels, numClasses, weights, validData,
                        validLabels, bestScore);
    }
  }
}

template<typename DecisionTree>
template<typename MatType, typename LabelsType>
double ReducedErrorPostPruning<DecisionTree>::ValidateScore(DecisionTree* root,
                                              MatType&& validData,
                                              LabelsType&& validLabels)
{
  arma::Row<size_t> predictions;
  root->Classify(validData, predictions);
  double correctClassification = 0.0;
  for (size_t i = 0; i < validLabels.n_elem; ++i)
  {
    if (validLabels[i] == predictions[i])
      correctClassification++;
  }
  return (correctClassification/validLabels.n_elem);
}

} // namespace tree
} // namespace mlpack

#endif // MLPACK_REDUCED_ERROR_POST_PRUNING_IMPL_HPP
