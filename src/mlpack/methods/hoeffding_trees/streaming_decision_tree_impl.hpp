/**
 * @file streaming_decision_tree_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of a streaming decision tree.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_STREAMING_DECISION_TREE_IMPL_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_STREAMING_DECISION_TREE_IMPL_HPP

// In case it hasn't been included yet.
#include "streaming_decision_tree.hpp"

namespace mlpack {
namespace tree {

template<typename SplitType, typename MatType>
StreamingDecisionTree<SplitType, MatType>::StreamingDecisionTree(
    const MatType& data,
    const data::DatasetInfo& datasetInfo,
    const arma::Row<size_t>& labels,
    const size_t numClasses) :
    split(data.n_rows, numClasses, datasetInfo, 0.95)
{
  Train(data, labels);
}

template<typename SplitType, typename MatType>
StreamingDecisionTree<SplitType, MatType>::StreamingDecisionTree(
    const data::DatasetInfo& datasetInfo,
    const size_t dimensionality,
    const size_t numClasses) :
    split(dimensionality, numClasses, datasetInfo, 0.95)
{
  // No training.  Anything else to do...?
}

template<typename SplitType, typename MatType>
StreamingDecisionTree<SplitType, MatType>::StreamingDecisionTree(
    const StreamingDecisionTree& other) :
    split(other.split)
{
  // Copy the children of the other tree.
}

template<typename SplitType, typename MatType>
template<typename VecType>
void StreamingDecisionTree<SplitType, MatType>::Train(const VecType& data,
                                                      const size_t label)
{
  if (children.size() == 0)
  {
    split.Train(data, label);

    const size_t numChildren = split.SplitCheck();
    if (numChildren > 0)
    {
      // We need to add a bunch of children.
      // Delete children, if we have them.
      if (children.size() > 0)
        children.clear();

      // The split knows how to add the children.
      split.CreateChildren(children);
    }
  }
  else
  {
    // We've already split this node.  But we need to train the child nodes.
    size_t direction = split.CalculateDirection(data);
    children[direction].Train(data, label);
  }
}

template<typename SplitType, typename MatType>
void StreamingDecisionTree<SplitType, MatType>::Train(
    const MatType& data,
    const arma::Row<size_t>& labels)
{
  // Train on each point sequentially.
  for (size_t i = 0; i < data.n_cols; ++i)
    Train(data.col(i), labels[i]);
}

template<typename SplitType, typename MatType>
template<typename VecType>
size_t StreamingDecisionTree<SplitType, MatType>::Classify(const VecType& data)
{
  // Get the direction we need to go, and continue classification.
  // If we're at a leaf, we don't need to go any deeper.
  if (children.size() == 0)
  {
    return split.Classify(data);
  }
  else
  {
    const size_t direction = split.CalculateDirection(data);
    return children[direction].Classify(data);
  }
}

template<typename SplitType, typename MatType>
void StreamingDecisionTree<SplitType, MatType>::Classify(
    const MatType& data,
    arma::Row<size_t>& predictions)
{
  for (size_t i = 0; i < data.n_cols; ++i)
    predictions[i] = Classify(data.col(i));
}

} // namespace tree
} // namespace mlpack

#endif
