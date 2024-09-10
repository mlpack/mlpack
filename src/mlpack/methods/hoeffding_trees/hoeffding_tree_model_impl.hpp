/**
 * @file methods/hoeffding_trees/hoeffding_tree_model_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the HoeffdingTreeModel class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HOEFFDING_TREE_HOEFFDING_TREE_MODEL_IMPL_HPP
#define MLPACK_METHODS_HOEFFDING_TREE_HOEFFDING_TREE_MODEL_IMPL_HPP

#include "hoeffding_tree_model.hpp"

namespace mlpack {

// Constructor.
inline HoeffdingTreeModel::HoeffdingTreeModel(const TreeType& type) :
    type(type),
    giniHoeffdingTree(NULL),
    giniBinaryTree(NULL),
    infoHoeffdingTree(NULL),
    infoBinaryTree(NULL)
{
  // Nothing to do.
}

// Copy constructor.
inline HoeffdingTreeModel::HoeffdingTreeModel(const HoeffdingTreeModel& other) :
    type(other.type),
    giniHoeffdingTree(other.giniHoeffdingTree ? new GiniHoeffdingTreeType(
        *other.giniHoeffdingTree) : NULL),
    giniBinaryTree(other.giniBinaryTree ? new GiniBinaryTreeType(
        *other.giniBinaryTree) : NULL),
    infoHoeffdingTree(other.infoHoeffdingTree ? new InfoHoeffdingTreeType(
        *other.infoHoeffdingTree) : NULL),
    infoBinaryTree(other.infoBinaryTree ? new InfoBinaryTreeType(
        *other.infoBinaryTree) : NULL)
{
  // Nothing else to do.
}

// Move constructor.
inline HoeffdingTreeModel::HoeffdingTreeModel(HoeffdingTreeModel&& other) :
    type(other.type),
    giniHoeffdingTree(other.giniHoeffdingTree),
    giniBinaryTree(other.giniBinaryTree),
    infoHoeffdingTree(other.infoHoeffdingTree),
    infoBinaryTree(other.infoBinaryTree)
{
  // Reset other model.
  other.type = GINI_HOEFFDING;
  other.giniHoeffdingTree = NULL;
  other.giniBinaryTree = NULL;
  other.infoHoeffdingTree = NULL;
  other.infoBinaryTree = NULL;
}

// Copy operator.
inline HoeffdingTreeModel& HoeffdingTreeModel::operator=(
    const HoeffdingTreeModel& other)
{
  if (this != &other)
  {
    // Clear this model.
    delete giniHoeffdingTree;
    delete giniBinaryTree;
    delete infoHoeffdingTree;
    delete infoBinaryTree;

    giniHoeffdingTree = NULL;
    giniBinaryTree = NULL;
    infoHoeffdingTree = NULL;
    infoBinaryTree = NULL;

    // Create the right tree.
    type = other.type;
    if (other.giniHoeffdingTree && (type == GINI_HOEFFDING))
      giniHoeffdingTree = new GiniHoeffdingTreeType(*other.giniHoeffdingTree);
    else if (other.giniBinaryTree && (type == GINI_BINARY))
      giniBinaryTree = new GiniBinaryTreeType(*other.giniBinaryTree);
    else if (other.infoHoeffdingTree && (type == INFO_HOEFFDING))
      infoHoeffdingTree = new InfoHoeffdingTreeType(*other.infoHoeffdingTree);
    else if (other.infoBinaryTree && (type == INFO_BINARY))
      infoBinaryTree = new InfoBinaryTreeType(*other.infoBinaryTree);
  }
  return *this;
}

// Move operator.
inline HoeffdingTreeModel& HoeffdingTreeModel::operator=(
    HoeffdingTreeModel&& other)
{
  if (this != &other)
  {
    // Clear this model.
    delete giniHoeffdingTree;
    delete giniBinaryTree;
    delete infoHoeffdingTree;
    delete infoBinaryTree;

    type = other.type;
    giniHoeffdingTree = other.giniHoeffdingTree;
    giniBinaryTree = other.giniBinaryTree;
    infoHoeffdingTree = other.infoHoeffdingTree;
    infoBinaryTree = other.infoBinaryTree;

    // Clear the other model.
    other.type = GINI_HOEFFDING;
    other.giniHoeffdingTree = NULL;
    other.giniBinaryTree = NULL;
    other.infoHoeffdingTree = NULL;
    other.infoBinaryTree = NULL;
  }
  return *this;
}

// Destructor.
inline HoeffdingTreeModel::~HoeffdingTreeModel()
{
  delete giniHoeffdingTree;
  delete giniBinaryTree;
  delete infoHoeffdingTree;
  delete infoBinaryTree;
}

// Create the model.
inline void HoeffdingTreeModel::BuildModel(
    const arma::mat& dataset,
    const data::DatasetInfo& datasetInfo,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const bool batchTraining,
    const double successProbability,
    const size_t maxSamples,
    const size_t checkInterval,
    const size_t minSamples,
    const size_t bins,
    const size_t observationsBeforeBinning)
{
  // Clean memory, if needed.
  delete giniHoeffdingTree;
  delete giniBinaryTree;
  delete infoHoeffdingTree;
  delete infoBinaryTree;

  // Depending on the type, create the tree.
  switch (type)
  {
    case GINI_HOEFFDING:
      // Create instantiated numeric split.
      {
        HoeffdingDoubleNumericSplit<GiniImpurity> ns(0, bins,
            observationsBeforeBinning);

        giniHoeffdingTree = new GiniHoeffdingTreeType(dataset, datasetInfo,
            labels, numClasses, batchTraining, successProbability, maxSamples,
            checkInterval, minSamples,
            HoeffdingCategoricalSplit<GiniImpurity>(0, 0), ns);
      }
      break;

    case GINI_BINARY:
      giniBinaryTree = new GiniBinaryTreeType(dataset, datasetInfo, labels,
          numClasses, batchTraining, successProbability, maxSamples,
          checkInterval, minSamples);
      break;

    case INFO_HOEFFDING:
      // Create instantiated numeric split.
      {
        HoeffdingDoubleNumericSplit<HoeffdingInformationGain> ns(0, bins,
            observationsBeforeBinning);

        infoHoeffdingTree = new InfoHoeffdingTreeType(dataset, datasetInfo,
            labels, numClasses, batchTraining, successProbability, maxSamples,
            checkInterval, minSamples,
            HoeffdingCategoricalSplit<HoeffdingInformationGain>(0, 0), ns);
      }
      break;

    case INFO_BINARY:
      infoBinaryTree = new InfoBinaryTreeType(dataset, datasetInfo, labels,
          numClasses, batchTraining, successProbability, maxSamples,
          checkInterval, minSamples);
      break;
  }
}

// Train the model on one pass of the dataset.
inline void HoeffdingTreeModel::Train(const arma::mat& dataset,
                                      const arma::Row<size_t>& labels,
                                      const bool batchTraining)
{
  // Depending on the type, pass through once.
  switch (type)
  {
    case GINI_HOEFFDING:
      giniHoeffdingTree->Train(dataset, labels, giniHoeffdingTree->NumClasses(),
          batchTraining);
      break;

    case GINI_BINARY:
      giniBinaryTree->Train(dataset, labels, giniBinaryTree->NumClasses(),
          batchTraining);
      break;

    case INFO_HOEFFDING:
      infoHoeffdingTree->Train(dataset, labels, infoHoeffdingTree->NumClasses(),
          batchTraining);
      break;

    case INFO_BINARY:
      infoBinaryTree->Train(dataset, labels, infoBinaryTree->NumClasses(),
          batchTraining);
      break;
  }
}

// Classify the given points.
inline void HoeffdingTreeModel::Classify(const arma::mat& dataset,
                                         arma::Row<size_t>& predictions)
    const
{
  // Call Classify() with the right model.
  switch (type)
  {
    case GINI_HOEFFDING:
      giniHoeffdingTree->Classify(dataset, predictions);
      break;

    case GINI_BINARY:
      giniBinaryTree->Classify(dataset, predictions);
      break;

    case INFO_HOEFFDING:
      infoHoeffdingTree->Classify(dataset, predictions);
      break;

    case INFO_BINARY:
      infoBinaryTree->Classify(dataset, predictions);
      break;
  }
}

// Classify the given points.
inline void HoeffdingTreeModel::Classify(const arma::mat& dataset,
                                         arma::Row<size_t>& predictions,
                                         arma::rowvec& probabilities)
    const
{
  // Call Classify() with the right model.
  switch (type)
  {
    case GINI_HOEFFDING:
      giniHoeffdingTree->Classify(dataset, predictions, probabilities);
      break;

    case GINI_BINARY:
      giniBinaryTree->Classify(dataset, predictions, probabilities);
      break;

    case INFO_HOEFFDING:
      infoHoeffdingTree->Classify(dataset, predictions, probabilities);
      break;

    case INFO_BINARY:
      infoBinaryTree->Classify(dataset, predictions, probabilities);
      break;
  }
}

// Utility function for counting the number of nodes.
template<typename TreeType>
size_t CountNodes(TreeType& tree)
{
  std::queue<TreeType*> queue;
  size_t nodes = 0;
  queue.push(&tree);
  while (!queue.empty())
  {
    TreeType* node = queue.front();
    queue.pop();
    ++nodes;

    for (size_t i = 0; i < node->NumChildren(); ++i)
      queue.push(&node->Child(i));
  }

  return nodes;
}

// Get the number of nodes in the tree.
inline size_t HoeffdingTreeModel::NumNodes() const
{
  // Call CountNodes() with the right type of tree.
  switch (type)
  {
    case GINI_HOEFFDING:
      return CountNodes(*giniHoeffdingTree);
    case GINI_BINARY:
      return CountNodes(*giniBinaryTree);
    case INFO_HOEFFDING:
      return CountNodes(*infoHoeffdingTree);
    case INFO_BINARY:
      return CountNodes(*infoBinaryTree);
  }

  return 0; // This should never happen!
}

} // namespace mlpack

#endif
