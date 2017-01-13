/**
 * @param hoeffding_tree_model.cpp
 * @author Ryan Curtin
 *
 * Implementation of the HoeffdingTreeModel class.
 */
#include "hoeffding_tree_model.hpp"

using namespace mlpack;
using namespace mlpack::tree;

// Constructor.
HoeffdingTreeModel::HoeffdingTreeModel(const TreeType& type) :
    type(type),
    giniHoeffdingTree(NULL),
    giniBinaryTree(NULL),
    infoHoeffdingTree(NULL),
    infoBinaryTree(NULL)
{
  // Nothing to do.
}

// Copy constructor.
HoeffdingTreeModel::HoeffdingTreeModel(const HoeffdingTreeModel& other) :
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
HoeffdingTreeModel::HoeffdingTreeModel(HoeffdingTreeModel&& other) :
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
HoeffdingTreeModel& HoeffdingTreeModel::operator=(
    const HoeffdingTreeModel& other)
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
  if (type == GINI_HOEFFDING)
    giniHoeffdingTree = new GiniHoeffdingTreeType(*other.giniHoeffdingTree);
  else if (type == GINI_BINARY)
    giniBinaryTree = new GiniBinaryTreeType(*other.giniBinaryTree);
  else if (type == INFO_HOEFFDING)
    infoHoeffdingTree = new InfoHoeffdingTreeType(*other.infoHoeffdingTree);
  else if (type == INFO_BINARY)
    infoBinaryTree = new InfoBinaryTreeType(*other.infoBinaryTree);

  return *this;
}

// Move operator.
HoeffdingTreeModel& HoeffdingTreeModel::operator=(HoeffdingTreeModel&& other)
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

  return *this;
}

// Destructor.
HoeffdingTreeModel::~HoeffdingTreeModel()
{
  delete giniHoeffdingTree;
  delete giniBinaryTree;
  delete infoHoeffdingTree;
  delete infoBinaryTree;
}

// Create the model.
void HoeffdingTreeModel::BuildModel(
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
        HoeffdingDoubleNumericSplit<InformationGain> ns(0, bins,
            observationsBeforeBinning);

        infoHoeffdingTree = new InfoHoeffdingTreeType(dataset, datasetInfo,
            labels, numClasses, batchTraining, successProbability, maxSamples,
            checkInterval, minSamples,
            HoeffdingCategoricalSplit<InformationGain>(0, 0), ns);
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
void HoeffdingTreeModel::Train(const arma::mat& dataset,
                               const arma::Row<size_t>& labels,
                               const bool batchTraining)
{
  // Depending on the type, pass through once.
  switch (type)
  {
    case GINI_HOEFFDING:
      giniHoeffdingTree->Train(dataset, labels, batchTraining);
      break;

    case GINI_BINARY:
      giniBinaryTree->Train(dataset, labels, batchTraining);
      break;

    case INFO_HOEFFDING:
      infoHoeffdingTree->Train(dataset, labels, batchTraining);
      break;

    case INFO_BINARY:
      infoBinaryTree->Train(dataset, labels, batchTraining);
      break;
  }
}

// Classify the given points.
void HoeffdingTreeModel::Classify(const arma::mat& dataset,
                                  arma::Row<size_t>& predictions) const
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
void HoeffdingTreeModel::Classify(const arma::mat& dataset,
                                  arma::Row<size_t>& predictions,
                                  arma::rowvec& probabilities) const
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
size_t HoeffdingTreeModel::NumNodes() const
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
