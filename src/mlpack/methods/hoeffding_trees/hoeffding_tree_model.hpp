/**
 * @file methods/hoeffding_trees/hoeffding_tree_model.hpp
 * @author Ryan Curtin
 *
 * A serializable model for the mlpack_hoeffding_tree command-line program.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HOEFFDING_TREE_HOEFFDING_TREE_MODEL_HPP
#define MLPACK_METHODS_HOEFFDING_TREE_HOEFFDING_TREE_MODEL_HPP

#include "hoeffding_tree.hpp"
#include "binary_numeric_split.hpp"
#include "information_gain.hpp"
#include <queue>

namespace mlpack {

/**
 * This class is a serializable Hoeffding tree model that can hold four
 * different types of Hoeffding trees.  It is meant to be used by the
 * command-line program for Hoeffding trees.
 */
class HoeffdingTreeModel
{
 public:
  //! This enumerates the four types of trees we can hold.
  enum TreeType
  {
    GINI_HOEFFDING,
    GINI_BINARY,
    INFO_HOEFFDING,
    INFO_BINARY
  };

  //! Convenience typedef for GINI_HOEFFDING tree type.
  using GiniHoeffdingTreeType = HoeffdingTree<GiniImpurity,
      HoeffdingDoubleNumericSplit, HoeffdingCategoricalSplit>;
  //! Convenience typedef for GINI_BINARY tree type.
  using GiniBinaryTreeType = HoeffdingTree<GiniImpurity,
      BinaryDoubleNumericSplit, HoeffdingCategoricalSplit>;
  //! Convenience typedef for INFO_HOEFFDING tree type.
  using InfoHoeffdingTreeType = HoeffdingTree<HoeffdingInformationGain,
      HoeffdingDoubleNumericSplit, HoeffdingCategoricalSplit>;
  //! Convenience typedef for INFO_BINARY tree type.
  using InfoBinaryTreeType = HoeffdingTree<HoeffdingInformationGain,
      BinaryDoubleNumericSplit, HoeffdingCategoricalSplit>;

  /**
   * Construct the Hoeffding tree model, but don't initialize any tree.
   *
   * Be sure to call Train() before doing anything with the model!
   *
   * @param type Type of tree that will be used.
   */
  HoeffdingTreeModel(const TreeType& type = GINI_HOEFFDING);

  /**
   * Copy the Hoeffding tree model from the given other model.
   *
   * @param other Hoeffding tree model to copy.
   */
  HoeffdingTreeModel(const HoeffdingTreeModel& other);

  /**
   * Move the Hoeffding tree model from the given other model.
   *
   * @param other Hoeffding tree model to move.
   */
  HoeffdingTreeModel(HoeffdingTreeModel&& other);

  /**
   * Copy the Hoeffding tree model from the given other model.
   *
   * @param other Hoeffding tree model to copy.
   */
  HoeffdingTreeModel& operator=(const HoeffdingTreeModel& other);

  /**
   * Move the Hoeffding tree model from the given other model.
   *
   * @param other Hoeffding tree model to move.
   */
  HoeffdingTreeModel& operator=(HoeffdingTreeModel&& other);

  /**
   * Clean up the given model.
   */
  ~HoeffdingTreeModel();

  /**
   * Train the model on the given dataset with the given labels.  This method
   * just passes to the appropriate HoeffdingTree<...> constructor, and will
   * train with one pass over the dataset.
   *
   * @param dataset Dataset to train on.
   * @param datasetInfo Information about dimensions of dataset.
   * @param labels Labels for training set.
   * @param numClasses Number of classes in dataset.
   * @param batchTraining Whether or not to train in batch.
   * @param successProbability Probability of success required in Hoeffding
   *      bound before a split can happen.
   * @param maxSamples Maximum number of samples before a split is forced.
   * @param checkInterval Number of samples required before each split check.
   * @param minSamples If the node has seen this many points or fewer, no split
   *      will be allowed.
   * @param bins Number of bins, for Hoeffding numeric split.
   * @param observationsBeforeBinning Number of observations before binning, for
   *      Hoeffding numeric split.
   */
  void BuildModel(const arma::mat& dataset,
                  const data::DatasetInfo& datasetInfo,
                  const arma::Row<size_t>& labels,
                  const size_t numClasses,
                  const bool batchTraining,
                  const double successProbability,
                  const size_t maxSamples,
                  const size_t checkInterval,
                  const size_t minSamples,
                  const size_t bins,
                  const size_t observationsBeforeBinning);

  /**
   * Train in streaming mode on the given dataset.  This takes one pass.  Be
   * sure that BuildModel() has been called first!
   *
   * @param dataset Dataset to train on.
   * @param labels Labels for training set.
   * @param batchTraining Whether or not to train in batch.
   */
  void Train(const arma::mat& dataset,
             const arma::Row<size_t>& labels,
             const bool batchTraining);

  /**
   * Using the model, classify the given test points.  Be sure that BuildModel()
   * has been called first!
   *
   * @param dataset Dataset to classify.
   * @param predictions Vector to store predictions for test points in.
   */
  void Classify(const arma::mat& dataset,
                arma::Row<size_t>& predictions) const;

  /**
   * Using the model, classify the given test points, returning class
   * probabilities.
   *
   * @param dataset Dataset to classify.
   * @param predictions Vector to store predictions for test points in.
   * @param probabilities Vector to store probabilities for test points in.
   */
  void Classify(const arma::mat& dataset,
                arma::Row<size_t>& predictions,
                arma::rowvec& probabilities) const;

  /**
   * Get the number of nodes in the tree.
   */
  size_t NumNodes() const;

  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    // Clear memory if needed.
    if (cereal::is_loading<Archive>())
    {
      delete giniHoeffdingTree;
      delete giniBinaryTree;
      delete infoHoeffdingTree;
      delete infoBinaryTree;

      giniHoeffdingTree = NULL;
      giniBinaryTree = NULL;
      infoHoeffdingTree = NULL;
      infoBinaryTree = NULL;
    }

    ar(CEREAL_NVP(type));

    // Fake dataset info may be needed to create fake trees.
    data::DatasetInfo info;
    if (type == GINI_HOEFFDING)
      ar(CEREAL_POINTER(giniHoeffdingTree));
    else if (type == GINI_BINARY)
      ar(CEREAL_POINTER(giniBinaryTree));
    else if (type == INFO_HOEFFDING)
      ar(CEREAL_POINTER(infoHoeffdingTree));
    else if (type == INFO_BINARY)
      ar(CEREAL_POINTER(infoBinaryTree));
  }

 private:
  //! The type of tree we are using.
  TreeType type;

  //! This is used if we are using the Gini impurity and the Hoeffding numeric
  //! split.
  GiniHoeffdingTreeType* giniHoeffdingTree;

  //! This is used if we are using the Gini impurity and the binary numeric
  //! split.
  GiniBinaryTreeType* giniBinaryTree;

  //! This is used if we are using the information gain and the Hoeffding
  //! numeric split.
  InfoHoeffdingTreeType* infoHoeffdingTree;

  //! This is used if we are using the information gain and the binary numeric
  //! split.
  InfoBinaryTreeType* infoBinaryTree;
};

} // namespace mlpack

// Include implementation.
#include "hoeffding_tree_model_impl.hpp"

#endif
