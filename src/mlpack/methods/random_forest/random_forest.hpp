/**
 * @file methods/random_forest/random_forest.hpp
 * @author Ryan Curtin
 *
 * Definition of the RandomForest class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_HPP
#define MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_HPP

#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>
#include "bootstrap.hpp"

namespace mlpack {
namespace tree {

/**
 * The RandomForest class provides an implementation of random forests,
 * described in Breiman's seminal paper:
 *
 * @code
 * @article{breiman2001random,
 *   title={Random forests},
 *   author={Breiman, Leo},
 *   journal={Machine Learning},
 *   volume={45},
 *   number={1},
 *   pages={5--32},
 *   year={2001},
 *   publisher={Springer}
 * }
 * @endcode
 */
template<typename FitnessFunction = GiniGain,
         typename DimensionSelectionType = MultipleRandomDimensionSelect,
         template<typename> class NumericSplitType = BestBinaryNumericSplit,
         template<typename> class CategoricalSplitType = AllCategoricalSplit>
class RandomForest
{
 public:
  //! Allow access to the underlying decision tree type.
  typedef DecisionTree<FitnessFunction, NumericSplitType, CategoricalSplitType,
      DimensionSelectionType> DecisionTreeType;

  /**
   * Construct the random forest without any training or specifying the number
   * of trees.  Predict() will throw an exception until Train() is called.
   */
  RandomForest() { }

  /**
   * Create a random forest, training on the given labeled training data with
   * the given number of trees.  The minimumLeafSize and minimumGainSplit
   * parameters are given to each individual decision tree during tree building.
   * Optionally, you may specify a DimensionSelectionType to set parameters for
   * the strategy used to choose dimensions.
   *
   * @param dataset Dataset to train on.
   * @param labels Labels for dataset.
   * @param numClasses Number of classes in dataset.
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each tree's leaf nodes.
   * @param minimumGainSplit Minimum gain for splitting a decision tree node.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType>
  RandomForest(const MatType& dataset,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const size_t numTrees = 20,
               const size_t minimumLeafSize = 1,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Create a random forest, training on the given labeled training data with
   * the given dataset info and the given number of trees.  The minimumLeafSize
   * and minimumGainSplit parameters are given to each individual decision tree
   * during tree building.  Optionally, you may specify a DimensionSelectionType
   * to set parameters for the strategy used to choose dimensions.
   * This constructor can be used to train on categorical data.
   *
   * @param dataset Dataset to train on.
   * @param datasetInfo Dimension info for the dataset.
   * @param labels Labels for dataset.
   * @param numClasses Number of classes in dataset.
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each tree's leaf nodes.
   * @param minimumGainSplit Minimum gain for splitting a decision tree node.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType>
  RandomForest(const MatType& dataset,
               const data::DatasetInfo& datasetInfo,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const size_t numTrees = 20,
               const size_t minimumLeafSize = 1,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Create a random forest, training on the given weighted labeled training
   * data with the given number of trees.  The minimumLeafSize parameter is
   * given to each individual decision tree during tree building.
   *
   * @param dataset Dataset to train on.
   * @param labels Labels for dataset.
   * @param numClasses Number of classes in dataset.
   * @param weights Weights (importances) of each point in the dataset.
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each tree's leaf nodes.
   * @param minimumGainSplit Minimum gain for splitting a decision tree node.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType>
  RandomForest(const MatType& dataset,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const arma::rowvec& weights,
               const size_t numTrees = 20,
               const size_t minimumLeafSize = 1,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Create a random forest, training on the given weighted labeled training
   * data with the given dataset info and the given number of trees.  The
   * minimumLeafSize and minimumGainSplit parameters are given to each
   * individual decision tree during tree building.  Optionally, you may specify
   * a DimensionSelectionType to set parameters for the strategy used to choose
   * dimensions.  This can be used for categorical weighted training.
   *
   * @param dataset Dataset to train on.
   * @param datasetInfo Dimension info for the dataset.
   * @param labels Labels for dataset.
   * @param numClasses Number of classes in dataset.
   * @param weights Weights (importances) of each point in the dataset.
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each tree's leaf nodes.
   * @param minimumGainSplit Minimum gain for splitting a decision tree node.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType>
  RandomForest(const MatType& dataset,
               const data::DatasetInfo& datasetInfo,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const arma::rowvec& weights,
               const size_t numTrees = 20,
               const size_t minimumLeafSize = 1,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Train the random forest on the given labeled training data with the given
   * number of trees.  The minimumLeafSize and minimumGainSplit parameters are
   * given to each individual decision tree during tree building.  Optionally,
   * you may specify a DimensionSelectionType to set parameters for the strategy
   * used to choose dimensions.
   *
   * @param data Dataset to train on.
   * @param labels Labels for dataset.
   * @param numClasses Number of classes in dataset.
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each tree's leaf nodes.
   * @param minimumGainSplit Minimum gain for splitting a decision tree node.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   * @return The average entropy of all the decision trees trained under forest.
   */
  template<typename MatType>
  double Train(const MatType& data,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const size_t numTrees = 20,
               const size_t minimumLeafSize = 1,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Train the random forest on the given labeled training data with the given
   * dataset info and the given number of trees.  The minimumLeafSize parameter
   * is given to each individual decision tree during tree building.
   * Optionally, you may specify a DimensionSelectionType to set parameters for
   * the strategy used to choose dimensions.
   * This
   * overload can be used to train on categorical data.
   *
   * @param data Dataset to train on.
   * @param datasetInfo Dimension info for the dataset.
   * @param labels Labels for dataset.
   * @param numClasses Number of classes in dataset.
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each tree's leaf nodes.
   * @param minimumGainSplit Minimum gain for splitting a decision tree node.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   * @return The average entropy of all the decision trees trained under forest.
   */
  template<typename MatType>
  double Train(const MatType& data,
               const data::DatasetInfo& datasetInfo,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const size_t numTrees = 20,
               const size_t minimumLeafSize = 1,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Train the random forest on the given weighted labeled training data with
   * the given number of trees.  The minimumLeafSize and minimumGainSplit
   * parameters are given to each individual decision tree during tree building.
   * Optionally, you may specify a DimensionSelectionType to set parameters for
   * the strategy used to choose dimensions.
   *
   * @param data Dataset to train on.
   * @param labels Labels for dataset.
   * @param numClasses Number of classes in dataset.
   * @param weights Weights (importances) of each point in the dataset.
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each tree's leaf nodes.
   * @param minimumGainSplit Minimum gain for splitting a decision tree node.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   * @return The average entropy of all the decision trees trained under forest.
   */
  template<typename MatType>
  double Train(const MatType& data,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const arma::rowvec& weights,
               const size_t numTrees = 20,
               const size_t minimumLeafSize = 1,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Train the random forest on the given weighted labeled training data with
   * the given dataset info and the given number of trees.  The minimumLeafSize
   * and minimumGainSplit parameters are given to each individual decision tree
   * during tree building.  Optionally, you may specify a DimensionSelectionType
   * to set parameters for the strategy used to choose dimensions.  This
   * overload can be used for categorical weighted training.
   *
   * @param data Dataset to train on.
   * @param datasetInfo Dimension info for the dataset.
   * @param labels Labels for dataset.
   * @param numClasses Number of classes in dataset.
   * @param weights Weights (importances) of each point in the dataset.
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each tree's leaf nodes.
   * @param minimumGainSplit Minimum gain for splitting a decision tree node.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   * @return The average entropy of all the decision trees trained under forest.
   */
  template<typename MatType>
  double Train(const MatType& data,
               const data::DatasetInfo& datasetInfo,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const arma::rowvec& weights,
               const size_t numTrees = 20,
               const size_t minimumLeafSize = 1,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Predict the class of the given point.  If the random forest has not been
   * trained, this will throw an exception.
   *
   * @param point Point to be classified.
   */
  template<typename VecType>
  size_t Classify(const VecType& point) const;

  /**
   * Predict the class of the given point and return the predicted class
   * probabilities for each class.  If the random forest has not been trained,
   * this will throw an exception.
   *
   * @param point Point to be classified.
   * @param prediction size_t to store predicted class in.
   * @param probabilities Output vector of class probabilities.
   */
  template<typename VecType>
  void Classify(const VecType& point,
                size_t& prediction,
                arma::vec& probabilities) const;

  /**
   * Predict the classes of each point in the given dataset.  If the random
   * forest has not been trained, this will throw an exception.
   *
   * @param data Dataset to be classified.
   * @param predictions Output predictions for each point in the dataset.
   */
  template<typename MatType>
  void Classify(const MatType& data,
                arma::Row<size_t>& predictions) const;

  /**
   * Predict the classes of each point in the given dataset, also returning the
   * predicted class probabilities for each point.  If the random forest has not
   * been trained, this will throw an exception.
   *
   * @param data Dataset to be classified.
   * @param predictions Output predictions for each point in the dataset.
   * @param probabilities Output matrix of class probabilities for each point.
   */
  template<typename MatType>
  void Classify(const MatType& data,
                arma::Row<size_t>& predictions,
                arma::mat& probabilities) const;

  //! Access a tree in the forest.
  const DecisionTreeType& Tree(const size_t i) const { return trees[i]; }
  //! Modify a tree in the forest (be careful!).
  DecisionTreeType& Tree(const size_t i) { return trees[i]; }

  //! Get the number of trees in the forest.
  size_t NumTrees() const { return trees.size(); }

  /**
   * Serialize the random forest.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /**
   * Perform the training of the decision tree.  The template bool parameters
   * control whether or not the datasetInfo or weights arguments should be
   * ignored.
   *
   * @param data Dataset to train on.
   * @param datasetInfo Dimension information for the dataset (may be ignored).
   * @param labels Labels for the dataset.
   * @param numClasses Number of classes in the dataset.
   * @param weights Weights for each point in the dataset (may be ignored).
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for splitting a decision tree node.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   * @tparam UseWeights Whether or not to use the weights parameter.
   * @tparam UseDatasetInfo Whether or not to use the datasetInfo parameter.
   * @tparam MatType The type of data matrix (i.e. arma::mat).
   * @return The average entropy of all the decision trees trained under forest.
   */
  template<bool UseWeights, bool UseDatasetInfo, typename MatType>
  double Train(const MatType& data,
               const data::DatasetInfo& datasetInfo,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const arma::rowvec& weights,
               const size_t numTrees,
               const size_t minimumLeafSize,
               const double minimumGainSplit,
               const size_t maximumDepth,
               DimensionSelectionType& dimensionSelector);

  //! The trees in the forest.
  std::vector<DecisionTreeType> trees;
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "random_forest_impl.hpp"

#endif
