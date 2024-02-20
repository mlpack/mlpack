/**
 * @file methods/random_forest/random_forest_regressor.hpp
 * @author Dinesh Kumar
 *
 * Definition of the RandomForestRegressor class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_REGRESSOR_HPP
#define MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_REGRESSOR_HPP

#include <mlpack/methods/decision_tree/decision_tree_regressor.hpp>
#include "bootstrap.hpp"

namespace mlpack {

/**
 * This class implements a random forest regressor
 */
template<typename FitnessFunction = MSEGain,
         typename DimensionSelectionType = MultipleRandomDimensionSelect,
         template<typename> class NumericSplitType = BestBinaryNumericSplit,
         template<typename> class CategoricalSplitType = AllCategoricalSplit,
         bool UseBootstrap = true>
class RandomForestRegressor
{
 public:
  //! Allow access to the underlying decision tree type.
  typedef DecisionTreeRegressor<FitnessFunction, NumericSplitType, CategoricalSplitType,
      DimensionSelectionType> DecisionTreeType;

  /**
   * Construct the random forest without any training or specifying the number
   * of trees.  Predict() will throw an exception until Train() is called.
   */
  RandomForestRegressor();

  /**
   * Create a random forest, training on the given data and responses with
   * the given number of trees.  The minimumLeafSize and minimumGainSplit
   * parameters are given to each individual decision tree during tree building.
   * Optionally, you may specify a DimensionSelectionType to set parameters for
   * the strategy used to choose dimensions.
   *
   * @param dataset Dataset to train on.
   * @param responses Responses for each training point
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each tree's leaf nodes.
   * @param minimumGainSplit Minimum gain for splitting a decision tree node.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType>
  RandomForestRegressor(const MatType& dataset,
               const arma::Row<double>& responses,
               const size_t numTrees = 20,
               const size_t minimumLeafSize = 1,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Create a random forest, training on the given data and responses with
   * the given dataset info and the given number of trees.  The minimumLeafSize
   * and minimumGainSplit parameters are given to each individual decision tree
   * during tree building.  Optionally, you may specify a DimensionSelectionType
   * to set parameters for the strategy used to choose dimensions.
   * This constructor can be used to train on categorical data.
   *
   * @param dataset Dataset to train on.
   * @param datasetInfo Dimension info for the dataset.
   * @param responses Responses for each training point.
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each tree's leaf nodes.
   * @param minimumGainSplit Minimum gain for splitting a decision tree node.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType>
  RandomForestRegressor(const MatType& dataset,
               const data::DatasetInfo& datasetInfo,
               const arma::Row<double>& responses,
               const size_t numTrees = 20,
               const size_t minimumLeafSize = 1,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Create a random forest, training on the given data and responses with
   * weights with the given number of trees.  The minimumLeafSize parameter is
   * given to each individual decision tree during tree building.
   *
   * @param dataset Dataset to train on.
   * @param responses Responses for each training point.
   * @param weights Weights (importances) of each point in the dataset.
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each tree's leaf nodes.
   * @param minimumGainSplit Minimum gain for splitting a decision tree node.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType>
  RandomForestRegressor(const MatType& dataset,
               const arma::Row<double>& responses,
               const arma::rowvec& weights,
               const size_t numTrees = 20,
               const size_t minimumLeafSize = 1,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Create a random forest, training on the given data and responses with
   * weights with the given dataset info and the given number of trees.  The
   * minimumLeafSize and minimumGainSplit parameters are given to each
   * individual decision tree during tree building.  Optionally, you may specify
   * a DimensionSelectionType to set parameters for the strategy used to choose
   * dimensions.  This can be used for categorical weighted training.
   *
   * @param dataset Dataset to train on.
   * @param datasetInfo Dimension info for the dataset.
   * @param responses Responses for each training point.
   * @param weights Weights (importances) of each point in the dataset.
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each tree's leaf nodes.
   * @param minimumGainSplit Minimum gain for splitting a decision tree node.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType>
  RandomForestRegressor(const MatType& dataset,
               const data::DatasetInfo& datasetInfo,
               const arma::Row<double>& responses,
               const arma::rowvec& weights,
               const size_t numTrees = 20,
               const size_t minimumLeafSize = 1,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Train the random forest on the given data and responses with the given
   * number of trees.  The minimumLeafSize and minimumGainSplit parameters are
   * given to each individual decision tree during tree building.  Optionally,
   * you may specify a DimensionSelectionType to set parameters for the strategy
   * used to choose dimensions.
   *
   * @param data Dataset to train on.
   * @param responses Responses for each training point.
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each tree's leaf nodes.
   * @param minimumGainSplit Minimum gain for splitting a decision tree node.
   * @param maximumDepth Maximum depth for the tree.
   * @param warmStart When set to `true`, it adds `numTrees` new trees to the
   *     existing random forest otherwise a new forest is trained from scratch.
   * @param dimensionSelector Instantiated dimension selection policy.
   * @return The average entropy of all the decision trees trained under forest.
   */
  template<typename MatType>
  double Train(const MatType& data,
               const arma::Row<double>& responses,
               const size_t numTrees = 20,
               const size_t minimumLeafSize = 1,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               const bool warmStart = false,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Train the random forest on the given data and responses with the given
   * dataset info and the given number of trees.  The minimumLeafSize parameter
   * is given to each individual decision tree during tree building.
   * Optionally, you may specify a DimensionSelectionType to set parameters for
   * the strategy used to choose dimensions.
   * This overload can be used to train on categorical data.
   *
   * @param data Dataset to train on.
   * @param datasetInfo Dimension info for the dataset.
   * @param responses Responses for each training point.
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each tree's leaf nodes.
   * @param minimumGainSplit Minimum gain for splitting a decision tree node.
   * @param maximumDepth Maximum depth for the tree.
   * @param warmStart When set to `true`, it adds `numTrees` new trees to the
   *     existing random forest else a new forest is trained from scratch.
   * @param dimensionSelector Instantiated dimension selection policy.
   * @return The average entropy of all the decision trees trained under forest.
   */
  template<typename MatType>
  double Train(const MatType& data,
               const data::DatasetInfo& datasetInfo,
               const arma::Row<double>& responses,
               const size_t numTrees = 20,
               const size_t minimumLeafSize = 1,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               const bool warmStart = false,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Train the random forest on the given weighted data and responses with
   * the given number of trees.  The minimumLeafSize and minimumGainSplit
   * parameters are given to each individual decision tree during tree building.
   * Optionally, you may specify a DimensionSelectionType to set parameters for
   * the strategy used to choose dimensions.
   *
   * @param data Dataset to train on.
   * @param responses Responses for each training point.
   * @param weights Weights (importances) of each point in the dataset.
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each tree's leaf nodes.
   * @param minimumGainSplit Minimum gain for splitting a decision tree node.
   * @param maximumDepth Maximum depth for the tree.
   * @param warmStart When set to `true`, it adds `numTrees` new trees to the
   *     existing random forest else a new forest is trained from scratch.
   * @param dimensionSelector Instantiated dimension selection policy.
   * @return The average entropy of all the decision trees trained under forest.
   */
  template<typename MatType>
  double Train(const MatType& data,
               const arma::Row<double>& responses,
               const arma::rowvec& weights,
               const size_t numTrees = 20,
               const size_t minimumLeafSize = 1,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               const bool warmStart = false,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Train the random forest on the given weighted data and responses with
   * the given dataset info and the given number of trees.  The minimumLeafSize
   * and minimumGainSplit parameters are given to each individual decision tree
   * during tree building.  Optionally, you may specify a DimensionSelectionType
   * to set parameters for the strategy used to choose dimensions.  This
   * overload can be used for categorical weighted training.
   *
   * @param data Dataset to train on.
   * @param datasetInfo Dimension info for the dataset.
   * @param responses Responses for each training point.
   * @param weights Weights (importances) of each point in the dataset.
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each tree's leaf nodes.
   * @param minimumGainSplit Minimum gain for splitting a decision tree node.
   * @param maximumDepth Maximum depth for the tree.
   * @param warmStart When set to `true`, it adds `numTrees` new trees to the
   *     existing random forest else a new forest is trained from scratch.
   * @param dimensionSelector Instantiated dimension selection policy.
   * @return The average entropy of all the decision trees trained under forest.
   */
  template<typename MatType>
  double Train(const MatType& data,
               const data::DatasetInfo& datasetInfo,
               const arma::Row<double>& responses,
               const arma::rowvec& weights,
               const size_t numTrees = 20,
               const size_t minimumLeafSize = 1,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               const bool warmStart = false,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Predict the class of the given point.  If the random forest has not been
   * trained, this will throw an exception.
   *
   * @param point Point to be classified.
   */
  template<typename VecType>
  double Predict(const VecType& point) const;

  /**
   * Predict the classes of each point in the given dataset.  If the random
   * forest has not been trained, this will throw an exception.
   *
   * @param data Set of points to predict.
   * @param predictions Output predictions for each point in the dataset.
   */
  template<typename MatType>
  void Predict(const MatType& data,
                arma::Row<double>& predictions) const;

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
   * @param responses responses for the dataset.
   * @param weights Weights for each point in the dataset (may be ignored).
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for splitting a decision tree node.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   * @param warmStart When set to `true`, it fits new trees and add them to the
   *     previous forest else a new forest is trained from scratch.
   * @tparam UseWeights Whether or not to use the weights parameter.
   * @tparam UseDatasetInfo Whether or not to use the datasetInfo parameter.
   * @tparam MatType The type of data matrix (i.e. arma::mat).
   * @return The average entropy of all the decision trees trained under forest.
   */
  template<bool UseWeights, bool UseDatasetInfo, typename MatType>
  double Train(const MatType& data,
               const data::DatasetInfo& datasetInfo,
               const arma::Row<double>& responses,
               const arma::rowvec& weights,
               const size_t numTrees,
               const size_t minimumLeafSize,
               const double minimumGainSplit,
               const size_t maximumDepth,
               DimensionSelectionType& dimensionSelector,
               const bool warmStart = false);

  //! The trees in the forest.
  std::vector<DecisionTreeType> trees;

  //! The average gain of the forest.
  double avgGain;
};

/**
 * Convenience typedef for Extra Trees. (Extremely Randomized Trees Forest)
 *
 * @code
 * @article{10.1007/s10994-006-6226-1,
 *   author = {Geurts, Pierre and Ernst, Damien and Wehenkel, Louis},
 *   title = {Extremely Randomized Trees},
 *   year = {2006},
 *   issue_date = {April 2006},
 *   publisher = {Kluwer Academic Publishers},
 *   address = {USA},
 *   volume = {63},
 *   number = {1},
 *   issn = {0885-6125},
 *   url = {https://doi.org/10.1007/s10994-006-6226-1},
 *   doi = {10.1007/s10994-006-6226-1},
 *   journal = {Mach. Learn.},
 *   month = apr,
 *   pages = {3–42},
 *   numpages = {40},
 * }
 * @endcode
 */
template<typename FitnessFunction = MSEGain,
         typename DimensionSelectionType = MultipleRandomDimensionSelect,
         template<typename> class CategoricalSplitType = AllCategoricalSplit>
using ExtraTreesRegressor = RandomForestRegressor<FitnessFunction,
                                DimensionSelectionType,
                                RandomBinaryNumericSplit,
                                CategoricalSplitType,
                                false>;

} // namespace mlpack

// Include implementation.
#include "random_forest_regressor_impl.hpp"

#endif
