/**
 * @file methods/adaboost/adaboost_regressor.hpp
 * @author Dinesh Kumar
 *
 * The Adaboost regressor class.
 * An AdaBoost regressor is a meta-estimator that begins by fitting a
 * regressor on the original dataset and then fits additional copies of the
 * regressor on the same dataset but where the weights of instances are
 * adjusted according to the error of the current prediction. As such,
 * subsequent regressors focus more on difficult cases.
 *
 * This class implements the algorithm known as AdaBoost.R2.
 *
 * @code
 * @article{
 *   author = {Harris Drucker},
 *   title = {Improving Regressors using Boosting Techniques},
 *   publication name = {International Conference on Machine Learning},
 *   month = july,
 *   year = {1997},
 *   issn = {978-1-55860-486-5},
 *   pages = {107–-115},
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ADABOOST_ADABOOST_REGRESSOR_HPP
#define MLPACK_METHODS_ADABOOST_ADABOOST_REGRESSOR_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree_regressor.hpp>
#include "loss_functions/loss_functions.hpp"

namespace mlpack {

/**
 * LossFunctionType : to calculate the loss for each traning example.
 * FitnessFunction : the measure of goodness to use when deciding on tree splits.
 * DimensionSelectionType: the strategy used for proposing dimensions to attempt to split on.
 * NumericSplitType: the strategy used for finding splits on numeric data dimensions.
 * CategoricalSplitType: the strategy used for finding splits on categorical data dimensions.
 * UseBootstrap: a boolean indicating whether or not to use a bootstrap sample when training 
 *               each tree in the forest.
 */
template<typename LossFunctionType = LinearLoss,
         typename FitnessFunction = MSEGain,
         typename DimensionSelectionType = MultipleRandomDimensionSelect,
         template<typename> class NumericSplitType = BestBinaryNumericSplit,
         template<typename> class CategoricalSplitType = AllCategoricalSplit>
class AdaBoostRegressor
{
public:
  //! Allow access to the underlying decision tree type.
  typedef DecisionTreeRegressor<FitnessFunction, NumericSplitType, CategoricalSplitType,
      DimensionSelectionType> DecisionTreeType;

  /**
   * Construct the AdaBoost regressor without any training or specifying the number
   * of trees.  Predict() will throw an exception until Train() is called.
   */
  AdaBoostRegressor();

  /**
   * Create a AdaBoost regressor, training on the given data and responses with
   * the given number of trees.  The minimumLeafSize and minimumGainSplit
   * parameters are given to each individual decision tree during tree building.
   * Optionally, you may specify a DimensionSelectionType to set parameters for
   * the strategy used to choose dimensions.
   *
   * @param dataset Dataset to train on.
   * @param responses Responses for each training point.
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each tree's leaf nodes.
   * @param minimumGainSplit Minimum gain for splitting a decision tree node.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType, typename ElemType>
  AdaBoostRegressor(const MatType& dataset,
                    const arma::Row<ElemType>& responses,
                    const size_t numTrees = 20,
                    const size_t minimumLeafSize = 10,
                    const double minimumGainSplit = 1e-7,
                    const size_t maximumDepth = 4,
                    DimensionSelectionType dimensionSelector =
                        DimensionSelectionType());

  /**
   * Create a AdaBoost regressor, training on the given data and responses with
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
  template<typename MatType, typename ElemType>
  AdaBoostRegressor(const MatType& dataset,
                    const data::DatasetInfo& datasetInfo,
                    const arma::Row<ElemType>& responses,
                    const size_t numTrees = 20,
                    const size_t minimumLeafSize = 10,
                    const double minimumGainSplit = 1e-7,
                    const size_t maximumDepth = 4,
                    DimensionSelectionType dimensionSelector =
                        DimensionSelectionType());

  /**
   * Train the AdaBoost regressor on the given data and responses with the given
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
   * @param dimensionSelector Instantiated dimension selection policy.
   * @return The average entropy of all the decision trees trained under forest.
   */
  template<typename MatType, typename ElemType>
  double Train(const MatType& data,
               const arma::Row<ElemType>& responses,
               const size_t numTrees = 20,
               const size_t minimumLeafSize = 10,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 4,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Train the AdaBoost regressor on the given data and responses with the given
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
   * @param dimensionSelector Instantiated dimension selection policy.
   * @return The average entropy of all the decision trees trained under forest.
   */
  template<typename MatType, typename ElemType>
  double Train(const MatType& data,
               const data::DatasetInfo& datasetInfo,
               const arma::Row<ElemType>& responses,
               const size_t numTrees = 20,
               const size_t minimumLeafSize = 10,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 4,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Predict the class of the given point. If the AdaBoost regressor has not been
   * trained, this will throw an exception.
   *
   * @param point Point to be classified.
   */
  template<typename VecType>
  double Predict(const VecType& point) const;

  /**
   * Predict the classes of each point in the given dataset.  If the AdaBoost
   * regressor has not been trained, this will throw an exception.
   *
   * @param data Set of points to predict.
   * @param predictions Output predictions for each point in the dataset.
   */
  template<typename MatType, typename ElemType>
  void Predict(const MatType& data,
                arma::Row<ElemType>& predictions) const;

  //! Access a tree in the forest.
  const DecisionTreeType& Tree(const size_t i) const { return trees[i]; }
  //! Modify a tree in the forest (be careful!).
  DecisionTreeType& Tree(const size_t i) { return trees[i]; }

  //! Get the number of trees in the forest.
  size_t NumTrees() const { return trees.size(); }

  /**
   * Serialize the AdaBoost.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /**
   * Perform the training of the AdaBoost regressor. The template bool parameters
   * control whether or not the datasetInfo or weights arguments should be
   * ignored.
   *
   * @param data Dataset to train on.
   * @param datasetInfo Dimension information for the dataset (may be ignored).
   * @param responses responses for the dataset.
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
  template<bool UseDatasetInfo, typename MatType, typename ElemType>
  double Train(const MatType& data,
               const data::DatasetInfo& datasetInfo,
               const arma::Row<ElemType>& responses,
               const size_t numTrees,
               const size_t minimumLeafSize,
               const double minimumGainSplit,
               const size_t maximumDepth,
               DimensionSelectionType& dimensionSelector);

  //! The trees in the forest.
  std::vector<DecisionTreeType> trees;

  //! Confidence of each tree.
  std::vector<double> confidence;

  //! The average gain of the forest.
  double avgGain;
};

} // namespace mlpack

// Include implementation
#include "adaboost_regressor_impl.hpp"

#endif