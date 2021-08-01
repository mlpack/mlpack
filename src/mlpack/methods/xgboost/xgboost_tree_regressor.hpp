/**
 * @file methods/xgboost/xgboost_tree_regressor.hpp
 * @author Rishabh Garg
 *
 * Definition of the xgboost tree regressor class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_XGBOOST_XGBOOST_TREE_REGRESSOR_HPP
#define MLPACK_METHODS_XGBOOST_XGBOOST_TREE_REGRESSOR_HPP

#include <mlpack/methods/decision_tree/decision_tree_regressor.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>

namespace mlpack {
namespace ensemble {

/**
 * The XGboostTreeRegressor class provides the implementation of Gradient
 * Boosted Decision Tree Regressor as described in the XGBoost paper:
 *
 * @code
 * @inproceedings{Chen:2016:XST:2939672.2939785,
 *   author = {Chen, Tianqi and Guestrin, Carlos},
 *   title = {{XGBoost}: A Scalable Tree Boosting System},
 *   booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on
 *                Knowledge Discovery and Data Mining},
 *   series = {KDD '16},
 *   year = {2016},
 *   isbn = {978-1-4503-4232-2},
 *   location = {San Francisco, California, USA},
 *   pages = {785--794},
 *   numpages = {10},
 *   url = {http://doi.acm.org/10.1145/2939672.2939785},
 *   doi = {10.1145/2939672.2939785},
 *   acmid = {2939785},
 *   publisher = {ACM},
 *   address = {New York, NY, USA},
 *   keywords = {large-scale machine learning},
 * }
 * @endcode
 */
template<typename LossFunction = SSELoss,
         typename DimensionSelectionType =
             mlpack::tree::MultipleRandomDimensionSelect,
         template <typename> class NumericSplitType = XGBExactNumericSplit>
class XGBoostTreeRegressor
{
 public:
  //! Allow access to the underlying decision tree type.
  typedef mlpack::tree::DecisionTreeRegressor<FitnessFunction,
      NumericSplitType, mlpack::tree::AllCategoricalSplit,
      DimensionSelectionType> DecisionTreeType;
  /**
   * Construct the xgboost without any training or specifying the number
   * of trees.  Predict() will throw an exception until Train() is called.
   */
  XGBoostTreeRegressor();

  /**
   * Construct the xgboost forest and train.
   */
  template<typename MatType, typename ResponsesType>
  XGBoostTreeRegressor(const MatType& dataset,
                       const ResponsesType& responses,
                       const size_t numTrees = 100,
                       const size_t maxDepth = 6,
                       const double eta = 0.3,
                       const double minChildWeight = 1,
                       const double gamma = 0,
                       const double lambda = 1,
                       const double alpha = 0,
                       const bool warmStart = false,
                       DimensionSelectionType dimensionSelector =
                           DimensionSelectionType());

  // Train the model.
  template<typename MatType, typename ResponsesType>
  double Train(const MatType& dataset,
               const ResponsesType& responses,
               const size_t numTrees = 100,
               const size_t maxDepth = 6,
               const double minChildWeight = 1,
               const double gamma = 0,
               const double lambda = 1,
               const double alpha = 0,
               const bool warmStart = false,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  template<typename VecType>
  typename VecType::elem_type Predict(const VecType& point) const;

  template<typename MatType>
  void Predict(const MatType& data, arma::rowvec& predictions) const;

  //! Access a tree in the forest.
  const DecisionTreeType& Tree(const size_t i) const { return trees[i]; }
  //! Modify a tree in the forest (be careful!).
  DecisionTreeType& Tree(const size_t i) { return trees[i]; }
  //! Access the learning rate.
  const double& Eta() const { return eta; }
  //! Modify the learning rate (be careful).
  double& Eta() { return eta; }

  //! Get the number of trees in the forest.
  size_t NumTrees() const { return trees.size(); }

  /**
   * Serialize the random forest.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  std::vector<DecisionTreeType*> trees;
  double eta;
  double initialPred;
  double avgGain;

  template<bool UseDatasetInfo, typename MatType, typename ResponsesType>
  double Train(const MatType& dataset,
               const data::DatasetInfo& datasetInfo,
               const ResponsesType& responses,
               const size_t numTrees = 100,
               const size_t maxDepth = 6,
               const double minChildWeight = 1,
               const double gamma = 0,
               const double lambda = 1,
               const double alpha = 0,
               const bool warmStart = false,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());
}

} // namespace ensemble
} // namespace mlpack

// Include implementation.
#include "xgboost_impl.hpp"

#endif
