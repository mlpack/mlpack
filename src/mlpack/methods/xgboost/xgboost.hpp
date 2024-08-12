/**
 * @file methods/xgboost/xgboost.hpp
 * @author Abhimanyu Dayal
 *
 * XGBoost class. XGBoost optimises the XGBoost algorithm by implementing
 * various addition functionalities on top of it such as regularisation, pruning etc.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

// Include guard - Used to prevent including double copies of the same code
#ifndef MLPACK_METHODS_XGBOOST_XGBOOST_HPP
#define MLPACK_METHODS_XGBOOST_XGBOOST_HPP

// Importing base components required to write mlpack methods.
#include <mlpack/core.hpp>
#include "xgbtree/xgbtree.hpp"

// Written in mlpack namespace.
namespace mlpack {
/**
 * The XGBoost class. XGBoost is a boosting algorithm, meaning that it
 * combines an ensemble of trees to produce a strong learner.
 * 
 * XGBoost is generally implemented using Decision Trees, or more specifically 
 * Decision Stumps i.e. weak learner Decision Trees with low depth.
 * 
 * @tparam MatType Data matrix type (i.e. arma::mat or arma::sp_mat).
 */


template<typename MatType = arma::mat>
class XGBoost 
{
 public: 

  typedef typename MatType::elem_type ElemType;

  /**
   * Constructor for creating XGBoost without training. 
   * Be sure to call Train() before calling Classify()
   */
  XGBoost();

  /**
   * Constructor in case trees aren't defined.
   *
   * @param data Input data
   * @param labels Corresponding labels
   * @param numClasses Number of classes
   * @param numModels Number of trees
   */
  XGBoost(const MatType& data,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const size_t numModels);

  /**
   * Constructor for a XGBoost model. Any extra parameters are used as
   * hyperparameters for the weak learner. These should be the last arguments
   * to the weak learner's constructor or `Train()` function (i.e. anything
   * after `numClasses` or `weights`).
   *
   * @param data Input data.
   * @param labels Corresponding labels.
   * @param numClasses The number of classes.
   * @param numModels Number of trees.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   */
  XGBoost(const MatType& data,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const size_t numModels,
               const size_t minimumLeafSize,
               const double minimumGainSplit,
               const size_t maximumDepth);

  //! Set the number of trees explicitly.
  void SetNumModels(const size_t x) {numModels = x;}

  //! Get the number of classes this model is trained on.
  size_t NumClasses() const { return numClasses; }

  //! Get the number of trees .
  size_t NumModels() const { return numModels; }

  /**
   * Train XGBoost on the given dataset, using the given parameters.
   *
   * Default values are not used for `numModels`; instead, it is used to specify
   * the number of trees (models) to train.
   *
   * @param data Dataset to train on.
   * @param labels Labels for each point in the dataset.
   * @param numClasses The number of classes in the dataset.
   * @param numModels Number of boosting rounds.
   */
  void Train(const MatType& data,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const size_t numModels);

  /**
   * Train XGBoost on the given dataset, using the given parameters.
   * The last parameters are the hyperparameters to use for the trees;
   * these are all the arguments to `XGBTree::Train()` after `numClasses`
   * and `weights`.
   *
   * Default values are not used for `numModels`; instead, it is used to specify
   * the number of trees (models) to train during XGBoost.
   *
   * @param data Dataset to train on.
   * @param labels Labels for each point in the dataset.
   * @param numClasses The number of classes in the dataset.
   * @param numModels Number of boosting rounds.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   */
  void Train(const MatType& data,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const size_t numModels,
             const size_t minimumLeafSize,
             const double minimumGainSplit,
             const size_t maximumDepth);

  /**
   * Classify the given test point.
   *
   * @param point Test point.
   */
  template<typename VecType>
  size_t Classify(const VecType& point);

  /**
   * Classify the given test point.
   *
   * @param point Test point.
   * @param prediction Will be filled with the predicted class of `point`.
   */
  template<typename VecType>
  void Classify(const VecType& point,
                size_t& prediction,
                VecType& probabilities);

  /**
   * Classify the given test points.
   *
   * @param test Testing data.
   * @param predictedLabels Vector in which the predicted labels of the test
   *      set will be stored.
   */
  void Classify(const MatType& test,
                arma::Row<size_t>& predictedLabels,
                MatType& probabilities);

  /**
   * Serialize the XGBoost model.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const uint32_t version);

 private:

  /**
   * Internal utility training function.  
   */
  void TrainInternal(const MatType& data,
                     const arma::Row<size_t>& labels,
                     const size_t numClasses,
                     const size_t numModels = 10,
                     const size_t minimumLeafSize = 10,
                     const double minimumGainSplit = 1e-7,
                     const size_t maximumDepth = 2);

  //! The number of classes in the model.
  size_t numClasses;
  //! The number of trees in the model.
  size_t numModels;

  //! The vector of trees.
  std::vector<XGBTree*> trees;
  //! The weights corresponding to each weak learner.
  std::vector<ElemType> alpha;
}; 

}

CEREAL_TEMPLATE_CLASS_VERSION((typename MatType),
  (mlpack::XGBoost<MatType>), (1));

// Include implementation.
#include "xgboost_impl.hpp"

#endif