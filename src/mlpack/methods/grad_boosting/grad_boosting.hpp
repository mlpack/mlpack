/**
 * @file methods/grad_boosting/grad_boosting.hpp
 * @author Abhimanyu Dayal
 *
 * Gradient Boosting class. Gradient Boosting uses weak learners (primarily 
 * decision stumps), and trains them sequentially, such that future learners are 
 * trained to detect the errors (or gradients) of previous learners. 
 * The results obtained from all of these learners are subsequently 
 * aggregated to give a final result.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

// Include guard - Used to prevent including double copies of the same code
#ifndef MLPACK_METHODS_GRADBOOSTING_GRADBOOSTING_HPP
#define MLPACK_METHODS_GRADBOOSTING_GRADBOOSTING_HPP

// Importing base components required to write mlpack methods.
#include <mlpack/core.hpp>

// Only using decision trees for now, therefore only including decision tree functionalities.
// #include <mlpack/methods/decision_tree/decision_tree.hpp>
#include "../decision_tree.hpp"

// Defined DecisionTree with template as WeakLearnerType to avoid repetition
typedef mlpack::DecisionTree<mlpack::GiniGain,
                    mlpack::BestBinaryNumericSplit,
                    mlpack::AllCategoricalSplit,
                    mlpack::AllDimensionSelect,
                    false> WeakLearnerType;

// Written in mlpack namespace.
namespace mlpack {
/**
 * The Gradient Boosting class. Gradient Boosting is a boosting algorithm, meaning that it
 * combines an ensemble of weak learners to produce a strong learner.
 * 
 * Gradient Boosting is generally implemented using Decision Trees, or more specifically 
 * Decision Stumps i.e. weak learner Decision Trees with low depth.
 * 
 * @tparam MatType Data matrix type (i.e. arma::mat or arma::sp_mat).
 */


template<typename MatType = arma::mat>
class GradBoosting 
{
 public: 

  typedef typename MatType::elem_type ElemType;

  /**
   * Constructor for creating GradBoosting without training. 
   * Be sure to call Train() before calling Classify()
   */
  GradBoosting();

  /**
   * Constructor in case weak learners aren't defined.
   *
   * @param data Input data
   * @param labels Corresponding labels
   * @param numClasses Number of classes
   * @param numModels Number of weak learners
   */
  GradBoosting(const MatType& data,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const size_t numModels);

  /**
   * Constructor for a GradBoosting model. Any extra parameters are used as
   * hyperparameters for the weak learner. These should be the last arguments
   * to the weak learner's constructor or `Train()` function (i.e. anything
   * after `numClasses` or `weights`).
   *
   * @param data Input data.
   * @param labels Corresponding labels.
   * @param numClasses The number of classes.
   * @param numModels Number of weak learners.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   */
  GradBoosting(const MatType& data,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const size_t numModels = 10,
               const size_t minimumLeafSize = 10,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 2);


  //! Get the number of classes this model is trained on.
  size_t NumClasses() const { return numClasses; }

  //! Get the number of weak learners .
  size_t NumModels() const { return numModels; }

  //! Get the given weak learner.
  const WeakLearnerType& WeakLearner(const size_t i) const { return weakLearners[i]; }

  //! Modify the given weak learner (be careful!).
  WeakLearnerType& WeakLearner(const size_t i) { return weakLearners[i]; }

  /**
   * Train Gradient Boosting on the given dataset, using the given parameters.
   *
   * Default values are not used for `numModels`; instead, it is used to specify
   * the number of weak learners (models) to train.
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
   * Train Gradient Boosting on the given dataset, using the given parameters.
   * The last parameters are the hyperparameters to use for the weak learners;
   * these are all the arguments to `WeakLearnerType::Train()` after `numClasses`
   * and `weights`.
   *
   * Default values are not used for `numModels`; instead, it is used to specify
   * the number of weak learners (models) to train during gradient boosting.
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
             const size_t numModels = 10,
             const size_t minimumLeafSize = 10,
             const double minimumGainSplit = 1e-7,
             const size_t maximumDepth = 2);

  /**
   * Classify the given test point.
   *
   * @param point Test point.
   */
  template<typename VecType>
  size_t Classify(const VecType& point);

  /**
   * Classify the given test point and compute class probabilities.
   *
   * @param point Test point.
   * @param prediction Will be filled with the predicted class of `point`.
   * @param probabilities Vector in which the probabilities of each classes are
   *      stored
   */
  template<typename VecType>
  void Classify(const VecType& point,
                size_t& prediction,
                arma::vec& probabilities);

  /**
   * Classify the given test points.
   *
   * @param test Testing data.
   * @param predictedLabels Vector in which the predicted labels of the test
   *      set will be stored.
   */
  void Classify(const MatType& test,
                arma::Row<size_t>& predictedLabels);

  /**
   * Classify the given test points.
   *
   * @param test Testing data.
   * @param predictedLabels Vector in which the predicted labels of the test
   *      set will be stored.
   * @param probabilities Vector in which the probabilities of each classes are
   *      stored
   */
  void Classify(const MatType& test,
                arma::Row<size_t>& predictedLabels,
                arma::Row<ElemType>& probabilities);


  /**
   * Serialize the GradBoosting model.
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
  //! The number of weak learners in the model.
  size_t numModels;

  //! The vector of weak learners.
  std::vector<WeakLearnerType> weakLearners;
  //! The weights corresponding to each weak learner.
  std::vector<ElemType> alpha;
}; 

}

CEREAL_TEMPLATE_CLASS_VERSION((typename MatType),
  (mlpack::GradBoosting<MatType>), (1));

// Include implementation.
#include "grad_boosting_impl.hpp"

#endif
