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

// Written in mlpack namespace.
namespace mlpack {

template<typename WeakLearnerType = XGBTree, typename MatType = arma::mat>
class XGBoost
{
  XGBoost() {/*Nothing to do*/}

  /**
   * Constructor for a XGBoost model. Any extra parameters are used as
   * hyperparameters for the weak learner. These should be the last arguments
   * to the weak learner's constructor or `Train()` function (i.e. anything
   * after `numClasses` or `weights`).
   *
   * @param data Input data.
   * @param labels Corresponding labels.
   * @param numClasses The number of classes.
   * @param numModels Number of weak learners.
   * @param weakLearnerParams... Any hyperparameters for the weak learner.
   */
  template<typename... WeakLearnerArgs>
  XGBoost(const MatType& data,
                const arma::Row<size_t>& labels,
                const size_t numClasses,
                const size_t numModels = 10,
                WeakLearnerArgs&&... weakLearnerArgs);

  /**
   * Constructor takes an already-initialized weak learner; all other
   * weak learners will learn with the same parameters as the given
   * weak learner.
   *
   * @param data Input data.
   * @param labels Corresponding labels.
   * @param numClasses The number of classes.
   * @param numModels Number of weak learners.
   * @param other Weak learner that has already been initialized.
   */
  XGBoost (const MatType& data,
                const arma::Row<size_t>& labels,
                const size_t numClasses,
                const size_t numModels,
                const WeakLearnerType& other);
  
  //! Get the number of classes this model is trained on.
  size_t NumClasses() const { return numClasses; }

  //! Get the number of weak learners .
  size_t NumModels() const { return numModels; }

  //! Get the given weak learner.
  const WeakLearnerType& WeakLearner(const size_t i) const { return weakLearners[i]; }

  //! Modify the given weak learner (be careful!).
  WeakLearnerType& WeakLearner(const size_t i) { return weakLearners[i]; }

  /**
   * Train XGBoost on the given dataset. This method takes an initialized
   * WeakLearnerType; the parameters for this weak learner will be used to train
   * each of the weak learners during XGBoost training. Note that this will
   * completely overwrite any model that has already been trained with this
   * object.
   *
   * Default values are not used for `numModels`; instead, it is used to specify
   * the number of weak learners (models) to train.
   *
   * @param data Dataset to train on.
   * @param labels Labels for each point in the dataset.
   * @param numClasses The number of classes.
   * @param numModels Number of weak learners (models) to train.
   * @param learner Learner to use for training.
   */
  void Train(const MatType& data,
              const arma::Row<size_t>& labels,
              const size_t numClasses,
              const size_t numModels,
              const WeakLearnerType learner);

  /**
   * Train XGBoost on the given dataset, using the given parameters.
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
   * Train XGBoost on the given dataset, using the given parameters.
   * The last parameters are the hyperparameters to use for the weak learners;
   * these are all the arguments to `WeakLearnerType::Train()` after `numClasses`
   * and `weights`.
   *
   * Default values are not used for `numModels`; instead, it is used to specify
   * the number of weak learners (models) to train.
   *
   * @param data Dataset to train on.
   * @param labels Labels for each point in the dataset.
   * @param numClasses The number of classes in the dataset.
   * @param numModels Number of boosting rounds.
   * @param weakLearnerArgs Hyperparameters to use for each weak learner.
   */
  template<typename... WeakLearnerArgs>
  void Train(const MatType& data,
              const arma::Row<size_t>& labels,
              const size_t numClasses,
              const size_t numModels,
              WeakLearnerArgs&&... weakLearnerArgs);

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
                arma::Row<ElemType>& probabilities);

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
   * Serialize the XGBoost model.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const uint32_t version);


  private:

  template<bool UseExistingWeakLearner, typename... WeakLearnerArgs>
  void TrainInternal(const MatType& data,
                const arma::Row<size_t>& labels,
                const size_t numModels,
                const size_t numClasses,
                const WeakLearnerType& wl,
                WeakLearnerArgs&&... weakLearnerArgs);
};

}

// Include implementation.
#include "xgboost_impl.hpp"


#endif