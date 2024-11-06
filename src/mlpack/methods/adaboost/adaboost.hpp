/**
 * @file methods/adaboost/adaboost.hpp
 * @author Udit Saxena
 *
 * The AdaBoost class.  AdaBoost is a boosting algorithm, meaning that it
 * combines an ensemble of weak learners to produce a strong learner.  For more
 * information on AdaBoost, see the following paper:
 *
 * @code
 * @article{schapire1999improved,
 *   author = {Schapire, Robert E. and Singer, Yoram},
 *   title = {Improved Boosting Algorithms Using Confidence-rated Predictions},
 *   journal = {Machine Learning},
 *   volume = {37},
 *   number = {3},
 *   month = dec,
 *   year = {1999},
 *   issn = {0885-6125},
 *   pages = {297--336},
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ADABOOST_ADABOOST_HPP
#define MLPACK_METHODS_ADABOOST_ADABOOST_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/perceptron/perceptron.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>

namespace mlpack {

/**
 * The AdaBoost class.  AdaBoost is a boosting algorithm, meaning that it
 * combines an ensemble of weak learners to produce a strong learner.  For more
 * information on AdaBoost, see the following paper:
 *
 * @code
 * @article{schapire1999improved,
 *   author = {Schapire, Robert E. and Singer, Yoram},
 *   title = {Improved Boosting Algorithms Using Confidence-rated Predictions},
 *   journal = {Machine Learning},
 *   volume = {37},
 *   number = {3},
 *   month = dec,
 *   year = {1999},
 *   issn = {0885-6125},
 *   pages = {297--336},
 * }
 * @endcode
 *
 * This class is general, and can be used with any type of weak learner, so long
 * as the learner implements the following functions:
 *
 * @code
 * // A boosting constructor, which learns using the training parameters of the
 * // given other WeakLearner, but uses the given instance weights for training.
 * WeakLearner(WeakLearner& other,
 *             const MatType& data,
 *             const arma::Row<size_t>& labels,
 *             const arma::rowvec& weights);
 *
 * // Given the test points, classify them and output predictions into
 * // predictedLabels.
 * void Classify(const MatType& data, arma::Row<size_t>& predictedLabels);
 * @endcode
 *
 * For more information on and examples of weak learners, see Perceptron<> and
 * ID3DecisionStump.
 *
 * @tparam MatType Data matrix type (i.e. arma::mat or arma::sp_mat).
 * @tparam WeakLearnerType Type of weak learner to use.
 */
template<typename WeakLearnerType = Perceptron<>,
         typename MatType = arma::mat>
class AdaBoost
{
 public:
  using ElemType = typename MatType::elem_type;

  /**
   * Create the AdaBoost object without training.  Be sure to call Train()
   * before calling Classify()!
   */
  AdaBoost(const ElemType tolerance = 1e-6);

  /**
   * Construct an AdaBoost model.  Any extra parameters are used as
   * hyperparameters for the weak learner.  These should be the last arguments
   * to the weak learner's constructor or `Train()` function (i.e. anything
   * after `numClasses` or `weights`).
   *
   * @param data Input data.
   * @param labels Corresponding labels.
   * @param numClasses The number of classes.
   * @param maxIterations Number of boosting rounds.
   * @param tolerance The tolerance for change in values of rt.
   * @param weakLearnerParams... Any hyperparameters for the weak learner.
   */
  template<typename... WeakLearnerArgs>
  AdaBoost(const MatType& data,
           const arma::Row<size_t>& labels,
           const size_t numClasses,
           const size_t maxIterations = 100,
           const ElemType tolerance = 1e-6,
           WeakLearnerArgs&&... weakLearnerArgs);

  /**
   * Constructor.  This runs the AdaBoost.MH algorithm to provide a trained
   * boosting model.  This constructor takes an already-initialized weak
   * learner; all other weak learners will learn with the same parameters as the
   * given weak learner.
   *
   * @param data Input data.
   * @param labels Corresponding labels.
   * @param numClasses The number of classes.
   * @param maxIterations Number of boosting rounds.
   * @param tolerance The tolerance for change in values of rt.
   * @param other Weak learner that has already been initialized.
   */
  template<typename WeakLearnerInType>
  [[deprecated("Will be removed in mlpack 5.0.0, use other constructors")]]
  AdaBoost(const MatType& data,
           const arma::Row<size_t>& labels,
           const size_t numClasses,
           const WeakLearnerInType& other,
           const size_t maxIterations = 100,
           const ElemType tolerance = 1e-6,
           const std::enable_if_t<
              std::is_same_v<WeakLearnerType, WeakLearnerInType>>* = 0);

  //! Get the maximum number of weak learners allowed in the model.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of weak learners allowed in the model.
  size_t& MaxIterations() { return maxIterations; }

  //! Get the tolerance for stopping the optimization during training.
  ElemType Tolerance() const { return tolerance; }
  //! Modify the tolerance for stopping the optimization during training.
  ElemType& Tolerance() { return tolerance; }

  //! Get the number of classes this model is trained on.
  size_t NumClasses() const { return numClasses; }

  //! Get the number of weak learners in the model.
  size_t WeakLearners() const { return alpha.size(); }

  //! Get the weights for the given weak learner.
  ElemType Alpha(const size_t i) const { return alpha[i]; }
  //! Modify the weight for the given weak learner (be careful!).
  ElemType& Alpha(const size_t i) { return alpha[i]; }

  //! Get the given weak learner.
  const WeakLearnerType& WeakLearner(const size_t i) const { return wl[i]; }
  //! Modify the given weak learner (be careful!).
  WeakLearnerType& WeakLearner(const size_t i) { return wl[i]; }

  /**
   * Train AdaBoost on the given dataset.  This method takes an initialized
   * WeakLearnerType; the parameters for this weak learner will be used to train
   * each of the weak learners during AdaBoost training.  Note that this will
   * completely overwrite any model that has already been trained with this
   * object.
   *
   * Default values are not used for `maxIterations` and `tolerance`; instead,
   * multiple overloads are allowed; this is because we want to use the existing
   * setting internal to the class, if one is not specified.
   *
   * @param data Dataset to train on.
   * @param labels Labels for each point in the dataset.
   * @param numClasses The number of classes.
   * @param learner Learner to use for training.
   * @param maxIterations Number of boosting rounds.
   * @param tolerance The tolerance for change in values of rt.
   * @return The upper bound for training error.
   */
  template<typename WeakLearnerInType>
  [[deprecated("Will be removed in mlpack 5.0.0, use other Train() variants")]]
  ElemType Train(
      const MatType& data,
      const arma::Row<size_t>& labels,
      const size_t numClasses,
      const WeakLearnerInType& learner,
      const std::optional<size_t> maxIterations = std::nullopt,
      const std::optional<double> tolerance = std::nullopt,
      // Necessary to distinguish from other overloads.
      const std::enable_if_t<
          std::is_same_v<WeakLearnerType, WeakLearnerInType>>* = 0);

  /**
   * Train AdaBoost on the given dataset, using the given parameters.  The last
   * parameters are the hyperparameters to use for the weak learners; these are
   * all the arguments to `WeakLearnerType::Train()` after `numClasses` and
   * `weights`.
   *
   * Default values are not used for `maxIterations` and `tolerance`; instead,
   * multiple overloads are allowed; this is because we want to use the existing
   * setting internal to the class, if one is not specified.
   *
   * @param data Dataset to train on.
   * @param labels Labels for each point in the dataset.
   * @param numClasses The number of classes in the dataset.
   * @param maxIterations Number of boosting rounds.
   * @param tolerance The tolerance for change in values of rt.
   * @param weakLearnerArgs Hyperparameters to use for each weak learner.
   * @return The upper bound for training error.
   */
  template<typename... WeakLearnerArgs>
  ElemType Train(const MatType& data,
                 const arma::Row<size_t>& labels,
                 const size_t numClasses,
                 const std::optional<size_t> maxIterations = std::nullopt,
                 const std::optional<double> tolerance = std::nullopt,
                 WeakLearnerArgs&&... weakLearnerArgs);

  /**
   * Classify the given test point.
   *
   * @param point Test point.
   */
  template<typename VecType>
  size_t Classify(const VecType& point) const;

  /**
   * Classify the given test point and compute class probabilities.
   *
   * @param point Test point.
   * @param prediction Will be filled with the predicted class of `point`.
   * @param probabilities Will be filled with the class probabilities.
   */
  template<typename VecType>
  void Classify(const VecType& point,
                size_t& prediction,
                arma::Row<ElemType>& probabilities) const;

  /**
   * Classify the given test points.
   *
   * @param test Testing data.
   * @param predictedLabels Vector in which the predicted labels of the test
   *      set will be stored.
   */
  void Classify(const MatType& test,
                arma::Row<size_t>& predictedLabels) const;

  /**
   * Classify the given test points.
   *
   * @param test Testing data.
   * @param predictedLabels Vector in which the predicted labels of the test
   *      set will be stored.
   * @param probabilities matrix to store the predicted class probabilities for
   *      each point in the test set.
   */
  void Classify(const MatType& test,
                arma::Row<size_t>& predictedLabels,
                arma::Mat<ElemType>& probabilities) const;

  /**
   * Serialize the AdaBoost model.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /**
   * Internal utility training function.  `wl` is not used if
   * `UseExistingWeakLearner` is false.  `weakLearnerArgs` are not used if
   * `UseExistingWeakLearner` is true.
   */
  template<bool UseExistingWeakLearner, typename... WeakLearnerArgs>
  ElemType TrainInternal(const MatType& data,
                         const arma::Row<size_t>& labels,
                         const size_t numClasses,
                         const WeakLearnerType& wl,
                         WeakLearnerArgs&&... weakLearnerArgs);

  //! The number of classes in the model.
  size_t numClasses;
  //! The maximum number of weak learners allowed in the model.
  size_t maxIterations;
  //! The tolerance for change in rt and when to stop.
  ElemType tolerance;

  //! The vector of weak learners.
  std::vector<WeakLearnerType> wl;
  //! The weights corresponding to each weak learner.
  std::vector<ElemType> alpha;
}; // class AdaBoost

} // namespace mlpack

CEREAL_TEMPLATE_CLASS_VERSION((typename WeakLearnerType, typename MatType),
    (mlpack::AdaBoost<WeakLearnerType, MatType>), (1));

// Include implementation.
#include "adaboost_impl.hpp"

#endif
