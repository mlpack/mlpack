/**
 * @file adaboost.hpp
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
#include <mlpack/methods/decision_stump/decision_stump.hpp>

namespace mlpack {
namespace adaboost {

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
 * For more information on and examples of weak learners, see
 * perceptron::Perceptron<> and decision_stump::DecisionStump<>.
 *
 * @tparam MatType Data matrix type (i.e. arma::mat or arma::sp_mat).
 * @tparam WeakLearnerType Type of weak learner to use.
 */
template<typename WeakLearnerType = mlpack::perceptron::Perceptron<>,
         typename MatType = arma::mat>
class AdaBoost
{
 public:
  /**
   * Constructor.  This runs the AdaBoost.MH algorithm to provide a trained
   * boosting model.  This constructor takes an already-initialized weak
   * learner; all other weak learners will learn with the same parameters as the
   * given weak learner.
   *
   * @param data Input data.
   * @param labels Corresponding labels.
   * @param iterations Number of boosting rounds.
   * @param tol The tolerance for change in values of rt.
   * @param other Weak learner that has already been initialized.
   */
  AdaBoost(const MatType& data,
           const arma::Row<size_t>& labels,
           const WeakLearnerType& other,
           const size_t iterations = 100,
           const double tolerance = 1e-6);

  /**
   * Create the AdaBoost object without training.  Be sure to call Train()
   * before calling Classify()!
   */
  AdaBoost(const double tolerance = 1e-6);

  // Return the value of ztProduct.
  double ZtProduct() { return ztProduct; }

  //! Get the tolerance for stopping the optimization during training.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for stopping the optimization during training.
  double& Tolerance() { return tolerance; }

  //! Get the number of classes this model is trained on.
  size_t Classes() const { return classes; }

  //! Get the number of weak learners in the model.
  size_t WeakLearners() const { return alpha.size(); }

  //! Get the weights for the given weak learner.
  double Alpha(const size_t i) const { return alpha[i]; }
  //! Modify the weight for the given weak learner (be careful!).
  double& Alpha(const size_t i) { return alpha[i]; }

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
   * @param data Dataset to train on.
   * @param labels Labels for each point in the dataset.
   * @param learner Learner to use for training.
   */
  void Train(const MatType& data,
             const arma::Row<size_t>& labels,
             const WeakLearnerType& learner,
             const size_t iterations = 100,
             const double tolerance = 1e-6);

  /**
   * Classify the given test points.
   *
   * @param test Testing data.
   * @param predictedLabels Vector in which to the predicted labels of the test
   *      set will be stored.
   */
  void Classify(const MatType& test, arma::Row<size_t>& predictedLabels);

  /**
   * Serialize the AdaBoost model.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

private:
  //! The number of classes in the model.
  size_t classes;
  // The tolerance for change in rt and when to stop.
  double tolerance;

  //! The vector of weak learners.
  std::vector<WeakLearnerType> wl;
  //! The weights corresponding to each weak learner.
  std::vector<double> alpha;

  //! To check for the bound for the Hamming loss.
  double ztProduct;

}; // class AdaBoost

} // namespace adaboost
} // namespace mlpack

#include "adaboost_impl.hpp"

#endif
