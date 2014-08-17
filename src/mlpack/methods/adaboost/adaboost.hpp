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
 */
#ifndef __MLPACK_METHODS_ADABOOST_ADABOOST_HPP
#define __MLPACK_METHODS_ADABOOST_ADABOOST_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/perceptron/perceptron.hpp>
#include <mlpack/methods/decision_stump/decision_stump.hpp>

namespace mlpack {
namespace adaboost {

template<typename MatType = arma::mat,
         typename WeakLearner = mlpack::perceptron::Perceptron<> >
class AdaBoost
{
 public:
  /**
   * Constructor. Currently runs the AdaBoost.mh algorithm.
   *
   * @param data Input data.
   * @param labels Corresponding labels.
   * @param iterations Number of boosting rounds.
   * @param tol The tolerance for change in values of rt.
   * @param other Weak Learner, which has been initialized already.
   */
  AdaBoost(const MatType& data,
           const arma::Row<size_t>& labels,
           const int iterations,
           const double tol,
           const WeakLearner& other);

  // Stores the final classification of the Labels.
  arma::Row<size_t> finalHypothesis;

  // Return the value of ztProduct
  double GetztProduct() { return ztProduct; }

  // The tolerance for change in rt and when to stop.
  double tolerance;

  void Classify(const MatType& test, arma::Row<size_t>& predictedLabels);

private:
  /**
   *  This function helps in building the Weight Distribution matrix
   *  which is updated during every iteration. It calculates the
   *  "difficulty" in classifying a point by adding the weights for all
   *  instances, using D.
   *
   *  @param D The 2 Dimensional weight matrix from which the weights are
   *            to be calculated.
   *  @param weights The output weight vector.
   */
  void BuildWeightMatrix(const arma::mat& D, arma::rowvec& weights);

  size_t numClasses;
  
  std::vector<WeakLearner> wl;
  std::vector<double> alpha;

  // To check for the bound for the hammingLoss.
  double ztProduct;
  
}; // class AdaBoost

} // namespace adaboost
} // namespace mlpack

#include "adaboost_impl.hpp"

#endif
