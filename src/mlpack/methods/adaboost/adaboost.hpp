/**
 * @file adaboost.hpp
 * @author Udit Saxena
 *
 * AdaBoost header file
 */

#ifndef _MLPACK_METHODS_ADABOOST_ADABOOST_HPP
#define _MLPACK_METHODS_ADABOOST_ADABOOST_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/perceptron/perceptron.hpp>
#include <mlpack/methods/decision_stump/decision_stump.hpp>
 
namespace mlpack {
namespace adaboost {

template <typename MatType = arma::mat, typename WeakLearner = 
          mlpack::perceptron::Perceptron<> >
class Adaboost 
{
public:
  /**
   *  Constructor. Currently runs the Adaboost.mh algorithm
   *  
   *  @param data Input data
   *  @param labels Corresponding labels
   *  @param iterations Number of boosting rounds 
   *  @param tol The tolerance for change in values of rt.
   *  @param other Weak Learner, which has been initialized already
   */
  Adaboost(const MatType& data, const arma::Row<size_t>& labels,
           int iterations, double tol, const WeakLearner& other);

  /**
   *  This function helps in building a classification Matrix which is of 
   *  form: 
   *  -1 if l is not the correct label
   *  1 if l is the correct label
   *
   *  @param t The classification matrix to be built
   *  @param l The labels from which the classification matrix is to be built.
   */
  void buildClassificationMatrix(arma::mat& t, const arma::Row<size_t>& l);

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
  void buildWeightMatrix(const arma::mat& D, arma::rowvec& weights);

  // Stores the final classification of the Labels.
  arma::Row<size_t> finalHypothesis;

  // To check for the bound for the hammingLoss.
  double ztAccumulator;

  // The tolerance for change in rt and when to stop.
  double tolerance;
}; // class Adaboost

} // namespace adaboost
} // namespace mlpack

#include "adaboost_impl.hpp"

#endif