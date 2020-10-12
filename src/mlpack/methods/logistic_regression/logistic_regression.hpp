/**
 * @file methods/logistic_regression/logistic_regression.hpp
 * @author Sumedh Ghaisas
 * @author Arun Reddy
 *
 * The LogisticRegression class, which implements logistic regression.  This
 * implements supports L2-regularization.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_HPP
#define MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_HPP

#include <mlpack/prereqs.hpp>
#include <ensmallen.hpp>

#include "logistic_regression_function.hpp"

namespace mlpack {
namespace regression {

/**
 * The LogisticRegression class implements an L2-regularized logistic regression
 * model, and supports training with multiple optimizers and classification.
 * The class supports different observation types via the MatType template
 * parameter; for instance, logistic regression can be performed on sparse
 * datasets by specifying arma::sp_mat as the MatType parameter.
 *
 * LogisticRegression can be used for general classification tasks, but the
 * class is restricted to support only two classes.  For multiclass logistic
 * regression, see mlpack::regression::SoftmaxRegression.
 *
 * @tparam MatType Type of data matrix.
 */
template<typename MatType = arma::mat>
class LogisticRegression
{
 public:
  /**
   * Construct the LogisticRegression class with the given labeled training
   * data.  This will train the model.  Optionally, specify lambda, which is the
   * penalty parameter for L2-regularization.  If not specified, it is set to 0,
   * which results in standard (unregularized) logistic regression.
   *
   * It is not possible to set a custom optimizer with this constructor.  Either
   * use a constructor that does not train and call Train() with a custom
   * optimizer type, or use the constructor that takes an instantiated
   * optimizer.  (This unfortunate situation is a language restriction of C++.)
   *
   * @param predictors Input training variables.
   * @param responses Outputs resulting from input training variables.
   * @param lambda L2-regularization parameter.
   */
  LogisticRegression(const MatType& predictors,
                     const arma::Row<size_t>& responses,
                     const double lambda = 0);

  /**
   * Construct the LogisticRegression class with the given labeled training
   * data.  This will train the model.  Optionally, specify lambda, which is the
   * penalty parameter for L2-regularization.  If not specified, it is set to 0,
   * which results in standard (unregularized) logistic regression.
   *
   * It is not possible to set a custom optimizer with this constructor.  Either
   * use a constructor that does not train and call Train() with a custom
   * optimizer type, or use the constructor that takes an instantiated
   * optimizer.  (This unfortunate situation is a language restriction of C++.)
   *
   * @param predictors Input training variables.
   * @param responses Outputs results from input training variables.
   * @param initialPoint Initial model to train with.
   * @param lambda L2-regularization parameter.
   */
  LogisticRegression(const MatType& predictors,
                     const arma::Row<size_t>& responses,
                     const arma::rowvec& initialPoint,
                     const double lambda = 0);

  /**
   * Construct the LogisticRegression class without performing any training.
   * The dimensionality of the data (which will be used to set the size of the
   * parameters vector) must be specified, and all of the parameters in the
   * model will be set to 0.  Note that the dimensionality may be changed later
   * by directly modifying the parameters vector (using Parameters()).
   *
   * @param dimensionality Dimensionality of the data.
   * @param lambda L2-regularization parameter.
   */
  LogisticRegression(const size_t dimensionality = 0,
                     const double lambda = 0);

  /**
   * Construct the LogisticRegression class with the given labeled training
   * data.  This will train the model.  This overload takes an already
   * instantiated optimizer (which holds the LogisticRegressionFunction error
   * function, which must also be instantiated), so that the optimizer can be
   * configured before the training is run by this constructor.  The update
   * policy of the optimizer can be set through the policy argument.  The
   * predictors and responses and initial point are all taken from the error
   * function contained in the optimizer.
   *
   * @param predictors Input training variables.
   * @param responses Outputs results from input training variables.
   * @param optimizer Instantiated optimizer with instantiated error function.
   * @param lambda L2-regularization parameter.
   */
  template<typename OptimizerType>
  LogisticRegression(const MatType& predictors,
                     const arma::Row<size_t>& responses,
                     OptimizerType& optimizer,
                     const double lambda);

  /**
   * Train the LogisticRegression model on the given input data.  By default,
   * the L-BFGS optimization algorithm is used, but others can be specified
   * (such as ens::SGD).
   *
   * This will use the existing model parameters as a starting point for the
   * optimization.  If this is not what you want, then you should access the
   * parameters vector directly with Parameters() and modify it as desired.
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @tparam CallbackTypes Types of Callback Functions.
   * @param predictors Input training variables.
   * @param responses Outputs results from input training variables.
   * @param callbacks Callback function for ensmallen optimizer `OptimizerType`.
   *      See https://www.ensmallen.org/docs.html#callback-documentation.
   * @return The final objective of the trained model (NaN or Inf on error)
   */
  template<typename OptimizerType = ens::L_BFGS, typename... CallbackTypes>
  double Train(const MatType& predictors,
               const arma::Row<size_t>& responses,
               CallbackTypes&&... callbacks);

  /**
   * Train the LogisticRegression model with the given instantiated optimizer.
   * Using this overload allows configuring the instantiated optimizer before
   * training is performed.
   *
   * This will use the existing model parameters as a starting point for the
   * optimization.  If this is not what you want, then you should access the
   * parameters vector directly with Parameters() and modify it as desired.
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @tparam CallbackTypes Types of Callback Functions.
   * @param predictors Input training variables.
   * @param responses Outputs results from input training variables.
   * @param optimizer Instantiated optimizer with instantiated error function.
   * @param callbacks Callback function for ensmallen optimizer `OptimizerType`.
   *      See https://www.ensmallen.org/docs.html#callback-documentation.
   * @return The final objective of the trained model (NaN or Inf on error)
   */
  template<typename OptimizerType, typename... CallbackTypes>
  double Train(const MatType& predictors,
               const arma::Row<size_t>& responses,
               OptimizerType& optimizer,
               CallbackTypes&&... callbacks);

  //! Return the parameters (the b vector).
  const arma::rowvec& Parameters() const { return parameters; }
  //! Modify the parameters (the b vector).
  arma::rowvec& Parameters() { return parameters; }

  //! Return the lambda value for L2-regularization.
  const double& Lambda() const { return lambda; }
  //! Modify the lambda value for L2-regularization.
  double& Lambda() { return lambda; }

  /**
   * Classify the given point.  The predicted label is returned.  Optionally,
   * specify the decision boundary; logistic regression returns a value between
   * 0 and 1.  If the value is greater than the decision boundary, the response
   * is taken to be 1; otherwise, it is 0.  By default the decision boundary is
   * 0.5.
   *
   * @param point Point to classify.
   * @param decisionBoundary Decision boundary (default 0.5).
   * @return Predicted label of point.
   */
  template<typename VecType>
  size_t Classify(const VecType& point,
                  const double decisionBoundary = 0.5) const;

  /**
   * Classify the given points, returning the predicted labels for each point.
   * Optionally, specify the decision boundary; logistic regression returns a
   * value between 0 and 1.  If the value is greater than the decision boundary,
   * the response is taken to be 1; otherwise, it is 0.  By default the decision
   * boundary is 0.5.
   *
   * @param dataset Set of points to classify.
   * @param labels Predicted labels for each point.
   * @param decisionBoundary Decision boundary (default 0.5).
   */
  void Classify(const MatType& dataset,
                arma::Row<size_t>& labels,
                const double decisionBoundary = 0.5) const;

  /**
   * Classify the given points, returning class probabilities for each point.
   *
   * @param dataset Set of points to classify.
   * @param probabilities Class probabilities for each point (output).
   */
  void Classify(const MatType& dataset,
                arma::mat& probabilities) const;

  /**
   * Compute the accuracy of the model on the given predictors and responses,
   * optionally using the given decision boundary.  The responses should be
   * either 0 or 1.  Logistic regression returns a value between 0 and 1.  If
   * the value is greater than the decision boundary, the response is taken to
   * be 1; otherwise, it is 0.  By default, the decision boundary is 0.5.
   *
   * The accuracy is returned as a percentage, between 0 and 100.
   *
   * @param predictors Input predictors.
   * @param responses Vector of responses.
   * @param decisionBoundary Decision boundary (default 0.5).
   * @return Percentage of responses that are predicted correctly.
   */
  double ComputeAccuracy(const MatType& predictors,
                         const arma::Row<size_t>& responses,
                         const double decisionBoundary = 0.5) const;

  /**
   * Compute the error of the model.  This returns the negative objective
   * function of the logistic regression log-likelihood function.  For the model
   * to be optimal, the negative log-likelihood function should be minimized.
   *
   * @param predictors Input predictors.
   * @param responses Vector of responses.
   */
  double ComputeError(const MatType& predictors,
                      const arma::Row<size_t>& responses) const;

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Vector of trained parameters (size: dimensionality plus one).
  arma::rowvec parameters;
  //! L2-regularization penalty parameter.
  double lambda;
};

} // namespace regression
} // namespace mlpack

// Include implementation.
#include "logistic_regression_impl.hpp"

#endif // MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_HPP
