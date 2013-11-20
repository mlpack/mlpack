/**
 * @file logistic_regression.hpp
 * @author Sumedh Ghaisas
 *
 * The LogisticRegression class, which implements logistic regression.  This
 * implements supports L2-regularization.
 */
#ifndef __MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_HPP
#define __MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>

#include "logistic_regression_function.hpp"

namespace mlpack {
namespace regression {

template<
  template<typename> class OptimizerType = mlpack::optimization::L_BFGS
>
class LogisticRegression
{
 public:
  /**
   * Construct the LogisticRegression class with the given labeled training
   * data.  This will train the model.  Optionally, specify lambda, which is the
   * penalty parameter for L2-regularization.  If not specified, it is set to 0,
   * which results in standard (unregularized) logistic regression.
   *
   * @param predictors Input training variables.
   * @param responses Outputs resulting from input training variables.
   * @param lambda L2-regularization parameter.
   */
  LogisticRegression(const arma::mat& predictors,
                     const arma::vec& responses,
                     const double lambda = 0);

  /**
   * Construct the LogisticRegression class with the given labeled training
   * data.  This will train the model.  Optionally, specify lambda, which is the
   * penalty parameter for L2-regularization.  If not specified, it is set to 0,
   * which results in standard (unregularized) logistic regression.
   *
   * @param predictors Input training variables.
   * @param responses Outputs results from input training variables.
   * @param initialPoint Initial model to train with.
   * @param lambda L2-regularization parameter.
   */
  LogisticRegression(const arma::mat& predictors,
                     const arma::vec& responses,
                     const arma::mat& initialPoint,
                     const double lambda = 0);

  //! Return the parameters (the b vector).
  const arma::vec& Parameters() const { return parameters; }
  //! Modify the parameters (the b vector).
  arma::vec& Parameters() { return parameters; }

  //! Return the lambda value for L2-regularization.
  const double& Lambda() const { return lambda; }
  //! Modify the lambda value for L2-regularization.
  double& Lambda() { return lambda; }

  double LearnModel();

  //predict functions
  void Predict(arma::mat& predictors,
               arma::vec& responses,
               const double decisionBoundary = 0.5);

  double ComputeAccuracy(arma::mat& predictors,
                         const arma::vec& responses,
                         const double decisionBoundary = 0.5);

  double ComputeError(arma::mat& predictors, const arma::vec& responses);

 private:
  arma::vec parameters;
  const arma::mat& predictors;
  const arma::vec& responses;
  LogisticRegressionFunction errorFunction;
  OptimizerType<LogisticRegressionFunction> optimizer;
  double lambda;
};

}; // namespace regression
}; // namespace mlpack

// Include implementation.
#include "logistic_regression_impl.hpp"

#endif // __MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_HPP
