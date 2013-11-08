/**
 * @file logistic_regression.hpp
 * @author Sumedh Ghaisas
 *
 * The LogisticRegression class, which implements logistic regression.
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
  LogisticRegression(arma::mat& predictors,
                     arma::vec& responses,
                     const double lambda = 0);

  LogisticRegression(arma::mat& predictors,
                     arma::vec& responses,
                     const arma::mat& initialPoint,
                     const double lambda = 0);

  //! Return the parameters (the b vector).
  const arma::vec& Parameters() const { return parameters; }
  //! Modify the parameters (the b vector).
  arma::vec& Parameters() { return parameters; }

  //! Return the lambda value
  const double& Lambda() const { return lambda; }
  //! Modify the lambda value
  double& Lambda() { return lambda; }

  double LearnModel();

  //predict functions
  void Predict(arma::mat& predictors,
               arma::vec& responses,
               const double decisionBoundary = 0.5);

  double ComputeAccuracy(arma::mat& predictors,
                         const arma::vec& responses,
                         const double decisionBoundary = 0.5);

  double ComputeError(arma::mat& predictors,const arma::vec& responses);

 private:
  arma::vec parameters;
  arma::mat& predictors;
  arma::vec& responses;
  LogisticRegressionFunction errorFunction;
  OptimizerType<LogisticRegressionFunction> optimizer;
  double lambda;
};

}; // namespace regression
}; // namespace mlpack

// Include implementation.
#include "logistic_regression_impl.hpp"

#endif // __MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_HPP
