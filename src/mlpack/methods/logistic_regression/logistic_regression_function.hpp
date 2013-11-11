/**
 * @file logistic_regression_function.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the logistic regression function, which is meant to be
 * optimized by a separate optimizer class that takes LogisticRegressionFunction
 * as its FunctionType class.
 */
#ifndef __MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP
#define __MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace regression {

class LogisticRegressionFunction
{
 public:
  LogisticRegressionFunction(arma::mat& predictors,
                             arma::vec& responses,
                             const double lambda = 0);

  LogisticRegressionFunction(arma::mat& predictors,
                             arma::vec& responses,
                             const arma::mat& initialPoint,
                             const double lambda = 0);

  arma::vec getSigmoid(const arma::vec& values) const;

  //!Return the initial point
  const arma::mat& InitialPoint() const { return initialPoint; }
  //! Modify the initial point
  arma::mat& InitialPoint() { return initialPoint; }

  //!Return the lambda
  const double& Lambda() const { return lambda; }
  //! Modify the lambda
  double& Lambda() { return lambda; }

  //functions to optimize by l-bfgs
  double Evaluate(const arma::mat& values) const;

  void Gradient(const arma::mat& values, arma::mat& gradient);

  const arma::mat& GetInitialPoint() const { return initialPoint; }

  //functions to optimize by sgd
  double Evaluate(const arma::mat& values, const size_t i) const
  {
    return Evaluate(values);
  }
  void Gradient(const arma::mat& values,
                const size_t i,
                arma::mat& gradient)
  {
    Gradient(values,gradient);
  }

  size_t NumFunctions() { return 1; }

 private:
  arma::mat initialPoint;
  arma::mat& predictors;
  arma::vec& responses;
  double lambda;
};

}; // namespace regression
}; // namespace mlpack

#endif // __MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP
