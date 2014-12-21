/**
 * @file logistic_regression.hpp
 * @author Sumedh Ghaisas
 *
 * The LogisticRegression class, which implements logistic regression.  This
 * implements supports L2-regularization.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
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

  /**
   * Construct the LogisticRegression class with the given labeled training
   * data.  This will train the model.  This overload takes an already
   * instantiated optimizer (which holds the LogisticRegressionFunction error
   * function, which must also be instantiated), so that the optimizer can be
   * configured before the training is run by this constructor.  The predictors
   * and responses and initial point are all taken from the error function
   * contained in the optimizer.
   *
   * @param optimizer Instantiated optimizer with instantiated error function.
   */
  LogisticRegression(OptimizerType<LogisticRegressionFunction>& optimizer);

  /**
   * Construct a logistic regression model from the given parameters, without
   * performing any training.  The lambda parameter is used for the
   * ComputeAccuracy() and ComputeError() functions; this constructor does not
   * train the model itself.
   *
   * @param parameters Parameters making up the model.
   * @param lambda L2-regularization penalty parameter.
   */
  LogisticRegression(const arma::vec& parameters, const double lambda = 0);

  //! Return the parameters (the b vector).
  const arma::vec& Parameters() const { return parameters; }
  //! Modify the parameters (the b vector).
  arma::vec& Parameters() { return parameters; }

  //! Return the lambda value for L2-regularization.
  const double& Lambda() const { return lambda; }
  //! Modify the lambda value for L2-regularization.
  double& Lambda() { return lambda; }

  /**
   * Predict the responses to a given set of predictors.  The responses will be
   * either 0 or 1.  Optionally, specify the decision boundary; logistic
   * regression returns a value between 0 and 1.  If the value is greater than
   * the decision boundary, the response is taken to be 1; otherwise, it is 0.
   * By default the decision boundary is 0.5.
   *
   * @param predictors Input predictors.
   * @param responses Vector to put output predictions of responses into.
   * @param decisionBoundary Decision boundary (default 0.5).
   */
  void Predict(const arma::mat& predictors,
               arma::vec& responses,
               const double decisionBoundary = 0.5) const;

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
  double ComputeAccuracy(const arma::mat& predictors,
                         const arma::vec& responses,
                         const double decisionBoundary = 0.5) const;

  /**
   * Compute the error of the model.  This returns the negative objective
   * function of the logistic regression log-likelihood function.  For the model
   * to be optimal, the negative log-likelihood function should be minimized.
   *
   * @param predictors Input predictors.
   * @param responses Vector of responses.
   */
  double ComputeError(const arma::mat& predictors,
                      const arma::vec& responses) const;

  // Returns a string representation of this object. 
  std::string ToString() const;

 private:
  //! Vector of trained parameters.
  arma::vec parameters;
  //! L2-regularization penalty parameter.
  double lambda;
};

}; // namespace regression
}; // namespace mlpack

// Include implementation.
#include "logistic_regression_impl.hpp"

#endif // __MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_HPP
