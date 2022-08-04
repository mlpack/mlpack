/**
 * @file methods/linear_regression/linear_regression.hpp
 * @author James Cline
 * @author Michael Fox
 *
 * Simple least-squares linear regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP
#define MLPACK_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/log.hpp>
#include <mlpack/core/util/size_checks.hpp>

namespace mlpack {
namespace regression /** Regression methods. */ {

/**
 * A simple linear regression algorithm using ordinary least squares.
 * Optionally, this class can perform ridge regression, if the lambda parameter
 * is set to a number greater than zero.
 */
template <template<class> class M, class Ele>
class LinearRegressionModel
{
 public:
  /**
   * Creates the model.
   *
   * @param predictors X, matrix of data points.
   * @param responses y, the measured data for each point in X.
   * @param lambda Regularization constant for ridge regression.
   * @param intercept Whether or not to include an intercept term.
   */
  LinearRegressionModel(const M<Ele>& predictors,
                   const arma::Row<Ele>& responses,
                   const double lambda = 0,
                   const bool intercept = true);

  /**
   * Creates the model.
   *
   * @param predictors X, matrix of data points.
   * @param responses Y, matrix of responses.
   * @param lambda Regularization constant for ridge regression.
   * @param intercept Whether or not to include an intercept term.
   */
  LinearRegressionModel(const M<Ele>& predictors,
                   const arma::Mat<Ele>& responses,
                   const double lambda = 0,
                   const bool intercept = true);

  /**
   * Creates the model with weighted learning.
   *
   * @param predictors X, matrix of data points.
   * @param responses y, the measured data for each point in X.
   * @param weights Observation weights (for boosting).
   * @param lambda Regularization constant for ridge regression.
   * @param intercept Whether or not to include an intercept term.
   */
  LinearRegressionModel(const M<Ele>& predictors,
                   const arma::Row<Ele>& responses,
                   const arma::Row<Ele>& weights,
                   const double lambda = 0,
                   const bool intercept = true);

  /**
   * Creates the model with weighted learning.
   *
   * @param predictors X, matrix of data points.
   * @param responses Y, matrix of responses.
   * @param weights Observation weights (for boosting).
   * @param lambda Regularization constant for ridge regression.
   * @param intercept Whether or not to include an intercept term.
   */
  LinearRegressionModel(const M<Ele>& predictors,
                   const arma::Mat<Ele>& responses,
                   const arma::Row<Ele>& weights,
                   const double lambda = 0,
                   const bool intercept = true);


  /**
   * Empty constructor.  This gives a non-working model, so make sure Train() is
   * called (or make sure the model parameters are set) before calling
   * Predict()!
   */
  LinearRegressionModel() : lambda(0.0), intercept(true) { }

  /**
   * Train the LinearRegression model on the given data. Careful! This will
   * completely ignore and overwrite the existing model. This particular
   * implementation does not have an incremental training algorithm.  To set the
   * regularization parameter lambda, call Lambda() or set a different value in
   * the constructor.
   *
   * @param predictors X, the matrix of data points to train the model on.
   * @param responses y, the responses to the data points.
   * @param intercept Whether or not to fit an intercept term.
   * @return The least squares error after training.
   */
  double Train(const M<Ele>& predictors,
               const arma::Row<Ele>& responses,
               const bool intercept = true);

  /**
   * Train the LinearRegression model on the given data and weights. Careful!
   * This will completely ignore and overwrite the existing model. This
   * particular implementation does not have an incremental training algorithm.
   * To set the regularization parameter lambda, call Lambda() or set a
   * different value in the constructor.
   *
   * @param predictors X, the matrix of data points to train the model on.
   * @param responses y, the responses to the data points.
   * @param weights Observation weights (for boosting).
   * @param intercept Whether or not to fit an intercept term.
   * @return The least squares error after training.
   */
  double Train(const M<Ele>& predictors,
               const arma::Row<Ele>& responses,
               const arma::Row<Ele>& weights,
               const bool intercept = true);

/**
   * Train the LinearRegression model on the given data and weights. Careful!
   * This will completely ignore and overwrite the existing model. This
   * particular implementation does not have an incremental training algorithm.
   * To set the regularization parameter lambda, call Lambda() or set a
   * different value in the constructor.
   *
   * @param predictors X, the matrix of data points to train the model on.
   * @param responses Y, matrix of the responses.
   * @param intercept Whether or not to fit an intercept term.
   * @return The least squares error after training.
   */

  double Train(const M<Ele>& predictors,
               const arma::Mat<Ele>& responses,
               const bool intercept=true);


/**
   * Train the LinearRegression model on the given data and weights. Careful!
   * This will completely ignore and overwrite the existing model. This
   * particular implementation does not have an incremental training algorithm.
   * To set the regularization parameter lambda, call Lambda() or set a
   * different value in the constructor.
   *
   * @param predictors X, the matrix of data points to train the model on.
   * @param responses Y, matrix of the responses.
   * @param weights Observation weights (for boosting).
   * @param intercept Whether or not to fit an intercept term.
   * @return The least squares error after training.
   */

  double Train(const M<Ele>& predictors,
               const arma::Mat<Ele>& responses,
               const arma::Row<Ele>& weights,
               const bool intercept=true);


  /**
   * Calculate y_i for each data point in points.
   *
   * @param points the data points to calculate with.
   * @param predictions y, will contain calculated values on completion.
   */
  void Predict(const M<Ele>& points, arma::Row<Ele>& predictions) const;

  /**
   * Calculate y_i for each data point in points.
   *
   * @param points the data points to calculate with.
   * @param predictions y, will contain calculated values on completion.
   */
  void Predict(const M<Ele>& points, arma::Mat<Ele>& predictions) const;


  /**
   * Calculate the L2 squared error on the given predictors and responses using
   * this linear regression model. This calculation returns
   *
   * \f[
   * (1 / n) * \| y - X B \|^2_2
   * \f]
   *
   * where \f$ y \f$ is the responses vector, \f$ X \f$ is the matrix of
   * predictors, and \f$ B \f$ is the parameters of the trained linear
   * regression model.
   *
   * As this number decreases to 0, the linear regression fit is better.
   *
   * @param points Matrix of predictors (X).
   * @param responses Transposed vector of responses (y^T).
   */
  double ComputeError(const M<Ele>& points, 
                      const arma::Row<Ele>& responses) const;

  /**
   * Calculate the L2 squared error on the given predictors and responses using
   * this linear regression model. This calculation returns
   *
   * \f[
   * (1 / n) * \| Y - X B \|^2_2
   * \f]
   *
   * where \f$ Y \f$ is the responses matrix, \f$ X \f$ is the matrix of
   * predictors, and \f$ B \f$ is the parameters of the trained linear
   * regression model.
   *
   * As this number decreases to 0, the linear regression fit is better.
   *
   * @param points Matrix of predictors (X).
   * @param responses Transposed vector of responses (y^T).
   */
  double ComputeError(const M<Ele>& points,
                      const arma::Mat<Ele>& responses) const;


  //! Return the parameters (the b vector).
  const arma::Mat<Ele>& Parameters() const { return parameters; }
  //! Modify the parameters (the b vector).
  arma::Mat<Ele>& Parameters() { return parameters; }

  //! Return the Tikhonov regularization parameter for ridge regression.
  double Lambda() const { return lambda; }
  //! Modify the Tikhonov regularization parameter for ridge regression.
  double& Lambda() { return lambda; }

  //! Return whether or not an intercept term is used in the model.
  bool Intercept() const { return intercept; }

  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(parameters));
    ar(CEREAL_NVP(lambda));
    ar(CEREAL_NVP(intercept));
  }

 private:
  /**
   * The calculated B.
   * Initialized and filled by constructor to hold the least squares solution.
   */
  arma::Mat<Ele> parameters;

  /**
   * The Tikhonov regularization parameter for ridge regression (0 for linear
   * regression).
   */
  double lambda;

  //! Indicates whether first parameter is intercept.
  bool intercept;

  void _solve(arma::SpMat<Ele>& A,
                        arma::Mat<Ele> Y){ parameters =  arma::spsolve(A, Y);}
  void _solve(arma::Mat<Ele>& A,
                        arma::Mat<Ele> Y) { parameters =  arma::solve(A, Y);} 
};

typedef LinearRegressionModel <arma::Mat, double> LinearRegression;
} // namespace regression
} // namespace mlpack

// Include implementation.
#include "linear_regression_impl.hpp"

#endif // MLPACK_METHODS_LINEAR_REGRESSION_HPP
