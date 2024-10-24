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

// Because RegressionDistribution uses LinearRegression internally, we need to
// make sure we define LinearRegression fully before we define
// RegressionDistribution.  Therefore we have to include the prereqs first, and
// include the core later.
#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * A simple linear regression algorithm using ordinary least squares.
 * Optionally, this class can perform ridge regression, if the lambda parameter
 * is set to a number greater than zero.
 */
template<typename ModelMatType = arma::mat>
class LinearRegression
{
 public:
  using ModelColType = typename GetColType<ModelMatType>::type;
  using ElemType = typename ModelMatType::elem_type;

  /**
   * Creates the model.
   *
   * @param predictors X, matrix of data points.
   * @param responses y, the measured data for each point in X.
   * @param lambda Regularization constant for ridge regression.
   * @param intercept Whether or not to include an intercept term.
   */
  template<typename MatType,
           typename ResponsesType,
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  LinearRegression(const MatType& predictors,
                   const ResponsesType& responses,
                   const double lambda = 0,
                   const bool intercept = true);

  /**
   * Creates the model with instance-weighted learning.
   *
   * @param predictors X, matrix of data points.
   * @param responses y, the measured data for each point in X.
   * @param weights Instance weights (for boosting).
   * @param lambda Regularization constant for ridge regression.
   * @param intercept Whether or not to include an intercept term.
   */
  template<typename MatType,
           typename ResponsesType,
           typename WeightsType,
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>,
           typename = std::enable_if_t<
               std::is_same_v<typename WeightsType::elem_type, ElemType>>>
  LinearRegression(const MatType& predictors,
                   const ResponsesType& responses,
                   const WeightsType& weights,
                   const double lambda = 0,
                   const bool intercept = true);

  /**
   * Empty constructor.  This gives a non-working model, so make sure Train() is
   * called (or make sure the model parameters are set) before calling
   * Predict()!
   */
  LinearRegression() : lambda(0.0), intercept(true) { }

  /**
   * Train the LinearRegression model on the given data. Careful! This will
   * completely ignore and overwrite the existing model. This particular
   * implementation does not have an incremental training algorithm.  To set the
   * regularization parameter lambda, call Lambda() or set a different value in
   * the constructor.
   *
   * This version of `Train()` is deprecated and will be removed in mlpack
   * 5.0.0.  Use the version of `Train()` that specifies `lambda` before
   * `intercept` instead.
   *
   * @param predictors X, the matrix of data points to train the model on.
   * @param responses y, the responses to the data points.
   * @param intercept Whether or not to fit an intercept term.
   * @return The least squares error after training.
   */
  template<typename T>
  [[deprecated("Will be removed in mlpack 5.0.0, use other constructors")]]
  double Train(const arma::mat& predictors,
               const arma::rowvec& responses,
               const T intercept,
               const std::enable_if_t<std::is_same_v<T, bool>>* = 0);

  /**
   * Train the LinearRegression model on the given data and instance weights.
   * Careful!  This will completely ignore and overwrite the existing model.
   * This particular implementation does not have an incremental training
   * algorithm.  To set the regularization parameter lambda, call Lambda() or
   * set a different value in the constructor.
   *
   * This version of `Train()` is deprecated and will be removed in mlpack
   * 5.0.0.  Use the version of `Train()` that specifies `lambda` before
   * `intercept` instead.
   *
   * @param predictors X, the matrix of data points to train the model on.
   * @param responses y, the responses to the data points.
   * @param weights Instance weights (for boosting).
   * @param intercept Whether or not to fit an intercept term.
   * @return The least squares error after training.
   */
  template<typename T>
  [[deprecated("Will be removed in mlpack 5.0.0, use other constructors")]]
  double Train(const arma::mat& predictors,
               const arma::rowvec& responses,
               const arma::rowvec& weights,
               const T intercept,
               const std::enable_if_t<std::is_same_v<T, bool>>* = 0);

  /**
   * Train the LinearRegression model.  This is a dummy overload so that
   * MetaInfoExtractor can properly detect that LinearRegression is a regression
   * method.
   */
  template<typename MatType>
  ElemType Train(const MatType& predictors,
                 const arma::rowvec& responses);

  /**
   * Train the LinearRegression model on the given data and weights. Careful!
   * This will completely ignore and overwrite the existing model. This
   * particular implementation does not have an incremental training algorithm.
   * To set the regularization parameter lambda, call Lambda() or set a
   * different value in the constructor.
   *
   * @param predictors X, the matrix of data points to train the model on.
   * @param responses y, the responses to the data points.
   * @param lambda L2 regularization penalty parameter to use.
   * @param intercept Whether or not to fit an intercept term.
   * @return The least squares error after training.
   */
  template<typename MatType,
           typename ResponsesType,
           typename = void, /* so MetaInfoExtractor does not get confused */
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  ElemType Train(const MatType& predictors,
                 const ResponsesType& responses,
                 const std::optional<double> lambda = std::nullopt,
                 const std::optional<bool> intercept = std::nullopt);

  /**
   * Train the LinearRegression model.  This is a dummy overload so that
   * MetaInfoExtractor can properly detect that LinearRegression is a regression
   * method.
   */
  template<typename MatType>
  ElemType Train(const MatType& predictors,
                 const arma::rowvec& responses,
                 const arma::rowvec& weights);

  /**
   * Train the LinearRegression model on the given data and instance weights.
   * Careful!  This will completely ignore and overwrite the existing model.
   * This particular implementation does not have an incremental training
   * algorithm.  To set the regularization parameter lambda, call Lambda() or
   * set a different value in the constructor.
   *
   * @param predictors X, the matrix of data points to train the model on.
   * @param responses y, the responses to the data points.
   * @param weights Instance weights (for boosting).
   * @param lambda L2 regularization penalty parameter to use.
   * @param intercept Whether or not to fit an intercept term.
   * @return The least squares error after training.
   */
  template<typename MatType,
           typename ResponsesType,
           typename WeightsType,
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>,
           typename = std::enable_if_t<
               std::is_same_v<typename WeightsType::elem_type, ElemType>>>
  ElemType Train(const MatType& predictors,
                 const ResponsesType& responses,
                 const WeightsType& weights,
                 const std::optional<double> lambda = std::nullopt,
                 const std::optional<bool> intercept = std::nullopt);

  /**
   * Calculate y_i for a single data point.
   *
   * @param point the data point to calculate with.
   */
  template<typename VecType>
  ElemType Predict(const VecType& point) const;

  /**
   * Calculate y_i for each data point in points.
   *
   * @param points the data points to calculate with.
   * @param predictions y, will contain calculated values on completion.
   */
  template<typename MatType, typename ResponsesType>
  void Predict(const MatType& points, ResponsesType& predictions) const;

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
  template<typename MatType, typename ResponsesType>
  ElemType ComputeError(const MatType& points,
                        const ResponsesType& responses) const;

  //! Return the parameters (the b vector).
  const ModelColType& Parameters() const { return parameters; }
  //! Modify the parameters (the b vector).
  ModelColType& Parameters() { return parameters; }

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
  void serialize(Archive& ar, const uint32_t version);

 private:
  /**
   * The calculated B.
   * Initialized and filled by constructor to hold the least squares solution.
   */
  ModelColType parameters;

  /**
   * The Tikhonov regularization parameter for ridge regression (0 for linear
   * regression).
   */
  double lambda;

  //! Indicates whether first parameter is intercept.
  bool intercept;
};

} // namespace mlpack

CEREAL_TEMPLATE_CLASS_VERSION((typename ModelMatType),
    (mlpack::LinearRegression<ModelMatType>), (1));

// Include implementation.
#include "linear_regression_impl.hpp"

// Now that LinearRegression is defined, we can include the core.
#include <mlpack/core.hpp>

#endif // MLPACK_METHODS_LINEAR_REGRESSION_HPP
