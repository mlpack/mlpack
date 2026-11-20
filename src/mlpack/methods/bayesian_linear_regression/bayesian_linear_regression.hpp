/**
 * @file methods/bayesian_linear_regression/bayesian_linear_regression.hpp
 * @author Clement Mercier
 *
 * Definition of the BayesianRidge class, which performs the
 * bayesian linear regression. According to the armadillo standards,
 * all the functions consider data in column-major format.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
**/

#ifndef MLPACK_METHODS_BAYESIAN_LINEAR_REGRESSION_HPP
#define MLPACK_METHODS_BAYESIAN_LINEAR_REGRESSION_HPP

#include <mlpack/core.hpp>

namespace mlpack {

/**
 * A Bayesian approach to the maximum likelihood estimation of the parameters
 * \f$ \omega \f$ of the linear regression model. The Complexity is governed by
 * the addition of a gaussian isotropic prior of precision \f$ \alpha \f$ over
 * \f$ \omega \f$:
 *
 * \f[
 * p(\omega|\alpha) = \mathcal{N}(\omega|0, \alpha^{-1}I)
 * \f]
 *
 * The optimization procedure calculates the posterior distribution of
 * \f$ \omega \f$ knowing the data by maximizing an approximation of the log
 * marginal likelihood derived from a type II maximum likelihood approximation.
 * The determination of \f$ alpha \f$ and of the noise precision \f$ beta \f$
 * is part of the optimization process, leading to an automatic determination of
 * w. The model being entirely based on probabilty distributions, uncertainties
 * are available and easly computed for both the parameters and the predictions.
 *
 * The advantage over linear regression and ridge regression is that the
 * regularization is determined from all the training data alone without any
 * require to an hold out method.
 *
 * The code below is an implementation of the maximization of the evidence
 * function described in the section 3.5.2 of the C.Bishop book, Pattern
 * Recognition and Machine Learning.
 *
 * @code
 * @article{MacKay91bayesianinterpolation,
 *   author = {David J.C. MacKay},
 *   title = {Bayesian Interpolation},
 *   journal = {NEURAL COMPUTATION},
 *   year = {1991},
 *   volume = {4},
 *   pages = {415--447}
 * }
 * @endcode
 *
 * @code
 * @book{Bishop:2006:PRM:1162264,
 *   author = {Bishop, Christopher M.},
 *   title = {Pattern Recognition and Machine Learning (Information Science
 *            and Statistics)},
 *   chapter = {3}
 *   year = {2006},
 *   isbn = {0387310738},
 *   publisher = {Springer-Verlag},
 *   address = {Berlin, Heidelberg},
 * }
 * @endcode
 *
 * Example of use:
 *
 * @code
 * arma::mat xTrain; // Train data matrix. Column-major.
 * arma::rowvec yTrain; // Train target values.
 *
 * // Train the model. Regularization strength is optimally tunned with the
 * // training data alone by applying the Train method.
 * // Instantiate the estimator with default option.
 * BayesianLinearRegression estimator;
 * estimator.Train(xTrain, yTrain);
 *
 * // Prediction on test points.
 * arma::mat xTest; // Test data matrix. Column-major.
 * arma::rowvec predictions;
 *
 * estimator.Predict(xTest, prediction);
 *
 * arma::rowvec yTest; // Test target values.
 * estimator.RMSE(xTest, yTest); // Evaluate using the RMSE score.
 *
 * // Compute the standard deviations of the predictions.
 * arma::rowvec stds;
 * estimator.Predict(xTest, responses, stds)
 * @endcode
 */
template<typename ModelMatType = arma::mat>
class BayesianLinearRegression
{
 public:
  using ElemType = typename ModelMatType::elem_type;
  using DenseVecType = typename GetDenseColType<ModelMatType>::type;
  using DenseRowType = typename GetDenseRowType<ModelMatType>::type;

  /**
   * Set the parameters of Bayesian Ridge regression object. The regularization
   * parameter will be automatically set to its optimal value by maximization of
   * the marginal likelihood when training is done.
   *
   * @param centerData Whether or not center the data according to the
   *    examples.
   * @param scaleData Whether or not scale the data according to the
   *    standard deviation of each feature.
   * @param maxIterations Maximum number of iterations for convergency.
   * @param tolerance Level from which the solution is considered sufficientlly
   *    stable.
   */
  BayesianLinearRegression(const bool centerData = true,
                           const bool scaleData = false,
                           const size_t maxIterations = 50,
                           const double tolerance = 1e-4);

  /**
   * Create the BayesianLinearRegression object and train the model.  The
   * regularization parameter is automatically set to its optimal value by
   * maximization of the maginal likelihood.
   *
   * @param data Column-major input data, dim(P, N).
   * @param responses A vector of targets, dim(N).
   * @param centerData Whether or not center the data according to the
   *    examples.
   * @param scaleData Whether or not scale the data according to the
   *    standard deviation of each feature.
   * @param maxIterations Maximum number of iterations for convergency.
   * @param tolerance Level from which the solution is considered sufficientlly
   *    stable.
   * @return Root mean squared error.
   */
  template<typename MatType,
           typename ResponsesType,
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  BayesianLinearRegression(const MatType& data,
                           const ResponsesType& responses,
                           const bool centerData = true,
                           const bool scaleData = false,
                           const size_t maxIterations = 50,
                           const double tolerance = 1e-4);

  /**
   * Run BayesianLinearRegression. The input matrix (like all mlpack matrices)
   * should be column-major -- each column is an observation and each row is a
   * dimension.
   *
   * @param data Column-major input data, dim(P, N).
   * @param responses A vector of targets, dim(N).
   * @param centerData Whether or not center the data according to the
   *    examples.
   * @param scaleData Whether or not scale the data according to the
   *    standard deviation of each feature.
   * @param maxIterations Maximum number of iterations for convergency.
   * @param tolerance Level from which the solution is considered sufficientlly
   *    stable.
   * @return Root mean squared error.
   */
  // Many overloads necessary here until std::optional is available with C++17.
  // The first overload is also necessary to avoid confusing the hyperparameter
  // tuner, so that this can be correctly detected as a regression algorithm.
  template<typename MatType>
  ElemType Train(const MatType& data,
                 const arma::rowvec& responses);

  template<typename MatType,
           typename ResponsesType,
           typename = void, /* so MetaInfoExtractor does not get confused */
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  ElemType Train(const MatType& data,
                 const ResponsesType& responses,
                 const std::optional<bool> centerData = std::nullopt,
                 const std::optional<bool> scaleData = std::nullopt,
                 const std::optional<size_t> maxIterations = std::nullopt);

  template<typename MatType,
           typename ResponsesType,
           typename = void, /* so MetaInfoExtractor does not get confused */
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  ElemType Train(const MatType& data,
                 const ResponsesType& responses,
                 const bool centerData,
                 const bool scaleData,
                 const size_t maxIterations,
                 const double tolerance);

  /**
   * Predict \f$y\f$ for a single data point \f$x\f$ using the currently-trained
   * Bayesian ridge regression model.
   *
   * @param point The data point to apply the model to.
   * @return Prediction for the `point`.
   */
  template<typename VecType>
  ElemType Predict(const VecType& point) const;

  /**
   * Predict \f$y\f$ for a single data point \f$x\f$ using the currently-trained
   * Bayesian ridge regression model, storing the prediction in `prediction` and
   * the standard deviation of the prediction in `stddev`.
   *
   * @param point The data point to apply the model to.
   * @param prediction `double` to store the prediction into.
   * @param stddev `double` to store the standard deviation of the prediction
   * into.
   */
  template<typename VecType>
  void Predict(const VecType& point,
               ElemType& prediction,
               ElemType& stddev) const;

  /**
   * Predict \f$y_{i}\f$ for each data point in the given data matrix using the
   * currently-trained Bayesian Ridge model.
   *
   * @param points The data points to apply the model.
   * @param predictions y, Contains the  predicted values on completion.
   */
  template<typename MatType,
           typename ResponsesType,
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  void Predict(const MatType& points,
               ResponsesType& predictions) const;

  /**
   * Predict \f$y_{i}\f$ and the standard deviation of the predictive posterior
   * distribution for each data point in the given data matrix, using the
   * currently-trained Bayesian Ridge estimator.
   *
   * @param points The data point to apply the model.
   * @param predictions Vector which will contain calculated values on
   *     completion.
   * @param std Standard deviations of the predictions.
   */
  template<typename MatType,
           typename ResponsesType,
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  void Predict(const MatType& points,
               ResponsesType& predictions,
               ResponsesType& std) const;

  /**
   * Compute the Root Mean Square Error between the predictions returned by the
   * model and the true responses.
   *
   * @param data Data points to predict
   * @param responses A vector of targets.
   * @return Root mean squared error.
   **/
  template<typename MatType,
           typename ResponsesType,
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  ElemType RMSE(const MatType& data,
                const ResponsesType& responses) const;

  /**
   * Get the solution vector.
   *
   * @return omega Solution vector.
   */
  const arma::colvec& Omega() const { return omega; }

  /**
   * Get the precision (or inverse variance) of the gaussian prior. Train()
   * must be called before.
   *
   * @return \f$ \alpha \f$
   */
  double Alpha() const { return alpha; }

  /**
   * Get the precision (or inverse variance) beta of the model. Train() must be
   * called before.
   *
   * @return \f$ \beta \f$
   */
  double Beta() const { return beta; }

  /**
   * Get the estimated variance. Train() must be called before.
   *
   * @return 1.0 / \f$ \beta \f$
   */
  double Variance() const { return 1.0 / Beta(); }

  /**
   * Get the mean vector computed on the features over the training points.
   *
   * @return responsesOffset
   */
  const DenseVecType& DataOffset() const { return dataOffset; }

  /**
   * Get the vector of standard deviations computed on the features over the
   * training points.
   *
   * @return dataOffset
   */
  const DenseVecType& DataScale() const { return dataScale; }

  /**
   * Get the mean value of the train responses.
   *
   * @return responsesOffset
   */
  ElemType ResponsesOffset() const { return responsesOffset; }

  //! Get whether the data will be centered during training.
  bool CenterData() const { return centerData; }
  //! Modify whether the data will be centered during training.
  bool& CenterData() { return centerData; }

  //! Get whether the data will be scaled by standard deviations during
  //! training.
  bool ScaleData() const { return scaleData; }
  //! Modify whether the data will be scaled by standard deviations during
  //! training.
  bool& ScaleData() { return scaleData; }

  //! Get the maximum number of iterations for training.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations for training.
  size_t& MaxIterations() { return maxIterations; }

  //! Get the tolerance for training to converge.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for training to converge.
  double& Tolerance() { return tolerance; }

  /**
   * Serialize the BayesianLinearRegression model.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);

 private:
  //! Center the data if true.
  bool centerData;

  //! Scale the data by standard deviations if true.
  bool scaleData;

  //! Maximum number of iterations for convergence.
  size_t maxIterations;

  //! Level from which the solution is considered sufficientlly stable.
  double tolerance;

  //! Mean vector computed over the points.
  DenseVecType dataOffset;

  //! Std vector computed over the points.
  DenseVecType dataScale;

  //! Mean of the response vector computed over the points.
  ElemType responsesOffset;

  //! Precision of the prior pdf (gaussian).
  ElemType alpha;

  //! Noise inverse variance.
  ElemType beta;

  //! Effective number of parameters.
  ElemType gamma;

  //! Solution vector.
  DenseVecType omega;

  //! Covariance matrix of the solution vector omega.
  ModelMatType matCovariance;

  /**
   * Center and scale the data accordind to centerData and scaleData.
   * Allows future modifications of new points.
   *
   * @param data Design matrix in column-major format, dim(P, N).
   * @param responses A vector of targets.
   * @param dataProc Data processed, dim(P, N).
   * @param responsesProc Responses processed, dim(N).
   * @return reponsesOffset Mean of responses.
   */
  template<typename MatType, typename ResponsesType>
  double CenterScaleData(const MatType& data,
                         const ResponsesType& responses,
                         MatType& dataProc,
                         ResponsesType& responsesProc);

  /**
   * Center and scale the points before prediction.  This should only be called
   * if centerData or scaleData is true; if neither is true, then `dataProc`
   * will be unmodified.
   *
   * @param data Design matrix in column-major format, dim(P, N).
   * @param dataProc Data processed, dim(P, N).
   */
  template<typename MatType, typename OutMatType>
  void CenterScaleDataPred(const MatType& data,
                           OutMatType& dataProc) const;
};

} // namespace mlpack

CEREAL_TEMPLATE_CLASS_VERSION((typename ModelMatType),
    (mlpack::BayesianLinearRegression<ModelMatType>), (1));

// Include implementation of serialize.
#include "bayesian_linear_regression_impl.hpp"

#endif
