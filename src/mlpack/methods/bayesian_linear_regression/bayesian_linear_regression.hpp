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

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace regression {

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
 
 * // Train the model. Regularization strength is optimally tunned with the
 * // training data alone by applying the Train method.
 * BayesianLinearRegression estimator(); // Instanciate the estimator with default option.
 * estimator.Train(xTrain, yTrain);
 
 * // Prediction on test points.
 * arma::mat xTest; // Test data matrix. Column-major.
 * arma::rowvec predictions;
 
 * estimator.Predict(xTest, prediction);
 
 * arma::rowvec yTest; // Test target values.
 * estimator.RMSE(xTest, yTest); // Evaluate using the RMSE score.
 
 * // Compute the standard deviations of the predictions.
 * arma::rowvec stds;
 * estimator.Predict(xTest, responses, stds)
 * @endcode
 */
class BayesianLinearRegression
{
 public:
  /**
   * Set the parameters of Bayesian Ridge regression object. The
   * regularization parameter is automatically set to its optimal value by
   * maximization of the marginal likelihood.
   *
   * @param centerData Whether or not center the data according to the
   *    examples.
   * @param scaleData Whether or not scale the data according to the
   *    standard deviation of each feature.
   * @param nIterMax Maximum number of iterations for convergency.
   * @param tol Level from which the solution is considered sufficientlly 
   *    stable.  
   */
  BayesianLinearRegression(const bool centerData = true,
                           const bool scaleData = false,
                           const size_t nIterMax = 50,
                           const double tol = 1e-4);

  /**
   * Run BayesianLinearRegression. The input matrix (like all mlpack matrices) should be
   * column-major -- each column is an observation and each row is a dimension.
   * 
   * @param data Column-major input data, dim(P, N).
   * @param responses A vector of targets, dim(N).
   * @return Root mean squared error.
   */
  double Train(const arma::mat& data,
               const arma::rowvec& responses);

  /**
   * Predict \f$y_{i}\f$ for each data point in the given data matrix using the
   * currently-trained Bayesian Ridge model.
   *
   * @param points The data points to apply the model.
   * @param predictions y, Contains the  predicted values on completion.
   * @return Root mean squared error computed on the train set.
   */
  void Predict(const arma::mat& points,
               arma::rowvec& predictions) const;

  /**
   * Predict \f$y_{i}\f$ and the standard deviation of the predictive posterior 
   * distribution for each data point in the given data matrix, using the
   * currently-trained Bayesian Ridge estimator.
   *
   * @param points The data point to apply the model.
   * @param predictions Vector which will contain calculated values on completion.
   * @param std Standard deviations of the predictions.
   */
  void Predict(const arma::mat& points,
               arma::rowvec& predictions,
               arma::rowvec& std) const;

  /**
   * Compute the Root Mean Square Error between the predictions returned by the
   * model and the true responses.
   *
   * @param data Data points to predict
   * @param responses A vector of targets.
   * @return Root mean squared error.
   **/
  double RMSE(const arma::mat& data,
              const arma::rowvec& responses) const;

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
  const arma::colvec& DataOffset() const { return dataOffset; }

  /**
   * Get the vector of standard deviations computed on the features over the 
   * training points.
   *
   * @return dataOffset
   */
  const arma::colvec& DataScale() const { return dataScale; }

  /**
   * Get the mean value of the train responses.
   *
   * @return responsesOffset
   */
  double ResponsesOffset() const { return responsesOffset; }

  /**
   * Serialize the BayesianLinearRegression model.
   **/
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Center the data if true.
  bool centerData;

  //! Scale the data by standard deviations if true.
  bool scaleData;

  //! Maximum number of iterations for convergency.
  size_t nIterMax;

  //! Level from which the solution is considered sufficientlly stable.
  double tol;

  //! Mean vector computed over the points.
  arma::colvec dataOffset;

  //! Std vector computed over the points.
  arma::colvec dataScale;

  //! Mean of the response vector computed over the points.
  double responsesOffset;

  //! Precision of the prior pdf (gaussian).
  double alpha;

  //! Noise inverse variance.
  double beta;

  //! Effective number of parameters.
  double gamma;

  //! Solution vector
  arma::colvec omega;

  //! Covariance matrix of the solution vector omega.
  arma::mat matCovariance;

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
  double CenterScaleData(const arma::mat& data,
                         const arma::rowvec& responses,
                         arma::mat& dataProc,
                         arma::rowvec& responsesProc);

  /**
   * Center and scale the points before prediction.
   *
   * @param data Design matrix in column-major format, dim(P, N).
   * @param dataProc Data processed, dim(P, N).
   */
  void CenterScaleDataPred(const arma::mat& data,
                           arma::mat& dataProc) const;
};
} // namespace regression
} // namespace mlpack

// Include implementation of serialize.
#include "bayesian_linear_regression_impl.hpp"

#endif
