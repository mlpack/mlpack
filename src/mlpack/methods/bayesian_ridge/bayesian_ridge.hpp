/**
 * @file bayesian_ridge.hpp
 * @ Clement Mercier
 *
 * Definition of the BayesianRidge class, which performs the
 * bayesian linear regression. According to the armadillo standards,
 * all the functions consider data in column-major format.
**/
#ifndef MLPACK_METHODS_BAYESIAN_RIDGE_BAYESIAN_RIDGE_HPP
#define  MLPACK_METHODS_BAYESIAN_RIDGE_BAYESIAN_RIDGE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack{
namespace regression{
/**
 * This class implements the bayesian linear regression. "Bayesian treatment
 * of linear regression, which will avoid the over-fitting problem of maximum
 * likelihood, and which will also lead to automatic methods of determining
 * model complexity using the training data alone.", C.Bishop.
 *
 * More details and description in :
 * Christopher Bishop (2006), Pattern Recognition and Machine Learning.
 * David J.C MacKay (1991), Bayesian Interpolation, Computation and Neural
 * systems.
 
 * Model optimization is automatic and does not require cross validation
 * procedure to be optimized.
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
 * @encode
 *   
 * Example of use:
 *
 * @code
 * arma::mat Xtrain; // Train data matrix. Column-major.
 * arma::rowvec ytrain; // Train target values.
 
 * // Train the model. Regularization strength is optimally tunned with the
 * // training data alone by applying the Train method.
 * BayesianRidge estimator(); // Instanciate the estimator with default option.
 * estimator.Train(Xtrain, ytrain);
 
 * // Prediction on test points.
 * arma::mat Xtest; // Test data matrix. Column-major.
 * arma::rowvec predictions;
 
 * estimator.Predict(Xtest, prediction);
 
 * arma::rowvec ytest; // Test target values.
 * estimator.Rmse(Xtest, ytest); // Evaluate using the RMSE score.
 
 * // Compute the standard deviations of the predictions.
 * arma::rowvec stds;
 * estimator.Predict(Xtest, responses, stds)
 * @endcode
 */
class BayesianRidge
{
 public:
  /**
   * Set the parameters of Bayesian Ridge regression object. The
   *    regulariation parameter is automaticaly set to its optimal value by
   *    maximmization of the marginal likelihood.
   *
   * @param fitIntercept Whether or not center the data according to the
   *    examples.
   * @param normalize Whether or to normalize the data according to the
   *    standard deviation of each feature.
   **/
  BayesianRidge(const bool fitIntercept = true,
                const bool normalize = false);

  /**
   * Run BayesianRidge regression. The input matrix (like all mlpack matrices)
   * should be
   * column-major -- each column is an observation and each row is a dimension.
   * 
   * @param data Column-major input data
   * @param responses A vector of targets.
   * @return score. Root Mean Square Error. Equal to -1 of two feature vectors 
   *    or more are colinear.
   **/
  float Train(const arma::mat& data,
              const arma::rowvec& responses);

  /**
   * Predict \f$y_{i}\f$ for each data point in the given data matrix using the
   * currently-trained Bayesian Ridge model.
   *
   * @param points The data points to apply the model.
   * @param predictions y, Contain the  predicted values on completion.
   **/
  void Predict(const arma::mat& points,
               arma::rowvec& predictions) const;

  /**
   * Predict \f$y_{i}\f$ for one point using the
   * currently-trained Bayesian Ridge model.
   *
   * @param point The data point to apply the model.
   * @param prediction y, which will contained predicted value on completion.
   **/

  void Predict(const arma::colvec& point, double& prediction) const;

  /**
   * Predict \f$y_{i}\f$ and the standard deviation of the predictive posterior 
   * distribution for each data point in the given data matrix, using the
   * currently-trained Bayesian Ridge estimator.
   *
   * @param points The data point to apply the model.
   * @param predictions y, which will contained calculated values on completion.
   * @param std Standard deviations of the predictions.
   */
  void Predict(const arma::mat& points,
               arma::rowvec& predictions,
               arma::rowvec& std) const;


  /**
   * Predict \f$y_{i}\f$ and the standard deviation of the predictive posterior 
   * distribution for point stored in a column vector using the
   * currently-trained Bayesian Ridge estimator.
   *
   * @param point The data points to apply the model.
   * @param prediction y, which will contained calculated values on completion.
   * @param std Standard deviation of the prediction.
   */
  void Predict(const arma::colvec& point,
               double& prediction,
               double& std) const;


   /**
   * Compute the Root Mean Square Error
   * between the predictions returned by the model
   * and the true repsonses.
   *
   * @param Points Data points to predict
   * @param responses A vector of targets.
   * @return RMSE
   **/
  double Rmse(const arma::mat& data,
              const arma::rowvec& responses) const;


  /**
   * Center and normalize the data. The last four arguments
   * allow future modifation of new points.
   *
   * @param data Design matrix in column-major format, dim(P,N).
   * @param responses A vector of targets.
   * @param fit_interpept If true data will be centred according to the points.
   * @param fit_interpept If true data will be scales by the standard deviations
   *     of the features computed according to the points.
   * @param data_proc data processed, dim(N,P).
   * @param responses_proc responses processed, dim(N).
   * @param data_offset Mean vector of the design matrix according to the 
   *     points, dim(P).
   * @param data_scale Vector containg the standard deviations of the features
   *     dim(P).
   * @param reponses_offset Mean of responses.
   */
  void CenterNormalize(const arma::mat& data,
                       const arma::rowvec& responses,
                       const bool fit_intercept,
                       const bool normalize,
                       arma::mat& data_proc,
                       arma::rowvec& responses_proc,
                       arma::colvec& data_offset,
                       arma::colvec& data_scale,
                       double& responses_offset);


  /**
   * Copy constructor. Construct the BayesianRidge object by copying the 
   * given BayesianRidge object.
   *
   * @param other BayesianRidge to copy.
   */
  BayesianRidge(const BayesianRidge& other);

  /**
   * Move constructor. Construct the BayesianRidge object by taking ownership
   * of the the  given BayesianRidge object.
   *
   * @param other BayesianRidge to take the ownership.
   */
  BayesianRidge(BayesianRidge&& other);

  /**
   * Copy the given BayesianRidge object.
   *
   * @param other BayesianRidge object to copy.
   */
  BayesianRidge& operator=(const BayesianRidge& other);

  /**
   * Take ownership of the given BayesianRidge object.
   *
   * @param other BayesianRidge object to copy.
   */
  BayesianRidge& operator=(BayesianRidge&& other);


  /**
   * Get the solution vector
   *
   * @return omega Solution vector.
   **/
  inline arma::colvec Omega() const{return this->omega;}


  /**
   * Get the precesion (or inverse variance) beta of the model.
   *
   * @return \f$ \beta \f$
   **/
  inline double Beta() const {return this->beta;}


   /**
   * Get the estimated variance.
   *   
   * @return 1.0 / \f$ \beta \f$
   **/
  inline double Variance() const {return 1.0 / this->Beta();}


  /**
   * Get the mean vector computed on the features over the training points.
   * Vector of 0 if fitIntercept is false.
   *   
   * @return responses_offset
   **/
  inline arma::rowvec Data_offset() const {return this->data_offset;}


  /**
   * Get the vector of standard deviations computed on the features over the 
   * training points. Vector of 1 if normalize is false.
   *  
   * return data_offset
   **/
  inline arma::rowvec Data_scale() const {return this->data_scale;}


  /**
   * Get the mean value of the train responses.
   * @return responses_offset
   **/
  inline double Responses_offset() const
  {return this->responses_offset;}


  /**
   * Serialize the BayesianRidge model.
   **/
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Center the data if true.
  bool fitIntercept;

  //! Scale the data by standard deviations if true.
  bool normalize;

  //! Mean vector computed over the points.
  arma::colvec data_offset;

  //! Std vector computed over the points.
  arma::colvec data_scale;

  //! Mean of the response vector computed over the points.
  double responses_offset;

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
};
} // namespace regression
} // namespace mlpack

// Include implementation of serialize.
#include "bayesian_ridge_impl.hpp"

#endif
