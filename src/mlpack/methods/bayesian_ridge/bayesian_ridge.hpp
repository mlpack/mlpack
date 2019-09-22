/**
 * @file bayesian_ridge.hpp
 * @ Clement Mercier
 *
 * Definition of the BayesianRidge class, which performs the 
 * bayesian linear regression.
**/
#ifndef MLPACK_METHODS_BAYESIAN_RIDGE_BAYESIAN_RIDGE_HPP 
#define  MLPACK_METHODS_BAYESIAN_RIDGE_BAYESIAN_RIDGE_HPP 

#include <mlpack/prereqs.hpp>

namespace mlpack{
namespace regression{

class BayesianRidge
{
public:
  /**
   * Set the parameters of Bayesian Ridge regression object. The
   *    regulariation parameter is automaticaly set to its optimal value by 
   *    maximmization of the marginal likelihood.
   *
   * @param fitIntercept Whether or not center the data according to the *
   *      examples.
   * @param normalize Whether or to normalize the data according to the 
   * standard deviation of each feature.
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
   **/
  void Train(const arma::mat& data,
	     const arma::rowvec& responses);

  /**
   * Predict \f$y_{i}\f$ for each data point in the given data matrix using the
   * currently-trained Bayesian Ridge model.
   *
   * @param points The data points to apply the model.
   * @param predictions y, which will contained calculated values on completion.
   **/
  void Predict(const arma::mat& points,
               arma::rowvec& predictions) const;

  /**
   * Predict \f$y_{i}\f$ for one point using the
   * currently-trained Bayesian Ridge model.
   *
   * @param point The data point to apply the model.
   * @param prediction y, which will contained calculated value on completion.
   **/

  void Predict(const arma::colvec& point, double& prediction) const;

  /**
   * Predict \f$y_{i}\f$ and the standard deviation of the predictive posterior 
   * distribution for each data point in the given data matrix using the
   * currently-trained Bayesian Ridge estimator.
   *
   * @param points The data points to apply the model.
   * @param predictions y, which will contained calculated values on completion.
   * @param std Standard deviations of the predictions.
   * @param rowMajor Should be true if the data points matrix is row-major and
   *     false otherwise.
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
   * and the true repsonses
   * @param Points Data points to predict
   * @param responses A vector of targets.
   * @return RMSE
   **/
  double Rmse(const arma::mat& data,
	     const arma::rowvec& responses) const;
  
  /*
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
   * Move constructor . Construct the BayesianRidge object by taking ownership
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
   * Take ownershipof  the given BayesianRidge object.
   *
   * @param other BayesianRidge object to copy.
   */
  BayesianRidge& operator=(BayesianRidge&& other);

  
  /**
   * Get the solution vector
   * @return omega Solution vector.
   **/
  inline arma::colvec getCoefs() const{return this->omega;}


  /**
   * Get the precesion (or inverse variance) beta of the model.
   * @return \f$ \beta \f$ 
   **/
  inline double getBeta() const {return this->beta;} 
  

  /**
   * Get the estimated variance.
   * @return 1.0 / \f$ \beta \f$
   **/
  inline double getVariance() const {return 1.0 / this->getBeta();}


  /**
   * Get the mean vector computed on the features over the training points.
   * Vector of 0 if fitIntercept is false.
   * @return responses_offset
   **/
  inline arma::rowvec getdata_offset() const {return this->data_offset;}


  /**
   * Get the vector of standard deviations computed on the features over the 
   *    training points. Vector of 1 if normalize is false.
   * @return data_offset
   **/
  inline arma::rowvec getdata_scale() const {return this->data_scale;}


  /**
   * Get the mean value of the train responses.
   * @return responses_offset
   **/
  inline double getresponses_offset() const
  {return this->responses_offset;}

  

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

private:
  //! Center the data if true
  bool fitIntercept;

  //! Scale the data by standard deviations if true
  bool normalize;

  //! Mean vector computed over the points
  arma::colvec data_offset;

  //! Std vector computed over the points
  arma::colvec data_scale;

  //! Mean of the response vector computed over the points
  double responses_offset;

  //! Precision of the prio pdf (gaussian)
  double alpha;

  //! Noise inverse variance
  double beta;

  //! Effective number of parameters
  double gamma;

  //! Solution vector
  arma::colvec omega;

  //! Coavriance matrix of the solution vector omega
  arma::mat matCovariance;
};
} // namespace regression
} // namespace mlpack

// Include implementation of serialize
#include "bayesian_ridge_impl.hpp"

#endif

  
