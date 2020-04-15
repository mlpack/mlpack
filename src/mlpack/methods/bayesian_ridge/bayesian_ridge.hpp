/**
 * @file bayesian_ridge.hpp
 * @author Clement Mercier
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
   * @param centerData Whether or not center the data according to the
   *    examples.
   * @param scaleData Whether or to scale the data according to the
   *    standard deviation of each feature.
   * @param nIterMax Maximum number of iterations for convergency.
   * @param tol Level from which the solution is considered sufficientlly 
   *    stable.  
   **/
  BayesianRidge(const bool centerData = true,
                const bool scaleData = false,
                const int nIterMax = 50,
                const double tol = 1e-4);

  /**
   * Run BayesianRidge regression. The input matrix (like all mlpack matrices)
   * should be
   * column-major -- each column is an observation and each row is a dimension.
   * 
   * @param data Column-major input data
   * @param responses A vector of targets.
   * @return score. Root Mean Square Error. Equal to -1 if two feature vectors 
   *    or more are colinear.
   **/
  double Train(const arma::mat& data,
               const arma::rowvec& responses);

  /**
   * Predict \f$y_{i}\f$ for each data point in the given data matrix using the
   * currently-trained Bayesian Ridge model.
   *
   * @param points The data points to apply the model.
   * @param predictions y, Contains the  predicted values on completion.
   *
   * @return Root mean squared error computed on the train set.
   **/
  void Predict(const arma::mat& points,
               arma::rowvec& predictions) const;

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
   * Center and scaleData the data. The last four arguments
   * allow future modifation of new points.
   *
   * @param data Design matrix in column-major format, dim(P,N).
   * @param responses A vector of targets.
   * @param centerData If true data will be centred according to the points.
   * @param centerData If true data will be scales by the standard deviations
   *     of the features computed according to the points.
   * @param dataProc data processed, dim(N,P).
   * @param responsesProc responses processed, dim(N).
   * @param dataOffset Mean vector of the design matrix according to the 
   *     points, dim(P).
   * @param dataScale Vector containg the standard deviations of the features
   *     dim(P).
   * @return reponsesOffset Mean of responses.
   */
  double CenterScaleData(const arma::mat& data,
                         const arma::rowvec& responses,
                         const bool centerData,
                         const bool scaleData,
                         arma::mat& dataProc,
                         arma::rowvec& responsesProc,
                         arma::colvec& dataOffset,
                         arma::colvec& dataScale);

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
arma::colvec& Omega() const { return this->omega; }

  /**
   * Get the precesion (or inverse variance) beta of the model.
   *
   * @return \f$ \beta \f$
   */
  double Beta() const { return this->beta; }

  /**
   * Get the estimated variance.
   *   
   * @return 1.0 / \f$ \beta \f$
   */
  double Variance() const { return 1.0 / this->Beta(); }

  /**
   * Get the mean vector computed on the features over the training points.
   * Vector of 0 if centerData is false.
   *   
   * @return responsesOffset
   */
  arma::colvec& DataOffset() const { return this->dataOffset; }

  /**
   * Get the vector of standard deviations computed on the features over the 
   * training points. Vector of 1 if scaleData is false.
   *  
   * return dataOffset
   */
  inline arma::colvec DataScale() const {return this->dataScale;}

  /**
   * Get the mean value of the train responses.
   * @return responsesOffset
   */
  inline double ResponsesOffset() const
  {return this->responsesOffset;}

  /**
   * Serialize the BayesianRidge model.
   **/
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Center the data if true.
  bool centerData;

  //! Scale the data by standard deviations if true.
  bool scaleData;

  //! Maximum number of iterations for convergency.
  int nIterMax;

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
};
} // namespace regression
} // namespace mlpack

// Include implementation of serialize.
#include "bayesian_ridge_impl.hpp"

#endif
