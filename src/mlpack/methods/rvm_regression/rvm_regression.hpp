/**
 * @file rvmr.hpp
 * @ Clement Mercier
 *
 * Definition of the RVMR class, which performs the 
 * Relevance Vector Machine for regression
**/
#ifndef TATON_RVMR_HPP 
#define  TATON_RVMR_HPP

#include <mlpack/prereqs.hpp>
#include "utils.hpp"

namespace rvmr {

template<typename KernelType>
class RVMR
{
public:


  /**
   * Set the parameters of the RVMR (Relevance Vector Machine for regression) 
   *    object for a given kernel. There are numerous available kernels in 
   *    the mlpack::kernel namespace.
   *    Regulariation parameters are automaticaly set to their optimal values by 
   *    maximizing the marginal likelihood. Optimization is done by Evidence
   *    Maximization.
   * @param kernel Kernel to be used for computation.
   * @param fitIntercept Whether or not center the data according to the *
   *    examples.
   * @param normalize Whether or to normalize the data according to the 
   *    standard deviation of each feature.
   **/
  RVMR(const KernelType& kernel,
       const bool fitIntercept,
       const bool normalize);

  /**
   * Set the parameters of the ARD regression (Automatic Relevance Determination) 
   *    object without any kernel. The class Performs a linear regression with an ARD prior promoting 
   *    sparsity in the final solution. 
   *    Regulariation parameters are automaticaly set to their optimal values by 
   *    the maximmization of the marginal likelihood. Optimization is done by 
   *    Evidence Maximization.
   *    ARD regression is computed whatever the kernel type given for the 
   *    initalization.
   *
   * @param fitIntercept Whether or not center the data according to the 
   *    examples.
   * @param normalize Whether or to normalize the data according to the 
   *    standard deviation of each feature.
   **/
  RVMR(const bool fitIntercept = true,
       const bool normalize = false);

   
  /**
   * Run Relevance Vector Machine for regression. The input matrix 
   *    (like all mlpack matrices) should be
   *    column-major -- each column is an observation and each row is 
   *    a dimension.
   *    
   * @param data Column-major input data (or row-major input data if rowMajor =
   *     true).
   * @param responses Vector of targets.
   **/
  void Train(const arma::mat& data,
	     const arma::rowvec& responses);

  /**
   * Predict \f$\hat{y}_{i}\f$ for each data point in the given data matrix using the
   *    currently-trained RVM model. Only the coefficients of the active basis 
   *    funcions are used for prediction. This allows fast predictions.
   * @param points The data points to apply the model.
   * @param predictions y, which will contained calculated values on completion.
   */
  void Predict(const arma::mat& points,
               arma::rowvec& predictions) const;

  /**
   * Predict \f$\hat{y}_{i}\f$ and the standard deviation of the predictive posterior 
   *    distribution for each data point in the given data matrix using the
   *    currently-trained RVM model. Only the coefficients of the active basis 
   *    funcions are used for prediction. This allows fast predictions.
   * @param points The data points to apply the model.
   * @param predictions y, which will contained calculated values on completion.
   * @param std Standard deviations of the predictions.
   */
  void Predict(const arma::mat& points,
               arma::rowvec& predictions,
	       arma::rowvec& std) const;

  /**
   * Apply the kernel function between the column vectors of two matrices 
   *    X and Y. If X=Y this function comptutes the Gramian matrix.
   * @param X Matrix of dimension \f$ M \times N1 \f$.
   * @param Y Matrix of dimension \f$ M \times N2 \f$.
   * @param gramMatrix of dimension \f$N1 \times N2\f$. Elements are equal
   *    to kernel.Evaluate(\f$ x_{i} \f$,\f$ y_{j} \f$).
   **/
  void applyKernel(const arma::mat& X,
		   const arma::mat& Y,
		   arma::mat& gramMatrix) const;
  

  /**
   * Compute the Root Mean Square Error
   * between the predictions returned by the model
   * and the true repsonses.
   * @param Points Data points to predict.
   * @param responses A vector of targets.
   * @return RMSE
   **/
  double Rmse(const arma::mat& data,
	     const arma::rowvec& responses) const;

  /**
   * Get the coefficents of the full solution vector.
   * The 0 are associated to the inactive basis functions.
   **/
  arma::vec getCoefs() const;

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
   * Get the indices of the active basis functions.
   * 
   * @return activeSet 
   **/
  inline arma::uvec getActiveSet() const {return this->activeSet;}


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
  //! alpha_threshold limit to prune the basis functions.
  float alpha_threshold;
  //! kernel Kernel used.
  KernelType kernel;
  //! Indicates if ARD regression mode is used.
  bool ardRegression;
  //! Kernel length scale.
  double gamma;
  //! Train database.
  arma::mat phi;
  //! Precision of the prior pdfs (independant gaussian).
  arma::rowvec alpha;
  //! Noise inverse variance.
  double beta;
  //! Solution vector.
  arma::colvec omega;
  //! Coavriance matrix of the solution vector omega.
  arma::mat matCovariance;
  //! activeSetive Indices of active basis functions.
  arma::uvec activeSet;
   
};
} // namespace rvmr
// include implementation.
#include "rvmr_impl.hpp"

#endif
