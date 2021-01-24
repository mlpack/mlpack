/**
 * @file rvm_regression.hpp
 * @ Clement Mercier
 *
 * Definition of the RVMRegression class, which performs the 
 * Relevance Vector Machine for regression
**/

#ifndef MLPACK_METHODS_RVM_REGRESSION_HPP
#define MLPACK_METHODS_RVM_REGRESSION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack{
namespace regression{

/**
 * A sparse bayesian approach to the maximum likelihood estimation of the 
 * parameters \f$ \omega \f$ of the linear regression model. The Complexity is 
 * governed by the addition of a ARD prior impososed on the precisions 
 * \f$ \alpha_{i} \f$ over the \f$ \omega_{i} \f$:  
 *
 * \f[
 * p(\omega|\alpha) = \prod_{i=1}^{M} \mathcal{N}(\omega_{i}|0, \alpha_{i}^{-1})
 * \f]
 * 
 * The optimization procedure calculates the posterior distribution of 
 * \f$ \omega \f$ knowing the data by maximizing an approximation of the log 
 * marginal likelihood derived from a type II maximum likelihood approximation. 
 * The determination of the \f$ alpha_{i} \f$ and of the noise precision 
 * \f$ beta \f$ is part of the optimization process, leading to an automatic 
 * determination of w. The model being entirely based on probabilty 
 * distributions, uncertainties are available and easly computed for both the 
 * parameters and the predictions.
 *
 * The Relevance Vector Machine is a bayesian sparse kernel method similar to
 * the SVM in the handling of non linear data. Instead of simple scalars, the 
 * RVM outputs are posterior distributions informing about the uncertainties on
 * the predictions. 
 *
 * The ARD prior leads to sparse solution by pruning out all the \f$ w_{i} \f$ 
 * whose the precisions \f$ a_{i} \f$ tend to infinity. The solution is 
 * generally much sparser than for the SVM. 
 *  
 * The code below is an implementation of the maximization of the evidence 
 * function described in the section 7.2.1 of the C.Bishop's book, Pattern
 * Recognition and Machine Learning.  
 *
 * @code
 * @article{MacKay91bayesianinterpolation,
 *   author = {Micheal E. Tipping},
 *   title = {Sparse Bayesian Learning ad the Relevance Vector Machine},
 *   journal = {Journal if Machine Learning Research},
 *   year = {2001},
 *   volume = {1},
 *   pages = {211--244}
 * }
 * @endcode
 *
 * @code
 * @book{Bishop:2006:PRM:1162264,
 *   author = {Bishop, Christopher M.},
 *   title = {Pattern Recognition and Machine Learning (Information Science 
 *            and Statistics)},
 *   chapter = {7}
 *   year = {2006},
 *   isbn = {0387310738},
 *   publisher = {Springer-Verlag},
 *   address = {Berlin, Heidelberg},
 * } 
 * @endcode
 *   
 * Example of use:
 * This example shows how to train RVM regression model with kernel and 
 * a ARD regression model. The same class is used in both cases.

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
  template<typename KernelType=mlpack::kernel::LinearKernel>
class RVMRegression
{
public:
  /**
   * Set the parameters of the RVMRegression (Relevance Vector Machine for 
   * regression) object for a given kernel. There are numerous available kernels 
   * in the mlpack::kernel namespace. Regulariation parameters are automaticaly 
   * set to their optimal values by maximization the marginal likelihood. 
   *
   * @param kernel Kernel to be used for computation.
   * @param centerData Whether or not center the data according to the *
   *    examples.
   * @param scaleData Whether or to scaleData the data according to the 
   *    standard deviation of each feature.
   * @param ard If true fit a ARD regression model. kernel is ignored.
   * @param alphaTresh Value from which the posterior distributions of the w_i
   * are considered to be centered auround zero with certainty. 
   * @param tol Level from which the solution is considered sufficientlly 
   *    stable.  
   * @param nIterMax Maximum number of iterations for convergency.

   **/
  RVMRegression(const KernelType& kernel,
                const bool centerData,
                const bool scaleData,
                const bool ard,
		const double alphaTresh,
		const double tol,
		const int nIterMax);

  /**
   * Defaut constructor for ARD regression. The class Performs a linear 
   * regression with an ARD prior promoting sparsity in the final solution. 
   * Regulariation parameters are automaticaly set to their optimal values by 
   * the maximmization of the marginal likelihood. ARD regression is computed 
   * whatever the kernel given at initalization.
   *
   * @param centerData Whether or not center the data according to the 
   *    examples.
   * @param scaleData Whether or to scaleData the data according to the 
   *    standard deviation of each feature.
   * @param ard If true fit a ARD regression model. kernel is ignored.
   **/
  RVMRegression(const KernelType& kernel=mlpack::kernel::LinearKernel(),
		const bool centerData = false,
                const bool scaleData = false,
		const bool ard = true);
   
  /**
   * Run Relevance Vector Machine for regression. The input matrix 
   * (like all mlpack matrices) should be column-major -- each column is an 
   * observation and each row is a dimension.
   *    
   * @param data Column-major input data (or row-major input data if rowMajor =
   *     true).
   * @param responses Vector of targets.
   **/
  void Train(const arma::mat& data,
	     const arma::rowvec& responses);

  /**
   * Predict \f$\hat{y}_{i}\f$ for each data point in the given data matrix using the
   * currently-trained RVM model. Only the coefficients of the active basis 
   * funcions are used for prediction. This allows fast predictions.
   *
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
  const arma::colvec& Omega() const { return omega; };

  /**
   * Get the precesion (or inverse variance) beta of the model.
   * @return \f$ \beta \f$ 
   **/
  double Beta() const { return beta; }

   /**
   * Get the precesion (or inverse variance) of the coeffcients.
   * @return \f$ \alpha_{i} \f$ 
   **/
  const arma::rowvec& Alpha() const { return alpha; }

  /**
   * Get the estimated variance.
   * @return 1.0 / \f$ \beta \f$
   **/
  double Variance() const { return 1.0 / Beta(); }

  /**
   * Get the indices of the active basis functions.
   * 
   * @return activeSet 
   **/
  const arma::uvec& ActiveSet() const { return activeSet; }

  /**
   * Get the mean vector computed on the features over the training points.
   * Vector of 0 if centerData is false.
   *   
   * @return responsesOffset
   */
  const arma::colvec& DataOffset() const { return dataOffset; }

  /**
   * Get the vector of standard deviations computed on the features over the 
   * training points. Vector of 1 if scaleData is false.
   *  
   * return dataOffset
   */
  const arma::colvec& DataScale() const { return dataScale; }

  /**
   * Get the mean value of the train responses.
   * @return responsesOffset
   */
  double ResponsesOffset() const { return responsesOffset; }

  /**
   * Get the relevant vectors.
   * @return relevantVectors
   */
  const arma::mat& RelevantVectors() const { return relevantVectors; }

  /**
   * Serialize the RVM regression model.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

private:
  //! Center the data if true.
  bool centerData;

  //! Scale the data by standard deviations if true.
  bool scaleData;

  //! Mean vector computed over the points.
  arma::colvec dataOffset;

  //! Std vector computed over the points.
  arma::colvec dataScale;

  //! Mean of the response vector computed over the points.
  double responsesOffset;

  //! Indicates that ARD mode is used.
  bool ard;

  //! alphaThresh limit to prune the basis functions.
  double alphaThresh;

  //! Level from which the solution is considered sufficientlly stable.
  double tol;

  //! Maximum number of iterations for convergency.
  int nIterMax;  

  //! kernel Kernel used.
  KernelType kernel;

  //! Kernel length scale.
  double gamma;

  //! Relevant vectors.
  arma::mat relevantVectors;

  //! Precision of the prior pdfs (independant gaussian).
  arma::colvec alpha;

  //! Noise inverse variance.
  double beta;

  //! Solution vector.
  arma::colvec omega;

  //! Coavriance matrix of the solution vector omega.
  arma::mat matCovariance;

  //! activeSetive Indices of active basis functions.
  arma::uvec activeSet;

  /**
   * Center and scale the data. The last four arguments
   * allow future modification of new points.
   *
   * @param data Design matrix in column-major format, dim(P, N).
   * @param responses A vector of targets.
   * @param centerData If true data will be centred according to the points.
   * @param centerData If true data will be scales by the standard deviations
   *     of the features computed according to the points.
   * @param dataProc data processed, dim(N,P).
   * @param responsesProc responses processed, dim(N).
   *
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
   * @param responsesProc responses processed, dim(N).
  */ 
  void CenterScaleDataPred(const arma::mat& data, 
                           arma::mat& dataProc) const;
  /**
   * Apply the kernel function between the column vectors of two matrices 
   *    X and Y.
   * @param matX Matrix of dimension \f$ M \times N1 \f$.
   * @param matY Matrix of dimension \f$ M \times N2 \f$.
   * @param kernelMatrix Matrix of dimension \f$N1 \times N2\f$. 
   **/
  void applyKernel(const arma::mat& matX,
		   const arma::mat& matY,
		   arma::mat& kernelMatrix) const;

  /**
   * Construct the symmetric kernel matrix from one matrix by copying the 
   * upper triangular part in the lower triangular part.
   *
   * @param matX Marix of dimension \f$ M \times N1 \f$.
   * @param kernelMatrix of dimension \f$N1 \times N2\f$. Elements are equal.
   */
  void applyKernel(const arma::mat& matX,
		   arma::mat& kernelMatrix) const;
};
       
} // namespace regression
} // namespace mlpack

// Include implementation.
#include "rvm_regression_impl.hpp"

#endif
