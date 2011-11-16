/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file mog_em.h
 *
 * Defines a Gaussian Mixture model and
 * estimates the parameters of the model
 */
#ifndef __MLPACK_METHODS_MOG_MOG_EM_HPP
#define __MLPACK_METHODS_MOG_MOG_EM_HPP

#include <mlpack/core.h>

PARAM_MODULE("mog", "Parameters for the Gaussian mixture model.");

PARAM_INT("k", "The number of Gaussians in the mixture model (defaults to 1).",
    "mog", 1);
PARAM_INT("d", "The number of dimensions of the data on which the mixture "
    "model is to be fit.", "mog", 0);

namespace mlpack {
namespace gmm {

/**
 * A Gaussian mixture model class.
 *
 * This class uses maximum likelihood loss functions to
 * estimate the parameters of a gaussian mixture
 * model on a given data via the EM algorithm.
 *
 *
 * Example use:
 *
 * @code
 * MoGEM mog;
 * ArrayList<double> results;
 *
 * mog.Init(number_of_gaussians, dimension);
 * mog.ExpectationMaximization(data, &results, optim_flag);
 * @endcode
 */
class MoGEM {
 private:
  //! The number of Gaussians in the model.
  size_t gaussians;
  //! The dimensionality of the model.
  size_t dimension;
  //! Vector of means; one for each Gaussian.
  std::vector<arma::vec> means;
  //! Vector of covariances; one for each Gaussian.
  std::vector<arma::mat> covariances;
  //! Vector of a priori weights for each Gaussian.
  arma::vec weights;

 public:
  /**
   * Create a GMM with the given number of Gaussians, each of which have the
   * specified dimensionality.
   *
   * @param gaussians Number of Gaussians in this GMM.
   * @param dimension Dimensionality of each Gaussian.
   */
  MoGEM(size_t gaussians, size_t dimension) :
      gaussians(gaussians),
      dimension(dimension),
      means(gaussians),
      covariances(gaussians) { /* nothing to do */ }

  //! Return the number of gaussians in the model.
  const size_t Gaussians() { return gaussians; }

  //! Return the dimensionality of the model.
  const size_t Dimension() { return dimension; }

  //! Return a const reference to the vector of means (mu).
  const std::vector<arma::vec>& Means() const { return means; }
  //! Return a reference to the vector of means (mu).
  std::vector<arma::vec>& Means() { return means; }

  //! Return a const reference to the vector of covariance matrices (sigma).
  const std::vector<arma::mat>& Covariances() const { return covariances; }
  //! Return a reference to the vector of covariance matrices (sigma).
  std::vector<arma::mat>& Covariances() { return covariances; }

  //! Return a const reference to the a priori weights of each Gaussian.
  const arma::vec& Weights() const { return weights; }
  //! Return a reference to the a priori weights of each Gaussian.
  arma::vec& Weights() { return weights; }

  /**
   * This function outputs the parameters of the model
   * to an arraylist of doubles
   *
   * @code
   * ArrayList<double> results;
   * mog.OutputResults(&results);
   * @endcode
   */
  void OutputResults(std::vector<double>& results)
  {
    // Initialize the size of the output array
    results.resize(gaussians * (1 + dimension * (1 + dimension)));

    // Copy values to the array from the private variables of the class
    for (size_t i = 0; i < gaussians; i++)
    {
      results[i] = weights[i];
      for (size_t j = 0; j < dimension; j++)
      {
        results[gaussians + (i * dimension) + j] = (means[i])[j];
        for (size_t k = 0; k < dimension; k++)
        {
          results[gaussians * (1 + dimension) +
              (i * dimension * dimension) + (j * dimension) + k] =
              (covariances[i])(j, k);
        }
      }
    }
  }

  /**
   * This function calculates the parameters of the model
   * using the Maximum Likelihood function via the
   * Expectation Maximization (EM) Algorithm.
   *
   * @code
   * MoG mog;
   * Matrix data = "the data on which you want to fit the model";
   * ArrayList<double> results;
   * mog.ExpectationMaximization(data, &results);
   * @endcode
   */
  void ExpectationMaximization(const arma::mat& data_points);

  /**
   * This function computes the loglikelihood of model.
   * This function is used by the 'ExpectationMaximization'
   * function.
   *
   */
  long double Loglikelihood(const arma::mat& data_points,
                            const std::vector<arma::vec>& means,
                            const std::vector<arma::mat>& covars,
                            const arma::vec& weights) const;
};

}; // namespace gmm
}; // namespace mlpack

#endif
