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
  // The parameters of the mixture model
  std::vector<arma::vec> mu_;
  std::vector<arma::mat> sigma_;
  arma::vec omega_;
  size_t number_of_gaussians_;
  size_t dimension_;

 public:

  MoGEM() { }

  ~MoGEM() { }

  void Init(size_t num_gauss, size_t dimension) {
    // Initialize the private variables
    number_of_gaussians_ = num_gauss;
    dimension_ = dimension;

    // Resize the ArrayList of Vectors and Matrices
    mu_.resize(number_of_gaussians_);
    sigma_.resize(number_of_gaussians_);
  }

  std::vector<arma::vec>& mu() {
    return mu_;
  }

  std::vector<arma::mat>& sigma() {
    return sigma_;
  }

  arma::vec& omega() {
    return omega_;
  }

  size_t number_of_gaussians() {
    return number_of_gaussians_;
  }

  size_t dimension() {
    return dimension_;
  }

  arma::vec& mu(size_t i) {
    return mu_[i] ;
  }

  arma::mat& sigma(size_t i) {
    return sigma_[i];
  }

  double omega(size_t i) {
    return omega_[i];
  }

  // The set functions

  void set_mu(size_t i, arma::vec& mu) {
    assert(i < number_of_gaussians_);
    assert(mu.n_elem == dimension_);

    mu_[i] = mu;
  }

  void set_sigma(size_t i, arma::mat& sigma) {
    assert(i < number_of_gaussians_);
    assert(sigma.n_rows == dimension_);
    assert(sigma.n_cols == dimension_);

    sigma_[i] = sigma;
  }

  void set_omega(arma::vec& omega) {
    assert(omega.n_elem == number_of_gaussians());

    omega_ = omega;
    return;
  }

  /**
   * This function outputs the parameters of the model
   * to an arraylist of doubles
   *
   * @code
   * ArrayList<double> results;
   * mog.OutputResults(&results);
   * @endcode
   */
  void OutputResults(std::vector<double>& results) {

    // Initialize the size of the output array
    results.resize(number_of_gaussians_ * (1 + dimension_ * (1 + dimension_)));

    // Copy values to the array from the private variables of the class
    for (size_t i = 0; i < number_of_gaussians_; i++) {
      results[i] = omega_[i];
      for (size_t j = 0; j < dimension_; j++) {
        results[number_of_gaussians_ + (i * dimension_) + j] = (mu_[i])[j];
        for (size_t k = 0; k < dimension_; k++) {
          results[number_of_gaussians_ * (1 + dimension_) +
              (i * dimension_ * dimension_) + (j * dimension_) + k] =
              (sigma_[i])(j, k);
        }
      }
    }
  }

  /**
   * This function prints the parameters of the model
   *
   * @code
   * mog.Display();
   * @endcode
   */
  void Display() {
    // Output the model parameters as the omega, mu and sigma
    Log::Info << " Omega : [ ";
    for (size_t i = 0; i < number_of_gaussians_; i++) {
      Log::Info << omega_[i] << " ";
    }
    Log::Info << "]" << std::endl;

    Log::Info << " Mu : " << std::endl << "[";
    for (size_t i = 0; i < number_of_gaussians_; i++) {
      for (size_t j = 0; j < dimension_ ; j++) {
        Log::Info << (mu_[i])[j];
      }
      Log::Info << ";";
      if (i == (number_of_gaussians_ - 1)) {
        Log::Info << "\b]" << std::endl;
      }
    }
    Log::Info << "Sigma : ";
    for (size_t i = 0; i < number_of_gaussians_; i++) {
      Log::Info << std::endl << "[";
      for (size_t j = 0; j < dimension_ ; j++) {
        for(size_t k = 0; k < dimension_ ; k++) {
          Log::Info << (sigma_[i])(j, k);
        }
        Log::Info << ";";
      }
      Log::Info << "\b]";
    }
    Log::Info << std::endl;
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
                            const arma::vec& weights);
};

}; // namespace gmm
}; // namespace mlpack

#endif // __MLPACK_METHODS_MOG_MOG_EM_HPP
