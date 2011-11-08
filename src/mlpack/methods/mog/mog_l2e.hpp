/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file mog_l2e.hpp
 *
 * Defines a Gaussian Mixture model and estimates the parameters of the model.
 */
#ifndef __MLPACK_METHODS_MOG_MOG_L2E_HPP
#define __MLPACK_METHODS_MOG_MOG_L2E_HPP

#include <mlpack/core.h>

PARAM_MODULE("mog_l2e", "Parameters for the Gaussian mixture model.");

PARAM_INT("k", "The number of Gaussians in the mixture model (defaults to 1).",
    "mog_l2e", 1);
PARAM_INT("d", "The number of dimensions of the data on which the mixture "
    "model is to be fit.", "mog_l2e", 0);

namespace mlpack {
namespace gmm {

/**
 * A Gaussian mixture model class.
 *
 * This class uses L2 loss function to estimate the parameters of a gaussian
 * mixture model on a given data.
 *
 * The parameters are converted for optimization to maintain the following
 * facts:
 * - the weights sum to one; for this, the weights were parameterized using
 *     the logistic function
 * - the covariance matrix is always positive definite; for this, the Cholesky
 *     decomposition is used
 *
 * Example use:
 *
 * @code
 * MoGL2E mog;
 * ArrayList<double> results;
 * double *params;
 *
 * mog.MakeModel(number_of_gaussians, dimension, params);
 * mog.L2Error(data);
 * mog.OutputResults(&results);
 * @endcode
 */
class MoGL2E {
 private:
  // The parameters of the Mixture model
  size_t gaussians;
  size_t dimension;
  std::vector<arma::vec> means;
  std::vector<arma::mat> covariances;
  arma::vec weights;

  // The differential for the paramterization
  // for optimization
  arma::mat weightsGradients;
  std::vector<std::vector<arma::mat> > covariancesGradients;

 public:

  MoGL2E(size_t gaussians, size_t dimension) :
    gaussians(gaussians),
    dimension(dimension),
    means(gaussians),
    covariances(gaussians) { /* nothing to do */ }

  void Resize_d_sigma_()
  {
    d_sigma_.resize(gaussians);
    for(size_t i = 0; i < gaussians; i++)
      d_sigma_[i].resize(dimension * (dimension + 1) / 2);
  }

  /**
   * This function uses the parameters used for optimization
   * and converts it into athe parameters of a Gaussian
   * mixture model. This is to be used when you do not want
   * the gradient values.
   *
   * Example use:
   *
   * @code
   * MoGL2E mog;
   * mog.MakeModel(number_of_gaussians, dimension,
   *               parameters_for_optimization);
   * @endcode
   */

  void MakeModel(size_t num_mods, size_t dimension, const arma::vec& theta) {
    arma::vec temp_mu(dimension);
    arma::mat lower_triangle_matrix;
    double sum, s_min = 0.01;

    gaussians = num_mods;
    this->dimension = dimension;
    lower_triangle_matrix.set_size(dimension, dimension);

    // calculating the omega values
    arma::vec temp_array = exp(theta);
    temp_array[num_mods - 1] = 1;
    sum = accu(temp_array);

    temp_array /= sum;
    weights = temp_array;

    // calculating the mu values
    for(size_t k = 0; k < gaussians; k++) {
      for(size_t j = 0; j < dimension; j++) {
        temp_mu[j] = theta[num_mods + (k * dimension) + j - 1];
      }
      means[k] = temp_mu;
    }

    // calculating the sigma values
    // using a lower triangular matrix and its transpose
    // to obtain a positive definite symmetric matrix
    arma::mat sigma_temp(dimension, dimension);
    for(size_t k = 0; k < gaussians; k++) {
      lower_triangle_matrix.zeros();
      for(size_t j = 0; j < dimension; j++) {
        for(size_t i = 0; i < j; i++) {
          lower_triangle_matrix(j, i) = theta[(gaussians - 1)
              + (gaussians * dimension) + k * (dimension * (dimension + 1) / 2)
              + (j * (j + 1) / 2) + i];
        }
        lower_triangle_matrix(j, j) = theta[(gaussians - 1)
            + (gaussians * dimension) + k * (dimension * (dimension + 1) / 2)
            + (j * (j + 1) / 2) + j] + s_min;
      }
      sigma_temp = lower_triangle_matrix * trans(lower_triangle_matrix);
      covariances[k] = sigma_temp;
    }
  }

 /**
  * This function uses the parameters used for optimization
  * and converts it into the parameters of a Gaussian
  * mixture model. This is to be used when you want
  * the gradient values.
  *
  * Example use:
  *
  * @code
  * MoGL2E mog;
  * mog.MakeModelWithGradients(number_of_gaussians, dimension,
  *                            parameters_for_optimization);
  * @endcode
  */
  void MakeModelWithGradients(size_t num_mods,
                              size_t dimension,
                              const arma::vec& theta) {
    arma::vec temp_mu(dimension);
    arma::mat lower_triangle_matrix;
    double sum, s_min = 0.01;

    gaussians = num_mods;
    this->dimension = dimension;
    lower_triangle_matrix.set_size(dimension, dimension);

    // calculating the omega values
    arma::vec temp_array = exp(theta);
    temp_array[num_mods - 1] = 1;
    sum = accu(temp_array);

    temp_array /= sum;
    weights = temp_array;

    // calculating the d_omega values
    arma::mat d_omega_temp(num_mods - 1, num_mods);
    d_omega_temp.zeros();
    for (size_t i = 0; i < num_mods - 1; i++) {
      for (size_t j = 0; j < i; j++) {
        d_omega_temp(i, j) = -(weights[i] * weights[j]);
        d_omega_temp(j, i) = -(weights[i] * weights[j]);
      }
      d_omega_temp(i, i) = weights[i] * (1 - weights[i]);
    }

    for (size_t i = 0; i < num_mods - 1; i++)
      d_omega_temp(i, num_mods - 1) = -(weights[i] * weights[num_mods - 1]);

    set_d_omega(d_omega_temp);

    // calculating the mu values
    for (size_t k = 0; k < num_mods; k++) {
      for (size_t j = 0; j < dimension; j++)
        temp_mu[j] = theta[num_mods + (k * dimension) + j - 1];
      means[k] = temp_mu;
    }
    // d_mu is not computed because it is implicitly known
    // since no parameterization is applied on them

    // using a lower triangular matrix and its transpose
    // to obtain a positive definite symmetric matrix

    // initializing the d_sigma values

    arma::mat d_sigma_temp(dimension, dimension);
    Resize_d_sigma_();

    // calculating the sigma values
    arma::mat sigma_temp(dimension, dimension);
    for (size_t k = 0; k < gaussians; k++) {
      lower_triangle_matrix.zeros();
      for (size_t j = 0; j < dimension; j++) {
        for (size_t i = 0; i < j; i++) {
          lower_triangle_matrix(j, i) = theta[(gaussians - 1)
              + (gaussians * dimension) + k * (dimension * (dimension + 1) / 2)
              + (j * (j + 1) / 2) + i];
        }
        lower_triangle_matrix(j, j) = theta[(gaussians - 1)
            + (gaussians * dimension) + k * (dimension * (dimension + 1) / 2)
            + (j * (j + 1) / 2) + j] + s_min;
      }
      sigma_temp = lower_triangle_matrix * trans(lower_triangle_matrix);
      covariances[k] = sigma_temp;

      // calculating the d_sigma values
      for (size_t i = 0; i < dimension; i++) {
        for (size_t in = 0; in < i + 1; in++) {
          arma::mat d_sigma_d_r(dimension, dimension);
          d_sigma_d_r.zeros();
          d_sigma_d_r(i, in) = 1.0;

          d_sigma_temp = d_sigma_d_r * trans(lower_triangle_matrix) +
              lower_triangle_matrix * trans(d_sigma_d_r);
          set_d_sigma(k, (i * (i + 1) / 2) + in, d_sigma_temp);
        }
      }
    }
  }

  arma::mat& d_omega() {
    return d_omega_;
  }

  std::vector<std::vector<arma::mat> >& d_sigma() {
    return d_sigma_;
  }

  std::vector<arma::mat>& d_sigma(size_t i) {
    return d_sigma_[i];
  }

  void set_d_omega(const arma::mat& d_omega) {
    d_omega_ = d_omega;
  }

  void set_d_sigma(size_t i, size_t j, const arma::mat& d_sigma_i_j) {
    d_sigma_[i][j] = d_sigma_i_j;
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
    results.resize(gaussians * (1 + dimension * (1 + dimension)));

    // Copy values to the array from the private variables of the class
    for (size_t i = 0; i < gaussians; i++) {
      results[i] = weights[i];
      for (size_t j = 0; j < dimension; j++) {
        results[gaussians + (i * dimension) + j] = (means[i])[j];
          for (size_t k = 0; k < dimension; k++) {
            results[gaussians * (1 + dimension)
               + (i * dimension * dimension) + (j * dimension)
               + k] = (covariances[i])(j, k);
        }
      }
    }
  }

  /**
   * This function calculates the L2 error and
   * the gradient of the error with respect to the
   * parameters given the data and the parameterized
   * mixture
   *
   * Example use:
   *
   * @code
   * const Matrix data;
   * MoGL2E mog;
   * size_t num_gauss, dimension;
   * double *params; // get the parameters
   *
   * mog.MakeModel(num_gauss, dimension, params);
   * mog.L2Error(data);
   * @endcode
   */
  long double L2Error(const arma::mat& data);
  long double L2Error(const arma::mat& data, arma::vec& gradients);

  /**
   * Calculates the regularization value for a
   * Gaussian mixture and its gradient with
   * respect to the parameters
   *
   * Used by the 'L2Error' function to calculate
   * the regularization part of the error
   */
  long double RegularizationTerm_();
  long double RegularizationTerm_(arma::vec& g_reg);

  /**
   * Calculates the goodness-of-fit value for a
   * Gaussian mixture and its gradient with
   * respect to the parameters
   *
   * Used by the 'L2Error' function to calculate
   * the goodness-of-fit part of the error
   */
  long double GoodnessOfFitTerm_(const arma::mat& data);
  long double GoodnessOfFitTerm_(const arma::mat& data, arma::vec& g_fit);

  /**
   * This function computes multiple number of starting points
   * required for the Nelder Mead method
   *
   * Example use:
   * @code
   * double **p;
   * size_t n, num_gauss;
   * const Matrix data;
   *
   * MoGL2E::MultiplePointsGeneratot(p, n, data, num_gauss);
   * @endcode
   */
  static void MultiplePointsGenerator(arma::mat& points,
                                      const arma::mat& d,
                                      size_t number_of_components);

  /**
   * This function parameterizes the starting point obtained
   * from the 'k_means" for optimization purposes using the
   * Quasi Newton method
   *
   * Example use:
   * @code
   * double *p;
   * size_t num_gauss;
   * const Matrix data;
   *
   * MoGL2E::InitialPointGeneratot(p, data, num_gauss);
   * @endcode
   */
  static void InitialPointGenerator(arma::vec& theta,
                                    const arma::mat& data,
                                    size_t k_comp);

  /**
   * This is the function which would be used for
   * optimization. It creates its own object of
   * class MoGL2E and returns the L2 error
   * and the gradient which are computed by
   * the functions of the class
   *
   */
  static long double L2ErrorForOpt(const arma::vec& params,
                                   const arma::mat& data) {
    size_t num_gauss = (params.n_elem + 1) * 2 /
        ((data.n_rows + 1) * (data.n_rows + 2));

    MoGL2E model(num_gauss, data.n_rows);
    model.MakeModel(num_gauss, data.n_rows, params);

    return model.L2Error(data);
  }

  static long double L2ErrorForOpt(const arma::vec& params,
                                   const arma::mat& data,
                                   arma::vec& gradient) {

    size_t num_gauss = (params.n_elem + 1) * 2 /
        ((data.n_rows + 1) * (data.n_rows + 2));

    MoGL2E model(num_gauss, data.n_rows);
    model.MakeModelWithGradients(num_gauss, data.n_rows, params);

    return model.L2Error(data, gradient);
  }
};

}; // namespace gmm
}; // namespace mlpack

#endif // __MLPACK_METHODS_MOG_MOG_L2E_HPP
