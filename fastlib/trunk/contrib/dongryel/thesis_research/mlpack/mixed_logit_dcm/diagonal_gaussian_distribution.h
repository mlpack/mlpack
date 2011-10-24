/** @file diagonal_gaussian_distribution.h
 *
 *  The Gaussian distribution that can be used for mixed logit
 *  discrete choice model. This supports only the diagonal covariance.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_DIAGONAL_GAUSSIAN_DISTRIBUTION_H
#define MLPACK_MIXED_LOGIT_DCM_DIAGONAL_GAUSSIAN_DISTRIBUTION_H

namespace mlpack {
namespace mixed_logit_dcm {

/** @brief The Gaussian distribution which can be used for mixed logit
 *         discrete choice model. This Gaussian distribution has the
 *         mean and the square root of the covariance.
 */
class DiagonalGaussianDistribution {
  public:

    class PrivateData {
      public:

        int solution_dimension_;

        /** @brief Stores the cached solution to the diagonal linear
         *         system solved for computing the attribute gradient
         *         with respect to parameter.
         */
        arma::vec cached_solution_;

      public:
        PrivateData() {
          solution_dimension_ = 0;
        }
    };

  public:

    static void GenerateRandomParameters(
      int num_parameters_in,  const PrivateData &private_data_in,
      arma::vec *random_parameters_out) {

      // The first half is for the mean (any number), the second half
      // is for the standard deviation (positive number).
      for(int i = 0; i < num_parameters_in / 2; i++) {
        (*random_parameters_out)[i] = core::math::Random(-5.0, 5.0);
      }
      for(int i = num_parameters_in / 2; i < num_parameters_in; i++) {
        (*random_parameters_out)[i] = core::math::Random(0.3, 1.0);
      }
    }

    /** @brief This function is called whenever the parameter changes.
     */
    static void SetupDistribution(
      const arma::vec &parameters, PrivateData *private_data) {

    }

    /** @brief This function is called before each beta sample is used
     *         to accumulate the simulated probabilities and the
     *         gradient/Hessians.
     */
    static void SamplingAccumulatePrecompute(
      const arma::vec &parameters, const arma::vec &beta_vector,
      PrivateData *private_data) {

      // Solve. The right hand side is basically beta_vector shifted
      // by the means.
      arma::vec mean_vector(
        const_cast<arma::vec &>(parameters).memptr(),
        beta_vector.n_elem, false);
      arma::vec right_hand_side = beta_vector - mean_vector;

      // Divide by the diagoinal entries of the parameters.
      arma::vec diagonal_covariance(
        const_cast<arma::vec &>(parameters).memptr() + beta_vector.n_elem,
        beta_vector.n_elem, false);
      private_data->cached_solution_ = right_hand_side / diagonal_covariance;
    }

    /** @brief Returns the (row, col)-th entry of
     *         $\frac{\partial}{\partial \theta} \beta^{\nu}(\theta)$
     */
    static double AttributeGradientWithRespectToParameter(
      const PrivateData &private_data,
      const arma::vec &parameters, const arma::vec &beta_vector,
      int row_index, int col_index) {

      int num_attributes = beta_vector.n_elem;

      // Upper half of $K \times K$ block is the identity matrix.
      if(row_index < num_attributes) {
        if(row_index == col_index) {
          return 1.0;
        }
        else {
          return 0.0;
        }
      }

      // Lower half.
      else {
        if(row_index == col_index + num_attributes) {
          return private_data.cached_solution_[col_index];
        }
        else {
          return 0.0;
        }
      }
    }

    /** @brief Draws a new $\beta$ from the Gaussian distribution.
     */
    static void DrawBeta(
      const PrivateData &private_data,
      const arma::vec &parameters, arma::vec *beta_out) {

      arma::vec random_gaussian_vector;
      random_gaussian_vector.set_size(private_data.solution_dimension_);
      for(int i = 0; i < private_data.solution_dimension_; i++) {
        random_gaussian_vector[i] = core::math::RandGaussian(1.0);
      }

      // Multiply by the Cholesky factor and shift it by the mean.
      arma::vec mean(
        const_cast<arma::vec &>(parameters).memptr(),
        private_data.solution_dimension_, false);
      arma::vec diagonal_covariance(
        const_cast<arma::vec &>(parameters).memptr() +
        private_data.solution_dimension_,
        private_data.solution_dimension_, false);
      (*beta_out) =
        diagonal_covariance % random_gaussian_vector + mean;
    }

    static void Init(
      const std::vector<int> &attribute_dimensions_in, int *num_parameters_out,
      PrivateData *private_data_out) {

      int num_attributes_in = attribute_dimensions_in[0];
      *num_parameters_out = num_attributes_in * 2;
      private_data_out->solution_dimension_ = num_attributes_in;
    }
};
}
}

#endif
