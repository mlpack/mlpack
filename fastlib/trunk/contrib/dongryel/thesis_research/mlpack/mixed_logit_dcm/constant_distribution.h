/** @file constant_distribution.h
 *
 *  The constant (non-random) distribution that can be used for mixed
 *  logit discrete choice model.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_CONSTANT_DISTRIBUTION_H
#define MLPACK_MIXED_LOGIT_DCM_CONSTANT_DISTRIBUTION_H

namespace mlpack {
namespace mixed_logit_dcm {

/** @brief The constant distribution which can be used for mixed logit
 *         discrete choice model.
 */
class ConstantDistribution {

  public:
    class PrivateData {
    };

  public:

    static void GenerateRandomParameters(
      int num_parameters_in, const PrivateData &private_data_in,
      arma::vec *random_parameters_out) {

      // No parameter restriction here.
      random_parameters_out->set_size(num_parameters_in);
      for(int i = 0; i < num_parameters_in; i++) {
        (*random_parameters_out)[i] = core::math::Random(-5.0, 5.0);
      }
    }

    static void SetupDistribution(
      const arma::vec &parameters, PrivateData *private_data) {

      // Does not do anything.
    }

    static void SamplingAccumulatePrecompute(
      const arma::vec &parameters,
      const arma::vec &beta_vector,
      PrivateData *private_data) {

      // Does not do anything.
    }

    /** @brief Returns the (row, col)-th entry of
     *         $\frac{\partial}{\partial \theta} \beta^{\nu}(\theta)$
     */
    static double AttributeGradientWithRespectToParameter(
      const PrivateData &private_data_in,
      const arma::vec &parameters, const arma::vec &beta_vector,
      int row_index, int col_index) {

      // Since it is a constant distribution, the gradient is always
      // 1 for any (row, col) for which (row == col).
      return (row_index == col_index) ? 1.0 : 0.0;
    }

    static void DrawBeta(
      const PrivateData &private_data_in,
      const arma::vec &parameters, arma::vec *beta_out) {

      // Since it is a constant distribution, it is always equal to
      // the parameter.
      *beta_out = parameters;
    }

    static void Init(
      const std::vector<int> &attribute_dimensions_in, int *num_parameters_out,
      PrivateData *private_data_out) {

      *num_parameters_out = attribute_dimensions_in[0];
    }
};
}
}

#endif
