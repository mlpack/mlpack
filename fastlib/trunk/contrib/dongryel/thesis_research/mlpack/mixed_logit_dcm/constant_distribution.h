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

    static void AttributeGradientWithRespectToParameterPrecompute(
      const arma::vec &parameters, const arma::vec &beta_vector,
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
      // 1 for any (row, col) entry.
      return 1.0;
    }

    static void DrawBeta(
      const PrivateData &private_data_in,
      const arma::vec &parameters, arma::vec *beta_out) {

      // Since it is a constant distribution, it is always equal to
      // the parameter.
      *beta_out = parameters;
    }

    static void Init(
      int num_attributes_in, int *num_parameters_out,
      PrivateData *private_data_out) {

      *num_parameters_out = num_attributes_in;
    }
};
}
}

#endif
