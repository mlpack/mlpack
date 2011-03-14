/** @file constant_distribution.h
 *
 *  The constant (non-random) distribution that can be used for mixed
 *  logit discrete choice model.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_CONSTANT_DISTRIBUTION_H
#define MLPACK_MIXED_LOGIT_DCM_CONSTANT_DISTRIBUTION_H

#include "mlpack/mixed_logit_dcm/distribution.h"

namespace mlpack {
namespace mixed_logit_dcm {

/** @brief The constant distribution which can be used for mixed logit
 *         discrete choice model.
 */
class ConstantDistribution:
  public virtual Distribution {

  private:

    virtual void AttributeGradientWithRespectToParameterPrecompute_(
      const arma::vec &parameters, const arma::vec &beta_vector) const {

      // Does not do anything.
    }

  public:

    ~ConstantDistribution() {
    }

    /** @brief Returns the (row, col)-th entry of
     *         $\frac{\partial}{\partial \theta} \beta^{\nu}(\theta)$
     */
    double AttributeGradientWithRespectToParameter(
      const arma::vec &parameters, const arma::vec &beta_vector,
      int row_index, int col_index) const {

      // Since it is a constant distribution, the gradient is always
      // 1 for any (row, col) entry.
      return 1.0;
    }

    void DrawBeta(
      const arma::vec &parameters, arma::vec *beta_out) const {

      // Since it is a constant distribution, it is always equal to
      // the parameter.
      *beta_out = parameters;
    }

    void Init(int num_attributes_in) {
      num_parameters_ = num_attributes_in;
    }
};
}
}

#endif
