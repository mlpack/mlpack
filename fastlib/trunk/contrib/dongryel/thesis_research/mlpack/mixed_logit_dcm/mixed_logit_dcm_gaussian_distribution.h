/** @file mixed_logit_dcm_gaussian_distribution.h
 *
 *  The Gaussian distribution that can be used for mixed logit
 *  discrete choice model.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_GAUSSIAN_DISTRIBUTION_H
#define MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_GAUSSIAN_DISTRIBUTION_H

#include "mlpack/mixed_logit_dcm/mixed_logit_dcm_distribution.h"

namespace mlpack {
namespace mixed_logit_dcm {

/** @brief The Gaussian distribution which can be used for mixed logit
 *         discrete choice model.
 */
template<typename DCMTableType>
class MixedLogitDCMGaussianDistribution:
  public virtual MixedLogitDCMDistribution<DCMTableType> {
  private:
    int num_parameters_;

  public:

    MixedLogitDCMGaussianDistribution() {
      num_parameters_ = 0;
    }

    int num_parameters() const {
      return num_parameters_;
    }

    /** @brief Returns the (row, col)-th entry of
     *         $\frac{\partial}{\partial \theta} \beta^{\nu}(\theta)$
     */
    double AttributeGradientWithRespectToParameter(
      const arma::vec &parameters, int row_index, int col_index) const {

    }

    void DrawBeta(
      const arma::vec &parameters, arma::vec *beta_out) const {

    }

    void Init(const std::string &file_name) const {

    }
};
}
}

#endif
