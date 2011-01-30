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
 *         discrete choice model. This Gaussian distribution has the
 *         mean and the standard deviation per each dimension, for a
 *         total of twice the number of attribute values.
 */
template<typename DCMTableType>
class MixedLogitDCMGaussianDistribution:
  public virtual MixedLogitDCMDistribution<DCMTableType> {
  private:

    /** @brief Stores the number of parameters. This is the number of
     *         parameters for the mean and the variance in each
     *         dimension.
     */
    int num_parameters_;

  public:

    /** @brief The default constructor.
     */
    MixedLogitDCMGaussianDistribution() {
      num_parameters_ = 0;
    }

    /** @brief Returns the number of parameters.
     */
    int num_parameters() const {
      return num_parameters_;
    }

    /** @brief Returns the (row, col)-th entry of
     *         $\frac{\partial}{\partial \theta} \beta^{\nu}(\theta)$
     */
    double AttributeGradientWithRespectToParameter(
      const arma::vec &parameters, const arma::vec &beta_vector,
      int row_index, int col_index) const {

      // The number of attributes, which is assumed to be half the
      // number of parameters.
      int num_attributes = num_parameters_ / 2;

      // Upper half.
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
        if() {
        }
        else {
          return 0;
        }
      }
    }

    /** @brief Draws a new $\beta$ from the Gaussian distribution.
     */
    void DrawBeta(
      const arma::vec &parameters, arma::vec *beta_out) const {


    }

    void Init(const std::string &file_name) const {

    }
};
}
}

#endif
