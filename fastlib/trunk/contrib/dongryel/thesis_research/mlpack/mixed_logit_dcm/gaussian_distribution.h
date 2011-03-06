/** @file gaussian_distribution.h
 *
 *  The Gaussian distribution that can be used for mixed logit
 *  discrete choice model.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_GAUSSIAN_DISTRIBUTION_H
#define MLPACK_MIXED_LOGIT_DCM_GAUSSIAN_DISTRIBUTION_H

#include "mlpack/mixed_logit_dcm/distribution.h"

namespace mlpack {
namespace mixed_logit_dcm {

/** @brief The Gaussian distribution which can be used for mixed logit
 *         discrete choice model. This Gaussian distribution has the
 *         mean and the associated upper-triangular Cholesky factor.
 */
template<typename DCMTableType>
class GaussianDistribution:
  public virtual Distribution<DCMTableType> {
  private:

    /** @brief Stores the number of parameters. This is the number of
     *         parameters for the mean and the components of the upper
     *         triangular Cholesky factor.
     */
    int num_parameters_;

    int cholesky_factor_dimension_;

    int num_cholesky_factor_entries_;

    /** @brief Stores the cached solution to the upper triangular
     *         linear system solved for computing the attribute
     *         gradient with respect to parameter.
     */
    arma::vec cached_solution_;

  private:

    virtual void AttributeGradientWithRespectToParameterPrecompute_(
      const arma::vec &parameters, const arma::vec &beta_vector) const {

      // Set up the cholesky factor first.
      arma::mat cholesky_factor;
      cholesky_factor.zeros(
        cholesky_factor_dimension_, cholesky_factor_dimension_);
      int limit = cholesky_factor_dimension_;
      int add = cholesky_factor_dimension_ - 1;
      int row_num = 0;
      int start = 0;
      for(int i = 0; i < num_cholesky_factor_entries_; i++) {
        if(i == limit) {
          limit += add;
          add--;
          row_num++;
          start = i;
        }
        cholesky_factor.at(row_num, row_num + i - start) =
          parameters[cholesky_factor_dimension_ + i];
      }

      // Solve. The right hand side is basically beta_vector shifted
      // by the means.
      arma::vec mean_vector(
        const_cast<arma::vec &>(parameters).memptr(),
        beta_vector.n_elem, false);
      arma::vec right_hand_side = beta_vector - mean_vector;
      const_cast <
      mlpack::mixed_logit_dcm::GaussianDistribution<DCMTableType> >(
        this)->cached_solution_ =
          arma::solve(
            arma::trimatu(cholesky_factor), right_hand_side);
    }

  public:

    /** @brief The default constructor.
     */
    GaussianDistribution() {
      num_parameters_ = 0;
      cholesky_factor_dimension_ = 0;
      num_cholesky_factor_entries_ = 0;
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
