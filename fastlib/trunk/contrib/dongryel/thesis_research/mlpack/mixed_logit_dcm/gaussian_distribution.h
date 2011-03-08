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
class GaussianDistribution:
  public virtual Distribution {
  private:

    /** @brief Stores the number of parameters. This is the number of
     *         parameters for the mean and the components of the upper
     *         triangular Cholesky factor.
     */
    int num_parameters_;

    int cholesky_factor_dimension_;

    int num_cholesky_factor_entries_;

    std::vector<int> nonzero_column_indices_;

    std::vector<int> start_indices_;

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
      mlpack::mixed_logit_dcm::GaussianDistribution * >(
        this)->cached_solution_ =
          arma::solve(
            cholesky_factor, right_hand_side);
    }

  public:

    /** @brief The default constructor.
     */
    GaussianDistribution() {
      cholesky_factor_dimension_ = 0;
      num_cholesky_factor_entries_ = 0;
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

        // Locate the non-zero column for this row. If the col_index
        // matches this index, then return a non-zero value.
        int nonzero_column_index = nonzero_column_indices_[row_index];
        if(nonzero_column_index == col_index) {
          return cached_solution_[
                   row_index - start_indices_[row_index] +
                   nonzero_column_index];
        }
        else {
          return 0.0;
        }
      }
    }

    /** @brief Draws a new $\beta$ from the Gaussian distribution.
     */
    void DrawBeta(
      const arma::vec &parameters, arma::vec *beta_out) const {


    }

    void Init(int num_parameters_in) {
      num_parameters_ = num_parameters_in;
      cholesky_factor_dimension_ = (-3 + sqrt(9 + 8 * num_parameters_)) / 2;
      num_cholesky_factor_entries_ =
        cholesky_factor_dimension_ * (cholesky_factor_dimension_ + 1) / 2;
      nonzero_column_indices_.resize(num_parameters_);
      std::fill(
        nonzero_column_indices_.begin(), nonzero_column_indices_.end(), 0);
      start_indices_.resize(num_parameters_);
      std::fill(start_indices_.begin(), start_indices_.end(), 0);

      // Fill out the non-zero column indices for the gradient of the
      // attribute with respect to parameter.
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
        nonzero_column_indices_[cholesky_factor_dimension_ + i] = row_num;
        start_indices_[cholesky_factor_dimension_ + i] =
          cholesky_factor_dimension_ + start;
      }
    }
};
}
}

#endif
