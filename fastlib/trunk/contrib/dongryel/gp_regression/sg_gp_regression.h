/** @file sg_gp_regression.h
 *
 *  @brief An prototype of "Sparse Greedy Gaussian Process regression" by
 *         Smola et al.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_GP_REGRESSION_SG_GP_REGRESSION_H
#define MLPACK_GP_REGRESSION_SG_GP_REGRESSION_H

#include "fastlib/fastlib.h"
#include "dictionary_dev.h"

namespace ml {
namespace gp_regression {
class SparseGreedyGprModel {
  private:

    const Matrix *dataset_;

    const Vector *targets_;

    double frobenius_norm_targets_;

    std::vector< std::vector<double> > kernel_matrix_columns_;

    ml::Dictionary dictionary_;

    ml::Dictionary dictionary_for_error_;

  private:

    double QuadraticObjective_(const ml::Dictionary &dictionary_in) const;

    template<typename CovarianceType>
    void ComputeKernelValues_(
      const CovarianceType &covariance_in,
      int candidate_index,
      std::vector<double> *kernel_values_out) const;

    void FillSquaredKernelMatrix_(
      int candidate_index,
      const Vector &kernel_values,
      double noise_level_in,
      Vector *new_column_vector_out,
      double *new_self_value_out) const;

    void FillKernelMatrix_(
      int candidate_index,
      const Vector &kernel_values,
      double noise_level_in,
      Vector *new_column_vector_out,
      double *new_self_value_out) const;

  public:
    SparseGreedyGprModel();

    void Init(const Matrix *dataset_in, const Vector *targets_in);

    template<typename CovarianceType>
    void AddOptimalPoint(
      const CovarianceType &covariance_in,
      double noise_level_in,
      const std::vector<int> &candidate_indices,
      bool for_coeffs);
};

class SparseGreedyGpr {
  private:

    const int random_subset_size_ = 60;

    const Matrix *dataset_;

    const Vector *targets_;

  private:

    void InitInactiveSet_(std::vector<int> *inactive_set_out) const;

    void ChooseRandomSubset_(
      const std::vector<int> &inactive_set,
      int subset_size,
      std::vector<int> *subset_out) const;

  public:

    SparseGreedyGpr();

    void Init(const Matrix &dataset_in, const Vector &targets_in);

    template<typename CovarianceType>
    void Compute(
      const CovarianceType &covariance_in,
      double noise_level_in,
      double precision_in,
      SparseGreedyGprModel *model_out);
};
};
};

#endif
