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

    Vector coefficients_;

  private:

    void SolveSystem_(
      const ml::Dictionary &dictionary_in,
      const Vector &right_hand_side_in,
      Vector *solution_out) const;

    void ExtractWeightedTargetSubset_(
      const ml::Dictioanay &dictionary_in,
      const std::vector<double> &additional_kernel_values_in,
      Vector *target_subset_out) const;

    void ExtractTargetSubset_(
      const ml::Dictioanay &dictionary_in,
      Vector *target_subset_out) const;

    double QuadraticObjective_(
      const ml::Dictionary &dictionary_in,
      const std::vector<double> &kernel_values_in,
      bool for_coeffs) const;

    template<typename CovarianceType>
    void ComputeKernelValues_(
      const CovarianceType &covariance_in,
      const std::vector<int> &point_indices_in_dictionary,
      const Vector &point,
      Vector *kernel_values_out) const;

    template<typename CovarianceType>
    void ComputeKernelValues_(
      const CovarianceType &covariance_in,
      int candidate_index,
      std::vector<double> *kernel_values_out) const;

    void FillSquaredKernelMatrix_(
      int candidate_index,
      const Vector &kernel_values,
      double noise_level_in,
      std::vector<double> *new_column_vector_out,
      double *new_self_value_out) const;

    void FillKernelMatrix_(
      int candidate_index,
      const Vector &kernel_values,
      double noise_level_in,
      std::vector<double> *new_column_vector_out,
      double *new_self_value_out) const;

  public:

    double frobenius_norm_targets() const;

    SparseGreedyGprModel();

    void Init(const Matrix *dataset_in, const Vector *targets_in);

    template<typename CovarianceType>
    double AddOptimalPoint(
      const CovarianceType &covariance_in,
      double noise_level_in,
      const std::vector<int> &candidate_indices,
      bool for_coeffs);

    void FinalizeModel();

    template<typename CovarianceType>
    double PredictMean(
      const CovarianceType &covariance,
      const Vector &point) const;

    template<typename CovarianceType>
    double PredictVariance(
      const CovarianceType &covariance,
      const Vector &point) const;
};

class SparseGreedyGpr {
  private:

    const int random_subset_size_ = 60;

    const Matrix *dataset_;

    const Vector *targets_;

  private:

    bool Done_(
      double frobenius_norm_targets_in,
      double noise_level_in,
      double precision_in,
      double optimum_value,
      double optmum_value_for_error) const;

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
