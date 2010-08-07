/** @file sg_gp_regression.h
 *
 *  @brief An prototype of "Sparse Greedy Gaussian Process regression" by
 *         Smola et al.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef ML_GP_REGRESSION_SG_GP_REGRESSION_H
#define ML_GP_REGRESSION_SG_GP_REGRESSION_H

#include "fastlib/fastlib.h"

namespace ml {
namespace gp_regression {
class SparseGreedyGprModel {
  private:

    const Matrix *dataset_;

    const Vector *targets_;

    std::vector<int> subset_;

    std::vector<int> subset_for_error_;

    std::vector< std::vector<double> > squared_kernel_matrix_;

    std::vector< std::vector<double> > kernel_matrix_;

    std::vector< std::vector<double> > inverse_;

    std::vector< std::vector<double> > inverse_for_error_;

    std::vector<double> subset_coefficients_;

  private:
    void SetupMatrix_(
      const std::vector< std::vector<double> > &matrix_in,
      Matrix *matrix_out) const;

  public:
    SparseGreedyGprModel();

    void Init(const Matrix *dataset_in, const Vector *targets_in);

    void AddOptimalPoint(const std::vector<int> &candidate_indices);

    void AddOptimalPointForError(const std::vector<int> &candidate_indices);
};

class SparseGreedyGpr {
  private:
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

    void Compute(
      double noise_level_in,
      double precision_in,
      SparseGreedyGprModel *model_out);
};
};
};

#endif
