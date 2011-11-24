/**
 * @file nnsmo.hpp
 *
 * The non-negative SMO algorithm.
 */
#ifndef __MLPACK_METHODS_NNSVM_NNSMO_HPP
#define __MLPACK_METHODS_NNSVM_NNSMO_HPP

#include <mlpack/core.h>

namespace mlpack {
namespace nnsvm {

/* TODO: I don't actually want these to be public */
/* but sometimes we should provide freedoms for our advanced users */
const double NNSMO_ZERO = 1.0e-8;
const double NNSMO_TOLERANCE = 1.0e-3;

template<typename TKernel>
class NNSMO
{
 public:
  typedef TKernel Kernel;

 private:
  arma::mat kernel_cache_sign_;
  Kernel kernel_;
  size_t n_data_; // number of data samples
  arma::mat dataset_; // alias for the data matrix
  arma::vec alpha_; // the alphas, to be optimized
  arma::vec error_; // the error cache
  double thresh_; // negation of the intercept
  double c_;
  size_t budget_;
  double sum_alpha_;

  size_t n_feature_; // number of data features
  double w_square_sum_; // square sum of the weight vector
  arma::vec VTA_; //
  double eps_; // the tolerace of progress on alpha values
  size_t max_iter_; // the maximum iteration, termination criteria

 public:
  NNSMO() {}
  ~NNSMO() {}

  /**
   * Initializes an NNSMO problem.
   *
   * You must initialize separately the kernel.
   */
  void Init(const arma::mat& dataset_in, double c_in, size_t budget_in,
            double eps_in, size_t max_iter_in)
  {
    c_ = c_in;

    dataset_ = dataset_in;

    n_data_ = dataset_.n_cols;
    budget_ = std::min(budget_in, (size_t) n_data_);

    alpha_.zeros(n_data_);
    sum_alpha_ = 0;

    error_.zeros(n_data_);
    for(size_t i = 0; i < n_data_; i++)
    {
      error_[i] -= GetLabelSign_(i);
    }

    thresh_ = 0;

    n_feature_ = dataset_.n_rows - 1;
    VTA_.zeros(n_feature_);
    eps_ = eps_in;
    max_iter_ = max_iter_in;
  }

  void Train();

  double threshold() const
  {
    return thresh_;
  }

  void GetNNSVM(arma::mat& support_vectors, arma::vec& alpha, arma::vec& w)
      const;

 private:
  size_t TrainIteration_(bool examine_all);

  bool TryChange_(size_t j);

  double CalculateDF_(size_t i, size_t j, double error_j);

  bool TakeStep_(size_t i, size_t j, double error_j);

  double FixAlpha_(double alpha) const
  {
    if (alpha < NNSMO_ZERO)
      alpha = 0;
    else if (alpha > c_ - NNSMO_ZERO)
      alpha = c_;

    return alpha;
  }

  bool IsBound_(double alpha) const
  {
    return (alpha <= 0) || (alpha >= c_);
  }

  // labels: the last row of the data matrix, 0 or 1
  int GetLabelSign_(size_t i) const
  {
    return (dataset_(dataset_.n_rows - 1, i) != 0) ? 1 : -1;
  }

  void GetVector_(size_t i, arma::vec& v) const
  {
    v = arma::vec((double*) dataset_.colptr(i), dataset_.n_rows - 1, false,
        true); // manual ugly constructor
  }

  double Error_(size_t i) const
  {
    return error_[i];
  }

  double Evaluate_(size_t i) const;

  double EvalKernel_(size_t i, size_t j) const
  {
    return kernel_cache_sign_(i, j) * (GetLabelSign_(i) * GetLabelSign_(j));
  }

  void CalcKernels_()
  {
    kernel_cache_sign_.set_size(n_data_, n_data_);
    fprintf(stderr, "Kernel Start\n");
    for (size_t i = 0; i < n_data_; i++)
    {
      for (size_t j = 0; j < n_data_; j++)
      {
        arma::vec v_i;
        GetVector_(i, v_i);
        arma::vec v_j;
        GetVector_(j, v_j);
        double k = kernel_.Evaluate(v_i, v_j);
        kernel_cache_sign_(j, i) = k * GetLabelSign_(i) * GetLabelSign_(j);
      }
    }
    fprintf(stderr, "Kernel Stop\n");
  }
};

}; // namespace nnsvm
}; // namespace mlpack

#include "nnsmo_impl.hpp"

#endif // __MLPACK_METHODS_NNSVM_NNSMO_HPP
