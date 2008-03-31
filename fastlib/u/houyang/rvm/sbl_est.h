/**
 * @author Hua Ouyang
 *
 * @file sbl_est.h
 *
 * This head file contains functions for performing Sparse Bayesian Learning for Relevance Vector Machine
 *
 * The algorithms in the following papers are implemented (the notations follow Bishop_PRML):
 *
 * @ARTICLE{Tipping_RVM_JMLR,
 * author = "M. E. Tipping",
 * title = "{Sparse Bayesian Learning and the Relevance Vector machine}"
 * journal = "{Journal of Machine Learning Research}",
 * year = 2001
 * }
 *
 * @INPROCEEDINGS{Tipping_RVM_NIPS,
 * author = "M. E. Tipping",
 * title = "{The Relevance Vector machine}",
 * booktitle = NIPS,
 * number = 12,
 * year = 2000
 * }
 *
 * @BOOK{ Bishop_PRML,
 * author = "C.M. Bishop",
 * title = "{Pattern Recognition and Machine Learning}",
 * publisher = "Springer",
 * year = 2006
 * }
 *
 * @see rvm.h
 */

#ifndef U_RVM_SBL_EST_H
#define U_RVM_SBL_EST_H

#include "fastlib/fastlib.h"

/* TODO: I don't actually want these to be public */
// Prune basis function when its alpha is greater than this
const double ALPHA_MAX = 1.0e12;
// Iteration number during training where we switch to 'analytic pruning'
const index_t PRUNE_POINT = 50;

template<typename TKernel>
class SBL_EST {
  FORBID_ACCIDENTAL_COPIES(SBL_EST);

 public:
  typedef TKernel Kernel;

 private:
  Kernel kernel_;
  const Dataset *dataset_;
  index_t n_data_; /* number of data samples */
  index_t n_rv_; /* number of relevance vectors */
  Vector error_;

 public:
  SBL_EST() {}
  ~SBL_EST() {}

  /**
   * Initialization
   */
  void Init(double c_in, int budget_in) {
    // init number of relevance vectors
    n_rv_ = 0;
  }

  void Train(int learner_typeid, const Dataset* dataset_in, double* alpha, double* beta, index_t max_iter);

  Kernel& kernel() {
    return kernel_;
  }

  double threshold() const {
    return thresh_;
  }

  //index_t num_rv() const {
  //  return n_rv_;
  //}

  void GetRVM(ArrayList<index_t> &dataset_index, ArrayList<double> &coef, ArrayList<bool> &sv_indicator);

 private:
  double Error_(index_t i) const {
    double val;
    if (!IsBound_(alpha_[i])) {
      val = error_[i];
      VERBOSE_MSG(0, "error values %f and %f", error_[i], Evaluate_(i) - GetLabelSign_(i));
    } else {
      val = CalculateError_(i);
    }
    return val;
  }

  double CalculateError_(index_t i) const {
    return Evaluate_(i) - GetLabelSign_(i);
  }
  
  double Evaluate_(index_t i) const;

  double EvalKernel_(index_t i, index_t j) const {
    return kernel_cache_sign_.get(i, j) * (GetLabelSign_(i) * GetLabelSign_(j));
  }
  
  /**
   * Calculate kernel values
   */
  void CalcKernels_() {
    kernel_cache_sign_.Init(n_data_, n_data_);
    fprintf(stderr, "Kernel Start\n");
    for (index_t i = 0; i < n_data_; i++) {
      for (index_t j = 0; j < n_data_; j++) {
        Vector v_i;
        GetVector_(i, &v_i);
        Vector v_j;
        GetVector_(j, &v_j);
        double k = kernel_.Eval(v_i, v_j);
        kernel_cache_sign_.set(j, i, k * GetLabelSign_(i) * GetLabelSign_(j));
      }
    }
    fprintf(stderr, "Kernel Stop\n");
  }
};

/**
* RVM training, for both regression and classification
*
* @param: input 2-classes data matrix with labels (1,-1) in the last row
*/
template<typename TKernel>
void SBL_EST<TKernel>::Train(int learner_typeid, const Dataset* dataset_in, double* alpha, double* beta, int max_iter) {
  for (index_t i =0; i< max_iter; i++) {
    // Prune large values of alpha

    // Compute marginal likelihood (the objective function to be maximized)

    // Iterative Re-estimation for parameters (alpha and beta)

    // Termination check
  }
}

/* Get RVM results:weights, indecies of RVs
*
* @param: sample indices of the training (sub)set in the total training set
* @param: support vector coefficients: alpha*y
* @param: bool indicators  FOR THE TRAINING SET: is/isn't a support vector
*/
template<typename TKernel>
void SBL_EST<TKernel>::GetRVM(ArrayList<index_t> &dataset_index, ArrayList<double> &coef, ArrayList<bool> &sv_indicator) {
  for (index_t i = 0; i < n_data_; i++) {
    if (alpha_[i] != 0) { /* support vectors */
      *coef.AddBack() = alpha_[i] * GetLabelSign_(i);
      sv_indicator[dataset_index[i]] = true;
      n_sv_++;
    }
    else {
      *coef.AddBack() = 0;
    }
  }
}

#endif
