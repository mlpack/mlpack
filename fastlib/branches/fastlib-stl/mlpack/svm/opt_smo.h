/**
 * @author Hua Ouyang
 *
 * @file opt_smo.h
 *
 * This head file contains functions for performing Sequential Minimal Optimization (SMO)
 *
 * The algorithms in the following papers are implemented:
 *
 * 1. SMO and Working set selecting using 1st order expansion
 * @ARTICLE{Platt_SMO,
 * author = "J. C. Platt",
 * title = "{Fast Training of Support Vector Machines using Sequential Minimal Optimization}",
 * booktitle = "{Advances in Kernel Methods - Support Vector Learning}",
 * year = 1999,
 * publisher = "MIT Press"
 * }
 *
 * 2. Shrinkng and Caching for SMO
 * @ARTICLE{Joachims_SVMLIGHT,
 * author = "T. Joachims",
 * title = "{Making large-Scale SVM Learning Practical}",
 * booktitle = "{Advances in Kernel Methods - Support Vector Learning}",
 * year = 1999,
 * publisher = "MIT Press"
 * }
 *
 * 3. Working set selecting using 2nd order expansion
 * @ARTICLE{Fan_JMLR,
 * author = "R. Fan, P. Chen, C. Lin",
 * title = "{Working Set Selection using Second Order Information for Training Support Vector Machines}",
 * journal = "{Jornal of Machine Learning Research}",
 * year = 2005
 * }
 *
 * @see svm.h
 */
#ifndef __MLPACK_METHODS_SVM_OPT_SMO_H
#define __MLPACK_METHODS_SVM_OPT_SMO_H

#include <fastlib/fastlib.h>

// maximum # of interations for SMO training
const size_t MAX_NUM_ITER_SMO = 10000000;
// after # of iterations to do shrinking
const size_t SMO_NUM_FOR_SHRINKING = 1000;
// threshold that determines whether need to do unshrinking
const double SMO_UNSHRINKING_FACTOR = 10;
// threshold that determines whether an alpha is a SV or not
const double SMO_ALPHA_ZERO = 1.0e-7;
// for indefinite kernels
const double TAU = 1e-12;

const double SMO_ID_LOWER_BOUNDED = -1;
const double SMO_ID_UPPER_BOUNDED = 1;
const double SMO_ID_FREE = 0;

template <class T>
inline void swap(T& x, T& y) {
  T t = x;
  x = y;
  y = t;
}

template<typename TKernel>
class SMO {
 public:
  typedef TKernel Kernel;

 private:
  int learner_typeid_;
  int hinge_sqhinge_; // do L2-SVM or L1-SVM, default: L1

  size_t ct_iter_; /* counter for the number of iterations */
  size_t ct_shrinking_; /* counter for doing shrinking  */
  bool do_shrinking_; // 1(default): do shrinking after 1000 iterations; 0: don't do shrinking

  Kernel kernel_;
  size_t n_data_; /* number of data samples */
  size_t n_features_; /* # of features == # of row - 1, exclude the last row (for labels) */
  arma::mat *datamatrix_; /* alias for the data matrix, including labels in the last row */
  //Matrix datamatrix_samples_only_; /* alias for the data matrix excluding labels */

  arma::vec alpha_; /* the alphas, to be optimized */
  arma::vec alpha_status_; /*  ID_LOWER_BOUND (-1), ID_UPPER_BOUND (1), ID_FREE (0) */
  size_t n_sv_; /* number of support vectors */

  size_t n_alpha_; /* number of variables to be optimized */
  size_t n_active_; /* number of samples in the active set */
  std::vector<size_t> active_set_; /* list that stores the old indices of active alphas followed by inactive alphas. == old_from_new*/
  bool reconstructed_; /* indicator: where unshrinking has been carried out  */
  size_t i_cache_, j_cache_; /* indices for the most recently cached kernel value */
  double cached_kernel_value_; /* cache */

  std::vector<int> y_; /* list that stores "labels" */

  double bias_;

  arma::vec grad_; /* gradient value */
  arma::vec grad_bar_; /* gradient value when treat un-upperbounded variables as 0: grad_bar_i==C\sum_{j:a_j=C} y_i y_j K_ij */

  // parameters
  int budget_;
  double Cp_; // C_+, for SVM_C, y==1
  double Cn_; // C_-, for SVM_C, y==-1
  double C_;
  double inv_two_C_; // 1/2C
  double epsilon_; // for SVM_R
  int wss_; // working set selection scheme, 1 for 1st order expansion; 2 for 2nd order expansion
  size_t n_iter_; // number of iterations
  double accuracy_; // accuracy for stopping creterion
  double gap_; // for stopping criterion

 public:
  SMO() {}
  ~SMO() {}

  /**
   * Initialization for parameters
   */
  void InitPara(int learner_typeid,
                double Cp,
                double CnEpsilon,
                int hinge_sqhinge,
                int wss,
                size_t n_iter,
                double accuracy);

  void Train(int learner_typeid, arma::mat* dataset_in);

  Kernel& kernel() {
    return kernel_;
  }

  double Bias() const {
    return bias_;
  }

  void GetSV(std::vector<size_t> &dataset_index, std::vector<double> &coef, std::vector<bool> &sv_indicator);

 private:
  void LearnersInit_(int learner_typeid);

  int SMOIterations_();

  void ReconstructGradient_();

  bool TestShrink_(size_t i, double y_grad_max, double y_grad_min);

  void Shrinking_();

  bool WorkingSetSelection_(size_t &i, size_t &j);

  void UpdateGradientAlpha_(size_t i, size_t j);

  void CalcBias_();

  /**
   * Instead of C, we use C_+ and C_- to handle unbalanced data
   */
  double GetC_(size_t i) {
    return (y_[i] > 0 ? Cp_ : Cn_);
  }

  void UpdateAlphaStatus_(size_t i) {
    if (alpha_[i] >= GetC_(i)) {
      alpha_status_[i] = SMO_ID_UPPER_BOUNDED;
    } else if (alpha_[i] <= 0) {
      alpha_status_[i] = SMO_ID_LOWER_BOUNDED;
    } else { // 0 < alpha_[i] < C
      alpha_status_[i] = SMO_ID_FREE;
    }
  }

  inline bool IsUpperBounded(size_t i) {
    return alpha_status_[i] == SMO_ID_UPPER_BOUNDED;
  }
  inline bool IsLowerBounded(size_t i) {
    return alpha_status_[i] == SMO_ID_LOWER_BOUNDED;
  }

  /**
   * Calculate kernel values
   */
  double CalcKernelValue_(size_t ii, size_t jj);
};

// Include implementation.
#include "opt_smo_impl.h"

#endif
