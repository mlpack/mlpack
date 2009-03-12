/**
 * @author Hua Ouyang
 *
 * @file smo.h
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
 * 4. TODO: budget L2 SVM
 * @INPROCEEDINGS{Budge_SVM,
 * author = "O. Dekel and Y. Singer",
 * title = "{Support Vector Machines on a Budget}",
 * booktitle = NIPS,
 * number = 19,
 * year = 2006
 * }
 *
 * @see svm.h
 */

#ifndef U_SVM_SMO_H
#define U_SVM_SMO_H

#include "fastlib/fastlib.h"

// the tolerance for determing the optimality
const double SMO_OPT_TOLERANCE = 1.0e-6; // changed from 1.0e-4 by NISHANT for temporary check
// maximum # of interations for SMO training
const index_t MAX_NUM_ITER = 10000;
// after # of iterations to do shrinking
const index_t NUM_FOR_SHRINKING = 1000;
// threshold that determines whether need to do unshrinking
const double SMO_UNSHRINKING_TOLERANCE = 10 * SMO_OPT_TOLERANCE;
// threshold that determines whether an alpha is a SV or not
const double SMO_ALPHA_ZERO = 1.0e-4;
// for indefinite kernels
const double TAU = 1e-12;

const double ID_LOWER_BOUNDED = -1;
const double ID_UPPER_BOUNDED = 1;
const double ID_FREE = 0;

template<typename TKernel>
class SMO {
  FORBID_ACCIDENTAL_COPIES(SMO);

 public:
  typedef TKernel Kernel;

 private:
  int learner_typeid_;
  index_t ct_iter_; /* counter for the number of iterations */
  index_t ct_shrinking_; /* counter for doing shrinking  */

  Kernel kernel_;
  const Dataset *dataset_;
  index_t n_data_; /* number of data samples */
  index_t n_features_; /* # of features == # of row - 1, exclude the last row (for labels) */
  Matrix datamatrix_; /* alias for the data matrix */

  Vector alpha_; /* the alphas, to be optimized */
  Vector alpha_status_; /*  ID_LOWER_BOUND (-1), ID_UPPER_BOUND (1), ID_FREE (0) */
  
  index_t n_alpha_; /* number of variables to be optimized */
  index_t n_active_; /* number of samples in the active set */
  ArrayList<index_t> active_set_; /* list that stores the indices of active alphas */
  bool unshrinked_; /* indicator: where unshrinking has be carried out  */
  index_t i_cache_, j_cache_; /* indices for the most recently cached kernel value */
  double cached_kernel_value_; /* cache */

  ArrayList<int> y_; /* list that stores "labels" */

  double bias_;
  index_t n_sv_; /* number of support vectors */

  Vector grad_; /* gradient value */
  Vector grad_bar_; /* gradient value when treat free variables as 0 */

  // parameters
  //int budget_; // commented out by NISHANT because not intialized
  double Cp_; // C_+, for SVM_C, y==1
  double Cn_; // C_-, for SVM_C, y==-1
  double epsilon_; // for SVM_R
  int wss_; // working set selection scheme, 1 for 1st order expansion; 2 for 2nd order expansion

 public:
  SMO() {}
  ~SMO() {}

  /**
   * Initialization for parameters
   */
  void InitPara(int learner_typeid, ArrayList<double> &param_) {
    // init parameters
    //budget_ = (int)param_[0]; // commented out by NISHANT because param_[0] not initialized
    if (learner_typeid == 0) { // SVM_C
      Cp_ = param_[1];
      Cn_ = param_[2];
      wss_ = (int) param_[3];
    }
    else if (learner_typeid == 1) { // SVM_R
      Cp_ = param_[1];
      Cn_ = Cp_;
      epsilon_ = param_[2];
      wss_ = (int) param_[3];
    }
  }

  void Train(int learner_typeid, const Dataset* dataset_in);

  Kernel& kernel() {
    return kernel_;
  }

  double Bias() const {
    return bias_;
  }

  void GetSVM(ArrayList<index_t> &dataset_index, ArrayList<double> &coef, ArrayList<bool> &sv_indicator);

 private:
  void LearnersInit_(int learner_typeid);

  int TrainIteration_();

  void ReconstructGradient_(int learner_typeid);
  
  void Shrinking_();

  bool WorkingSetSelection_(index_t &i, index_t &j);

  void UpdatingGradientAlpha_(index_t i, index_t j);

  void CalcBias_();

  /*  void GetVector_(index_t i, Vector *v) const {
    datamatrix_.MakeColumnSubvector(i, 0, datamatrix_.n_rows()-1, v);
  }
  */

  /**
   * Instead of C, we use C_+ and C_- to handle unbalanced data
   */
  double GetC_(index_t i) {
    return (y_[i] > 0 ? Cp_ : Cn_);
  }

  void UpdateAlphaStatus_(index_t i) {
    if (alpha_[i] >= GetC_(i)) {
      alpha_status_[i] = ID_UPPER_BOUNDED;
    }
    else if (alpha_[i] <= 0) {
      alpha_status_[i] = ID_LOWER_BOUNDED;
    }
    else { // 0 < alpha_[i] < C
      alpha_status_[i] = ID_FREE;
    }
  }

  bool IsUpperBounded(index_t i) {
    return alpha_status_[i] == ID_UPPER_BOUNDED;
  }
  bool IsLowerBounded(index_t i) {
    return alpha_status_[i] == ID_LOWER_BOUNDED;
  }

  /**
   * Calculate kernel values
   */
  double CalcKernelValue_(index_t i, index_t j) {
    // the alpha indices have been swaped in shrinking processes
    i = active_set_[i];
    j = active_set_[j];

    // for SVM_R where n_alpha_==2*n_data_
    if (learner_typeid_ == 1) {
      i = i >= n_data_ ? (i-n_data_) : i;
      j = j >= n_data_ ? (j-n_data_) : j;
    }

    // Check cache
    //if (i == i_cache_ && j == j_cache_) {
    //  return cached_kernel_value_;
    //}

    double *v_i, *v_j;
    v_i = datamatrix_.GetColumnPtr(i);
    v_j = datamatrix_.GetColumnPtr(j);

    // Do Caching. Store the recently caculated kernel values.
    //i_cache_ = i;
    //j_cache_ = j;
    cached_kernel_value_ = kernel_.Eval(v_i, v_j, n_features_);
    return cached_kernel_value_;
  }
};


/**
 * Initialization according to different SVM learner types
 *
 * @param: learner type id 
 */
template<typename TKernel>
void SMO<TKernel>::LearnersInit_(int learner_typeid) {
  index_t i;
  learner_typeid_ = learner_typeid;
  
  if (learner_typeid_ == 0) { // SVM_C
    n_alpha_ = n_data_;

    alpha_.Init(n_alpha_);
    alpha_.SetZero();

    // initialize gradient
    grad_.Init(n_alpha_);
    for (i=0; i<n_alpha_; i++) {
      grad_[i] = 1;
    }

    y_.Init(n_alpha_);
    for (i = 0; i < n_alpha_; i++) {
      y_[i] = datamatrix_.get(datamatrix_.n_rows()-1, i) > 0 ? 1 : -1;
      }
  }
  else if (learner_typeid_ == 1) { // SVM_R
    n_alpha_ = 2 * n_data_;

    alpha_.Init(2 * n_alpha_);
    alpha_.SetZero();

    // initialize gradient
    grad_.Init(n_alpha_);
    y_.Init(n_alpha_);
    for (i = 0; i < n_data_; i++) {
      y_[i] = 1; // -> alpha_i
      y_[i + n_data_] = -1; // -> alpha_i^*
      grad_[i] = epsilon_ - datamatrix_.get(datamatrix_.n_rows()-1, i);
      grad_[i + n_data_] = epsilon_ + datamatrix_.get(datamatrix_.n_rows()-1, i);
    }
  }
  else if (learner_typeid_ == 2) { // SVM_DE
  }
  
  // initialize active set
  n_active_ = n_alpha_;
  active_set_.Init(n_active_);
  for (i=0; i<n_active_; i++) {
      active_set_[i] = i;
  }
}


/**
* Reconstruct inactive elements of G from G_bar and free variables 
*
* @param: learner type id
*/
template<typename TKernel>
void SMO<TKernel>::ReconstructGradient_(int learner_typeid) {
  index_t i, j;
  if (n_active_ == n_alpha_)
    return;
  if (learner_typeid == 0) { // SVM_C
    for (i=n_active_; i<n_alpha_; i++) {
      grad_[i] = grad_bar_[i] + 1;
    }
  }
  else if (learner_typeid == 1) { // SVM_R
    for (i=n_active_; i<n_alpha_; i++) {
      j = i >= n_data_ ? (i-n_data_) : i;
      grad_[j] = grad_bar_[j] + datamatrix_.get(datamatrix_.n_rows()-1, j) - epsilon_;
    }
  }

  for (i=0; i<n_active_; i++) {
    if (alpha_status_[i] == ID_FREE) {
      for (j=n_active_; j<n_alpha_; j++) {
	grad_[j] += alpha_[i] * CalcKernelValue_(i,j);
      }
    }
  }
}


/**
 * Do Shrinking. Temporarily remove alphas (from the active set) that are 
 * unlikely to be selected in the working set, since they have reached their 
 * lower/upper bound.
 * 
 */
template<typename TKernel>
void SMO<TKernel>::Shrinking_() {
  index_t t;
  double yg;

  // find grad_max and grad_min
  double grad_max = -INFINITY;
  double grad_min =  INFINITY;
  for (t=0; t<n_active_; t++) { // find argmax(y*grad), t\in I_up
    if (y_[t] == 1) {
      if (!IsUpperBounded(t)) // t\in I_up, y==1: y[t]alpha[t] <= C
	if (grad_[t] >= grad_max) { // y==1
	  grad_max = grad_[t];
	}
      if (!IsLowerBounded(t)) // t\in I_down, y==1: y[t]alpha[t] >= 0
	if (grad_[t] <= grad_min) { // y==1
	  grad_min = grad_[t];
	}
    }
    else { // y[t] == -1
      if (!IsLowerBounded(t)) // t\in I_up, y==-1: y[t]alpha[t] <= 0
	if (-grad_[t] >= grad_max) { // y==-1
	  grad_max = -grad_[t];
	}
      if (!IsUpperBounded(t)) // t\in I_down, y==-1: y[t]alpha[t] >= -C
	if (-grad_[t] <= grad_min) { // y==-1
	  grad_min = -grad_[t];
	}
    }
  }

  // find the alpha to be shrunk
  for (t=0; t<n_active_; t++) {
    // Shrinking: put inactive alphas behind the active set
    yg = y_[t] * grad_[t];
    if ( yg < grad_min || yg > grad_max ) { // this alpha need be shrunk
      n_active_ --;
      while (n_active_ > t) {
	yg = y_[n_active_] * grad_[n_active_];
	if ( yg >= grad_min && yg <= grad_max ) { // this alpha need not be shrunk
	  active_set_[t] = n_active_; // swap indices
	  active_set_[n_active_] = t;
	  break;
	}
	n_active_ --;
      }
    }
  }

  // determine whether need to do Unshrinking
  if ( unshrinked_ || grad_max - grad_min <= SMO_UNSHRINKING_TOLERANCE ) {
    // Unshrinking: put shrinked alphas back to active set
    ReconstructGradient_(learner_typeid_);
    for ( t=n_alpha_-1; t>n_active_; t-- ) {
      yg = y_[t] * grad_[t];
      if (yg >= grad_min && yg <= grad_max) { // this alpha need be unshrunk
	while (n_active_ < t) {
	  yg = y_[n_active_] * grad_[n_active_];
	  if ( yg < grad_min || yg > grad_max ) { // this alpha need not be unshrunk
	    active_set_[t] = n_active_; // swap indices
	    active_set_[n_active_] = t;
	    break;
	  }
	  n_active_ ++;
	}
	n_active_ ++;
      }
    }
    unshrinked_ = true; // indicator: unshrinking has been carried out in this round
  }

}


/**
* SMO training for 2-classes
*
* @param: input 2-classes data matrix with labels (1,-1) in the last row
*/
template<typename TKernel>
void SMO<TKernel>::Train(int learner_typeid, const Dataset* dataset_in) {
  index_t i,j;
  /* general learner-independent initializations */
  dataset_ = dataset_in;
  datamatrix_.Alias(dataset_->matrix());
  n_data_ = datamatrix_.n_cols();
  n_features_ = datamatrix_.n_rows() - 1;

  //budget_ = min(budget_, n_data_);  // commented out by NISHANT because not initialized
  bias_ = 0.0;
  n_sv_ = 0;
  unshrinked_ = false;
  i_cache_ = -1; j_cache_ = -1;
  cached_kernel_value_ = INFINITY;

  /* learners initialization */
  LearnersInit_(learner_typeid);
  
  alpha_status_.Init(n_alpha_);
  for (i=0; i<n_alpha_; i++)
    UpdateAlphaStatus_(i);

  // initialize gradient_bar
  grad_bar_.Init(n_alpha_);
  grad_bar_.SetZero();
  for(i=0; i<n_alpha_; i++) {
    if(!IsLowerBounded(i))
      {
	for(j=0; j<n_alpha_; j++)
	  grad_[j] += alpha_[i] * CalcKernelValue_(i,j);
	if(IsUpperBounded(i))
	  for(j=0; j<n_alpha_; j++)
	    grad_bar_[j] += GetC_(i) * CalcKernelValue_(i,j);
      }
  }

  ct_iter_ = 0;
  ct_shrinking_ = min(n_data_, NUM_FOR_SHRINKING) + 1;
  /* Begin SMO iterations */
  int stop_condition = 0;
  while (1) {
    VERBOSE_GOT_HERE(0);
    
    /* for every min(n_data_, 1000) iterations, do shrinking */
    if (--ct_shrinking_ == 0) {
      Shrinking_();
      ct_shrinking_ = min(n_data_, NUM_FOR_SHRINKING);
    }

    // Find working set, check stopping criterion, update gradient and alphas
    stop_condition = TrainIteration_();
    // termination check, stop_condition==1 or 2->terminate
    if (stop_condition == 1) // optimality reached
      break;
    else if (stop_condition == 2) {// max num of iterations exceeded
      fprintf(stderr, "Max # of iterations (%d) exceeded !!!\n", MAX_NUM_ITER);
      break;
    }
  }
}

/**
* SMO training iterations
* 
* @return: stopping condition id
*/
template<typename TKernel>
int SMO<TKernel>::TrainIteration_() {
  ct_iter_ ++;
  index_t i,j;
  if (WorkingSetSelection_(i,j) == true) {
    ReconstructGradient_(learner_typeid_); // reconstruct the whole gradient
    n_active_ = 1;
    if (WorkingSetSelection_(i,j) == true) { // optimality reached
      return 1;
    }
    else {
      ct_shrinking_ = 1; // do shrinking in the next iteration
      return 0;
    }
  }
  else if (ct_iter_ >= MAX_NUM_ITER) { // max num of iterations exceeded
    return 2;
  }
  else{ // update gradient and alphas, and continue iterations
    UpdatingGradientAlpha_(i, j);
    return 0;
  }
}

/**
* Try to find a working set (i,j). Both 1st order and 2nd order approximations of 
* the objective function Z(\alpha+\lambda u_ij)-Z(\alpha) are implemented.
*
* @param: working set (i, j)
*
* @return: indicator of whether the optimal solution is reached (true:reached)
*/
template<typename TKernel>
bool SMO<TKernel>::WorkingSetSelection_(index_t &out_i, index_t &out_j) {
  double grad_max = -INFINITY;
  double grad_min =  INFINITY;
  int idx_i = 0;
  int idx_j = 0;
  
  // Find i using maximal violating pair scheme
  index_t t;
  for (t=0; t<n_active_; t++) { // find argmax(y*grad), t\in I_up
    if (y_[t] == 1) {
      if (!IsUpperBounded(t)) // t\in I_up, y==1: y[t]alpha[t] <= C
	if (grad_[t] >= grad_max) { // y==1
	  grad_max = grad_[t];
	  idx_i = t;
	}
    }
    else { // y[t] == -1
      if (!IsLowerBounded(t)) // t\in I_up, y==-1: y[t]alpha[t] <= 0
	if (-grad_[t] >= grad_max) { // y==-1
	  grad_max = -grad_[t];
	  idx_i = t;
	}
    }
  }
  out_i = idx_i; // i found

  /*  Find j using maximal violating pair scheme (1st order approximation) */
  if (wss_ == 1) {
    for (t=0; t<n_active_; t++) { // find argmin(y*grad), t\in I_down
      if (y_[t] == 1) {
	if (!IsLowerBounded(t)) // t\in I_down, y==1: y[t]alpha[t] >= 0
	  if (grad_[t] <= grad_min) { // y==1
	    grad_min = grad_[t];
	    idx_j = t;
	  }
      }
      else { // y[t] == -1
	if (!IsUpperBounded(t)) // t\in I_down, y==-1: y[t]alpha[t] >= -C
	  if (-grad_[t] <= grad_min) { // y==-1
	    grad_min = -grad_[t];
	    idx_j = t;
	  }
      }
    }
    out_j = idx_j; // i found
  }
  /* Find j using 2nd order working set selection scheme; need to calc kernels, but faster convergence */
  else if (wss_ == 2) {
    double K_ii = CalcKernelValue_(out_i, out_i);
    double opt_gain_max = -INFINITY;
    double grad_diff;
    double quad_kernel;
    double opt_gain = -INFINITY;
    for (t=0; t<n_active_; t++) {
      double K_it = CalcKernelValue_(out_i, t);
      double K_tt = CalcKernelValue_(t, t);
      if (y_[t] == 1) {
	if (!IsLowerBounded(t)) { // t\in I_down, y==1: y[t]alpha[t] >= 0
	  // calculate grad_min for Stopping Criterion
	  if (grad_[t] <= grad_min) // y==1
	    grad_min = grad_[t];
	  // find j
	  grad_diff = grad_max - grad_[t]; // max(y_i*grad_i) - y_t*grad_t
	  if (grad_diff > 0) {
	    quad_kernel = K_ii + K_tt - 2 * K_it;
	    if (quad_kernel > 0) // for positive definite kernels
	      opt_gain = ( grad_diff * grad_diff ) / quad_kernel; // actually ../2*quad_kernel
	    else // handle non-positive definite kernels
	      opt_gain = ( grad_diff * grad_diff ) / TAU;
	    // find max(opt_gain)
	    if (opt_gain > opt_gain_max) {
	      idx_j = t;
	      opt_gain_max = opt_gain;
	    }
	  }
	}
      }
      else { // y[t] == -1
	if (!IsUpperBounded(t)) {// t\in I_down, y==-1: y[t]alpha[t] >= -C
	  // calculate grad_min for Stopping Criterion
	  if (-grad_[t] <= grad_min) // y==-1
	    grad_min = -grad_[t];
	  // find j
	  grad_diff = grad_max + grad_[t]; // max(y_i*grad_i) - y_t*grad_t
	  if (grad_diff > 0) {
	    quad_kernel = K_ii + K_tt - 2 * K_it;
	    if (quad_kernel > 0) // for positive definite kernels
	      opt_gain = ( grad_diff * grad_diff ) / quad_kernel; // actually ../2*quad_kernel
	    else // handle non-positive definite kernels
	      opt_gain = ( grad_diff * grad_diff ) / TAU;
	    // find max(opt_gain)
	    if (opt_gain > opt_gain_max) {
	      idx_j = t;
	      opt_gain_max = opt_gain;
	    }
	  }
	}
      }
    }
  }
  out_j = idx_j; // j found
  
  // Stopping Criterion check
  if (grad_max - grad_min <= SMO_OPT_TOLERANCE)
    return true; // optimality reached

  return false;
}

/**
* Search direction; Update gradient and alphas
* 
* @param: a working set (i,j) found by working set selection
*
*/
template<typename TKernel>
void SMO<TKernel>::UpdatingGradientAlpha_(index_t i, index_t j) {
  index_t t;

  double a_i = alpha_[i];
  double a_j = alpha_[j];
  int y_i = y_[i];
  int y_j = y_[j];
  double C_i = GetC_(i); // can be Cp (for y==1) or Cn (for y==-1)
  double C_j = GetC_(j);

  /* cached kernel values */
  double K_ii, K_ij, K_jj;
  K_ii = CalcKernelValue_(i, i);
  K_ij = CalcKernelValue_(i, j);
  K_jj = CalcKernelValue_(j, j);
  

  double first_order_diff = y_i * grad_[i] - y_j * grad_[j];
  double second_order_diff = K_ii + K_jj - 2 * K_ij;
  if (second_order_diff < 0) // handle non-positive definite kernels
    second_order_diff = TAU;
  double newton_step = first_order_diff / second_order_diff;

  double step_B, step_A;
  if (y_i == 1) {
    step_B = C_i - a_i;
  }
  else { // y_i == -1
    step_B = a_i;
  }
  if (y_j == 1) {
    step_A = a_j;
  }
  else { // y_j == -1
    step_A = C_j - a_j;
  }
  double min_step_temp = min(step_B, step_A);
  double min_step = min(min_step_temp, newton_step);

  // Update alphas
  alpha_[i] = a_i + y_i * min_step;
  alpha_[j] = a_j - y_j * min_step;

  // Update gradient
  for (t=0; t<n_active_; t++) {
    grad_[t] = grad_[t] + min_step * y_[t] *( CalcKernelValue_(j, t) - CalcKernelValue_(i, t) );
  }

  // Update alpha active status
  UpdateAlphaStatus_(i);
  UpdateAlphaStatus_(j);

  // Update gradient_bar
  bool ub_i = IsUpperBounded(i);
  bool ub_j = IsUpperBounded(j);
  if( ub_i != IsUpperBounded(i) ) {
      if(ub_i)
	for(t=0; t<n_alpha_; t++)
	  grad_bar_[t] -= C_i * CalcKernelValue_(i, t);
      else
	for(t=0; t<n_alpha_; t++)
	  grad_bar_[t] += C_i * CalcKernelValue_(i, t);
  }
  
  if( ub_j != IsUpperBounded(j) ) {
    if(ub_j)
      for(t=0; t<n_alpha_; t++)
	grad_bar_[t] -= C_j * CalcKernelValue_(j, t);
    else
      for(t=0; t<n_alpha_; t++)
	grad_bar_[t] += C_j * CalcKernelValue_(j, t);
  }
  
  // Calculate the bias term
  CalcBias_();

  VERBOSE_GOT_HERE(0);
}

/**
* Calcualte bias term
* 
* @return: the bias
*
*/
template<typename TKernel>
void SMO<TKernel>::CalcBias_() {
  double b;
  index_t n_free = 0;
  double ub = INFINITY, lb = -INFINITY, sum_free = 0;
  for (index_t i=0; i<n_active_; i++){
    double yg = y_[i] * grad_[i];
      
    if (IsUpperBounded(i)) {
      if(y_[i] == -1)
	ub = min(ub, yg);
      else
	lb = max(lb, yg);
    }
    else if (IsLowerBounded(i)) {
      if(y_[i] == +1)
	ub = min(ub, yg);
      else
	lb = max(lb, yg);
    }
    else {
      n_free++;
      sum_free += yg;
    }
  }
  
  if(n_free>0)
    b = - sum_free / n_free;
  else
    b = - (ub + lb) / 2;
  
  bias_ = b;
}

/* Get SVM results:coefficients, number and indecies of SVs
*
* @param: sample indices of the training (sub)set in the total training set
* @param: support vector coefficients: alpha*y
* @param: bool indicators  FOR THE TRAINING SET: is/isn't a support vector
*
*/
template<typename TKernel>
void SMO<TKernel>::GetSVM(ArrayList<index_t> &dataset_index, ArrayList<double> &coef, ArrayList<bool> &sv_indicator) {
  if (learner_typeid_ == 0) {// SVM_C
    for (index_t i = 0; i < n_data_; i++) {
      if (alpha_[i] >= SMO_ALPHA_ZERO) { // support vectors found
	coef.PushBack() = alpha_[i] * y_[i];
	sv_indicator[dataset_index[i]] = true;
	n_sv_++;
      }
      else {
	coef.PushBack() = 0;
      }
    }
  }
  else if (learner_typeid_ == 1) {// SVM_R
    for (index_t i = 0; i < n_data_; i++) {
      double alpha_diff = -alpha_[i] + alpha_[i+n_data_]; // alpha_i^* - alpha_i
      if (fabs(alpha_diff) >= SMO_ALPHA_ZERO) { // support vectors found
	coef.PushBack() = alpha_diff; 
	sv_indicator[dataset_index[i]] = true;
	n_sv_++;
      }
      else {
	coef.PushBack() = 0;
      }
    }
  }
}

#endif
