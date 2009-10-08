/**
 * @author Hua Ouyang
 *
 * @file opt_lasvm.h
 *
 * This head file contains functions for performing LASVM
 *
 * The algorithms in the following papers are implemented:
 *
 * 1. LASVM
 * @ARTICLE{Bordes_LASVM,
 * author = "A. Bordes, S. Ertekin, J. Weston, L. Bottou",
 * title = "{Fast Kernel Classifiers with Online and Active Learning}",
 * journal = "{Jornal of Machine Learning Research}",
 * year = 2005,
 * }
 *
 * @see svm.h
 */

#ifndef U_SVM_OPT_LASVM_H
#define U_SVM_OPT_LASVM_H

#include "fastlib/fastlib.h"

// maximum # of interations for LASVM training
const index_t MAX_NUM_ITER_LASVM = 10000000;
// after # of iterations to do shrinking
const index_t LASVM_NUM_FOR_SHRINKING = 1000;
// threshold that determines whether need to do unshrinking
const double LASVM_UNSHRINKING_FACTOR = 10;
// threshold that determines whether an alpha is a SV or not
const double LASVM_ALPHA_ZERO = 1.0e-7;

const double LASVM_ID_LOWER_BOUNDED = -1;
const double LASVM_ID_UPPER_BOUNDED = 1;
const double LASVM_ID_FREE = 0;

template<typename TKernel>
class LASVM {
  FORBID_ACCIDENTAL_COPIES(LASVM);

 public:
  typedef TKernel Kernel;

 private:
  int learner_typeid_;
  int hinge_; // do L2-SVM or L1-SVM, default: L1

  index_t ct_iter_; /* counter for the number of iterations (per epoch) */
  index_t ct_epo_; /* counter for the number of epochs */
  index_t ct_shrinking_; /* counter for doing shrinking  */
  bool do_shrinking_; // 1(default): do shrinking after 1000 iterations; 0: don't do shrinking
  bool do_finishing_; // optimize to the specified accuracy when # of epochs and iterations reached

  Kernel kernel_;
  index_t n_data_; /* number of data samples */
  index_t n_features_; /* # of features == # of row - 1, exclude the last row (for labels) */
  Matrix datamatrix_; /* alias for the data matrix, including labels in the last row */
  //Matrix datamatrix_samples_only_; /* alias for the data matrix excluding labels */

  Vector alpha_; /* the alphas, to be optimized */
  Vector alpha_status_; /*  ID_LOWER_BOUND (-1), ID_UPPER_BOUND (1), ID_FREE (0) */
  index_t n_sv_; /* number of support vectors */
  
  index_t n_alpha_; /* number of variables to be optimized: n_alpha_ == n_acitve + n_inactive */
  index_t n_active_; /* number of samples in the active set: n_active_ = n_active_sv + n_active_non_sv */
  index_t n_active_sv_; /* number of samples in the active set that have positive alpha values */
  ArrayList<index_t> active_set_; /* list that stores the old indices of active alphas followed by inactive alphas; == old_from_new*/
  // active_set == [active, non_active] == [active_sv, active_non_sv, non_active]


  double cached_kernel_value_; /* cache */

  ArrayList<int> y_; /* list that stores "labels" */

  double bias_;

  Vector grad_; /* gradient value */

  // parameters
  double Cp_; // C_+, for SVM_C, y==1
  double Cn_; // C_-, for SVM_C, y==-1
  double C_;
  double inv_two_C_; // 1/2C
  double epsilon_; // for SVM_R
  int wss_; // working set selection scheme, 1: Random 2: Gradient 3: Margin
  index_t n_iter_; // number of iterations
  index_t n_epochs_; // numver of epochs
  double accuracy_; // accuracy for stopping creterion
  double gap_; // for stopping criterion
  double yg_max_; // max(y*grad) s.t. y*alpha<B
  double yg_min_; // min(y*grad) s.t. y*alpha>A

 public:
  LASVM() {}
  ~LASVM() {}
  
  /**
   * Initialization for parameters
   */
  void InitPara(int learner_typeid, ArrayList<double> &param_) {
    // init parameters
    hinge_ = (int) param_[2];
    wss_ = (int) param_[3];
    n_epochs_ = (index_t) param_[4];
    n_iter_ = (index_t) param_[5];
    n_iter_ = n_iter_ < MAX_NUM_ITER_LASVM ? n_iter_: MAX_NUM_ITER_LASVM;
    accuracy_ = param_[6];
    if (learner_typeid == 0) { // SVM_C
      if (hinge_==2) { // L2-SVM: squared hinge loss
	Cp_ = INFINITY;
	Cn_ = INFINITY;
	C_ = param_[1];
	inv_two_C_ = 1/(2*C_);
      }
      else { // L1-SVM
	Cp_ = param_[0];
	Cn_ = param_[1];
      }
    }
    else if (learner_typeid == 1) { // SVM_R
      // TODO
    }
  }

  void Train(int learner_typeid, const Dataset* dataset_in);

  Kernel& kernel() {
    return kernel_;
  }

  double Bias() const {
    return bias_;
  }

  void GetSV(ArrayList<index_t> &dataset_index, ArrayList<double> &coef, ArrayList<bool> &sv_indicator);

 private:
  void LearnersInit_(int learner_typeid);

  int LASVMIterations_();

  void ReconstructGradient_();
  
  bool TestShrink_(index_t i_o, double y_grad_max, double y_grad_min);

  void Shrinking_();

  int Process_(index_t k_o);
  
  int Reprocess_();

  void WorkingSetSelection_(index_t k_n, index_t &i_n, index_t &j_n, double &yg_max, double &yg_min);

  void UpdateGradientAlpha_(index_t i_n, index_t j_n);

  void CalcBias_();

  /**
   * Instead of C, we use C_+ and C_- to handle unbalanced data
   */
  double GetC_(index_t i_o) {
    return (y_[i_o] > 0 ? Cp_ : Cn_);
  }

  void UpdateAlphaStatus_(index_t i_o) {
    if (alpha_[i_o] >= GetC_(i_o)) {
      alpha_status_[i_o] = LASVM_ID_UPPER_BOUNDED;
    }
    else if (alpha_[i_o] <= 0) {
      alpha_status_[i_o] = LASVM_ID_LOWER_BOUNDED;
    }
    else { // 0 < alpha < C
      alpha_status_[i_o] = LASVM_ID_FREE;
    }
  }

  bool IsUpperBounded(index_t i_o) {
    return alpha_status_[i_o] == LASVM_ID_UPPER_BOUNDED;
  }
  bool IsLowerBounded(index_t i_o) {
    return alpha_status_[i_o] == LASVM_ID_LOWER_BOUNDED;
  }

  /**
   * Calculate kernel values
   */
  double CalcKernelValue_(index_t i_o, index_t j_o) {
    // for SVM_R where n_alpha_==2*n_data_
    if (learner_typeid_ == 1) {
      i_o = i_o >= n_data_ ? (i_o - n_data_) : i_o;
      j_o = j_o >= n_data_ ? (j_o - n_data_) : j_o;
    }

    // Check cache
    //if (i == i_cache_ && j == j_cache_) {
    //  return cached_kernel_value_;
    //}

    double *v_i, *v_j;
    //v_i = datamatrix_samples_only_.GetColumnPtr(i);
    //v_j = datamatrix_samples_only_.GetColumnPtr(j);
    v_i = datamatrix_.GetColumnPtr(i_o);
    v_j = datamatrix_.GetColumnPtr(j_o);

    // Do Caching. Store the recently caculated kernel values.
    //i_cache_ = i;
    //j_cache_ = j;
    cached_kernel_value_ = kernel_.Eval(v_i, v_j, n_features_);
    
    if (hinge_ == 2) { // L2-SVM
      if (i_o == j_o) {
	cached_kernel_value_ = cached_kernel_value_ + inv_two_C_;
      }
    }

    return cached_kernel_value_;
  }
};



/**
* Reconstruct inactive elements of G from G_bar and free variables 
*
* @param: learner type id
*/
template<typename TKernel>
void LASVM<TKernel>::ReconstructGradient_() {
  index_t i_o, j_o, i_n, j_n;
  if (n_active_ == n_alpha_)
    return;
  if (learner_typeid_ == 0) { // SVM_C
    for (i_n = n_active_; i_n < n_alpha_; i_n++) {
      i_o = active_set_[i_n];
      grad_[i_o] = 1.0;
    }
  }
  else if (learner_typeid_ == 1) { // SVM_R
    // TODO
  }

  for (i_n=0; i_n<n_active_; i_n++) {
    i_o = active_set_[i_n];
    if (alpha_status_[i_o] != LASVM_ID_LOWER_BOUNDED) {
      for (j_n=n_active_; j_n<n_alpha_; j_n++) {
	j_o = active_set_[j_n];
	grad_[j_o] = grad_[j_o] - y_[j_o] * alpha_[i_o] * y_[i_o] * CalcKernelValue_(i_o, j_o);
      }
    }
  }

}


/**
 * Test whether need to do shrinking for provided index and y_grad_max, y_grad_min
 * 
 */
template<typename TKernel>
bool LASVM<TKernel>::TestShrink_(index_t i_o, double y_grad_max, double y_grad_min) {
  if ( IsUpperBounded(i_o) ) { // alpha_[i] = C
    if (y_[i_o] == 1) {
      return ( grad_[i_o] > y_grad_max );
    }
    else { // y_[i] == -1
      return ( grad_[i_o] + y_grad_min > 0 ); // -grad_[i]<y_grad_min
    }
  }
  else if ( IsLowerBounded(i_o) ) {
    if (y_[i_o] == 1) {
      return ( grad_[i_o] < y_grad_min );
    }
    else { // y_[i] == -1
      return ( grad_[i_o] + y_grad_max < 0 ); // -grad_[i]>y_grad_max
    }
  }
  else
    return false;
}

/**
 * Do Shrinking. Temporarily remove alphas (from the active set) that are 
 * unlikely to be selected in the working set, since they have reached their 
 * lower/upper bound.
 * 
 */
template<typename TKernel>
void LASVM<TKernel>::Shrinking_() {
  index_t t_n;

  // Find the alpha to be shrunk
  printf("Shrinking...\n");
  for (t_n = 0; t_n < n_active_; t_n++) {
    // Shrinking: put inactive alphas behind the active set
    if ( TestShrink_(active_set_[t_n], yg_max_, yg_min_) ) {
      n_active_ --;
      while (n_active_ > t_n) {
	if ( !TestShrink_(active_set_[n_active_], yg_max_, yg_min_) ) {
	  swap(active_set_[t_n], active_set_[n_active_]);
	  break;
	}
	n_active_ --;
      }
    }
  }

}


/**
 * Initialization according to different SVM learner types
 *
 * @param: learner type id 
 */
template<typename TKernel>
void LASVM<TKernel>::LearnersInit_(int learner_typeid) {
  index_t i; // i_o
  learner_typeid_ = learner_typeid;
  
  if (learner_typeid_ == 0) { // SVM_C
    n_alpha_ = n_data_;

    alpha_.Init(n_alpha_);
    alpha_.SetZero();

    // initialize gradient
    grad_.Init(n_alpha_);
    grad_.SetAll(1.0);

    y_.Init(n_alpha_);
    for (i = 0; i < n_alpha_; i++) {
      y_[i] = datamatrix_.get(datamatrix_.n_rows()-1, i) > 0 ? 1 : -1;
    }

  }
  else if (learner_typeid_ == 1) { // SVM_R
    n_alpha_ = 2 * n_data_;

    alpha_.Init(2 * n_alpha_); // TODO
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
    // TODO
  }
}


/**
* LASVM training for 2-classes
*
* @param: input 2-classes data matrix with labels (1,-1) in the last row
*/
template<typename TKernel>
void LASVM<TKernel>::Train(int learner_typeid, const Dataset* dataset_in) {
  index_t i_o, i_n;
  int stop_condition = 0;

  // Load data
  datamatrix_.Alias(dataset_in->matrix());
  n_data_ = datamatrix_.n_cols();
  n_features_ = datamatrix_.n_rows() - 1; // excluding the last row for labels

  if (n_epochs_ > 0) { // # of epochs provided, use it
    n_iter_ = n_data_;
  }
  else { // # of epochs not provided, use n_iter_ to count iterations
    n_epochs_ = 1; // not exactly one epoch, just use it for one loop
  }

  gap_ = INFINITY;

  // Learners initialization
  LearnersInit_(learner_typeid);

  // General learner-independent initializations
  do_shrinking_ = fx_param_int(NULL, "shrink", 0);
  do_finishing_ = fx_param_int(NULL, "finish", 1);
  bias_ = 0.0;
  n_sv_ = 0;
  ct_iter_ = 0;
  ct_epo_ = 0;

  cached_kernel_value_ = INFINITY;

  n_active_ = n_alpha_;
  n_active_sv_ = 0;
  active_set_.Init(n_alpha_);
  alpha_status_.Init(n_alpha_);
  for (i_o = 0; i_o < n_alpha_; i_o++) {
      active_set_[i_o] = i_o;
      UpdateAlphaStatus_(i_o);
  }

  // first add 5 samples of each class, just to balance the initial set
  int c1 = 0, c2 = 0;
  for (i_n = 0; i_n < n_alpha_; i_n++) {
    i_o = active_set_[i_n];
    if (y_[i_o] == 1 && c1 < 5) {
      Process_(i_n);
      c1++;
      ct_iter_ ++;
    }
    if (y_[i_o] == -1 && c2 < 5) {
      Process_(i_n);
      c2++;
      ct_iter_ ++;
    }
    if (c1 == 5 && c2 == 5) {
      break;
    }
  }
  
  printf("LASVM initialization done!\n");

  // Begin LASVM iterations
  for (ct_epo_ = 0; ct_epo_ < n_epochs_; ct_epo_++) {
    // In each epoch, random permute the active_non_sv samples to mimic the online setting
    for (index_t i=n_active_sv_; i<n_active_; i++) {
      index_t j = rand() % (n_active_ - n_active_sv_);
      swap( active_set_[i], active_set_[n_active_sv_+j] );
    }
    ct_iter_ = 0;
    while (ct_iter_ <= n_iter_) {
      // Find working set, check stopping criterion, update gradient and alphas
      stop_condition = LASVMIterations_();
      // Termination check
      if (stop_condition == 1) {// max num of iterations exceeded
	// Calculate the bias term
	CalcBias_();
	fprintf(stderr, "LASVM terminates since the number of iterations %d exceeded !!! Gap: %f.\n", n_iter_, gap_);
	break;
      }
      else if (stop_condition == 2) {// optimality reached
	// Calculate the bias term
	CalcBias_();
	printf("LASVM terminates since the accuracy %g achieved!!! Number of epochs: %d; Number of iterations: %d.\n", accuracy_, ct_epo_+1, ct_iter_);
	break;
      }
    }
  }
}

/**
* LASVM training iterations
* 
* @return: stopping condition id
*/
template<typename TKernel>
int LASVM<TKernel>::LASVMIterations_() {
  index_t k_n = -1;
  ct_iter_++;
  //printf("epo=%d, iter=%d\n", ct_epo_, ct_iter_);

  // if number of epochs and iterations exceeded, do Finishing: repeat REPROCESS until accuracy reached
  if ( gap_ <= accuracy_ ) {
    return 2;
  }
  else if ( ( ct_epo_ == n_epochs_ -1 ) && ( ct_iter_ >= n_iter_ ) ) {
    if (do_finishing_) {
      while (gap_ > accuracy_) {
	if (do_shrinking_) {
	  // do shrinking for every 1000 REPROCESS iterations
	  if ( --ct_shrinking_ == 0 ) {
	    Shrinking_();
	    ct_shrinking_ = min(n_data_, LASVM_NUM_FOR_SHRINKING);
	  }
	}
	Reprocess_();
      }
      if (do_shrinking_) {
	ReconstructGradient_();
	n_active_ = n_alpha_;
	Reprocess_();
	if ( gap_ <= accuracy_ ) {
	  return 2;
	}
	else {
	  return 0;
	}
      }
      else {
	return 2;
      }
    }
    return 1;
  }

  // Select k for PROCESS
  if (wss_ == 1) { // Random selection
    if ( n_active_ > n_active_sv_) {
      k_n = n_active_sv_ + ct_iter_ % (n_active_ - n_active_sv_);
    }
    else { // all samples are in S, PROCESS just do nothing, so only need REPROCESS
      k_n = -1;
      if ( Reprocess_() == 0 ) {
	// bail out since i, j is not a violating pair
	return 0;
      }
      return 0;
    }
  }
  else if(wss_ == 2) { // Active learning via gradient
    // TODO
    k_n = -1;
  }
  else if(wss_ == 3) { // Active learning via margin
    // TODO
    k_n = -1;
  }

  // LASVM PROCESS AND REPROCESS
  if ( Process_(k_n) == 0 ) {
    // bail out since k already in the expansion, or i, j is not a violating pair
    return 0;
  }
  if ( Reprocess_() == 0 ) {
    // bail out since i, j is not a violating pair
    return 0;
  }
  
  // continue iterations
  return 0;
}


/**
* LASVM Process
*
* @param: 
*
* @return: 0: bailed out; 1: gradient and alpha updated
*/
template<typename TKernel>
int LASVM<TKernel>::Process_(index_t k_n) {
  index_t i_n, j_n, t_n, t_o, k_o;

  //printf("Process: epo:%d, iter:%d, k_n=%d, n_active_sv_=%d, n_active_=%d\n", ct_epo_, ct_iter_, k_n, n_active_sv_, n_active_);

  // if k is already in the expansion, bail it out
  if (k_n < n_active_sv_) {
    return 0;
  }
  k_o = active_set_[k_n];
  alpha_[k_o] = 0.0;
  // calculate grad_k
  double grad_tmp = 1.0;
  for (t_n = 0; t_n < n_active_sv_; t_n++) {
    t_o = active_set_[t_n];
    grad_tmp = grad_tmp - y_[k_o] * y_[t_o] * alpha_[t_o] * CalcKernelValue_(k_o, t_o);
  }
  // insert k into S
  swap( active_set_[k_n], active_set_[n_active_sv_] );
  n_active_sv_ ++;
  grad_[k_o] = grad_tmp;

  WorkingSetSelection_(n_active_sv_-1, i_n, j_n, yg_max_, yg_min_);
  if (yg_max_ - yg_min_ <= accuracy_) { // i, j is not a violating pair, bailed out
    //printf("Bail out; epo:%d, iter:%d, k_n=%d, n_active_sv_=%d, gap=%f\n", ct_epo_, ct_iter_, k_n, n_active_sv_, yg_max_ - yg_min_);
    return 0;
  }
  UpdateGradientAlpha_(i_n, j_n);
  //printf("PROCESS epo:%d, iter:%d, k_n=%d, n_active_sv_=%d\n", ct_epo_, ct_iter_, k_n, n_active_sv_);
  return 1;
}



/**
* LASVM Reprocess
*
* @return: 0: bailed out; 1: gradient and alpha updated
*/
template<typename TKernel>
int LASVM<TKernel>::Reprocess_() {
  index_t i_n, j_n, t_n, t_o;
  
  WorkingSetSelection_(-2, i_n, j_n, yg_max_, yg_min_);
  //printf("REProcess: epo:%d, iter:%d, n_active_sv_=%d, gap=%f\n", ct_epo_, ct_iter_, n_active_sv_, yg_max_ - yg_min_);
  gap_ = yg_max_ - yg_min_;
  if (gap_ <= accuracy_) { // i, j is not a violating pair, bailed out
    return 0;
  }
  UpdateGradientAlpha_(i_n, j_n);

  // remove samples from S
  WorkingSetSelection_(-2, i_n, j_n, yg_max_, yg_min_);
  gap_ = yg_max_ - yg_min_;
  for (t_n = 0; t_n < n_active_sv_; t_n++) {
    t_o = active_set_[t_n];
    if (alpha_[t_o] < LASVM_ALPHA_ZERO ) {
      if ( y_[t_o] == 1 && grad_[t_o] <= yg_min_ ) {
	swap( active_set_[t_n], active_set_[n_active_sv_-1] );
	n_active_sv_ --;
      }
      else if ( y_[t_o] == -1 && grad_[t_o] + yg_max_ <= 0 ) {
	swap( active_set_[t_n], active_set_[n_active_sv_-1] );
	n_active_sv_ --;
      }
    }
  }

  // TODO: update bias b=(gi+gj)/2

  gap_ = yg_max_ - yg_min_;

  printf("REPROCESS ct_iter:%d, n_active_sv_=%d, gap=%f\n", ct_iter_, n_active_sv_, gap_);
  
  return 1;
}



/**
* Try to find a working set (i,j)
*
* @return: working set (i, j); indicator of whether the optimal solution is reached (true:reached)
*/
template<typename TKernel>
void LASVM<TKernel>::WorkingSetSelection_(index_t k_n, index_t &out_i_n, index_t &out_j_n, double &out_yg_max, double &out_yg_min) {
  index_t t_n, t_o, k_o;
  index_t i_n = -1, j_n = -1;
  double y_grad_max = -INFINITY;
  double y_grad_min =  INFINITY;

  if (k_n >= 0) { // for PROCESS, k provided
    k_o = active_set_[k_n];
    if ( y_[k_o] == 1 ) {
      for (t_n = 0; t_n < n_active_; t_n++) {
	t_o = active_set_[t_n];
	if (y_[t_o] == 1) {
	  if ( !IsLowerBounded(t_o) ) {
	    if (grad_[t_o] < y_grad_min) {
	      y_grad_min = grad_[t_o];
	      j_n = t_n;
	    }
	  }
	}
	else { // y_[t_o] == -1
	  if ( !IsUpperBounded(t_o) ) {
	    if (grad_[t_o] + y_grad_min > 0) {
	      y_grad_min = -grad_[t_o];
	      j_n = t_n;
	    }
	  }
	}
      }
      out_i_n = k_n;
      out_j_n = j_n;
      
      out_yg_max = grad_[k_o];
      out_yg_min = y_grad_min;
      //printf("epo:%d, iter:%d. in=%d, jn=%d, ygmax=%f, yg_min=%f\n", ct_epo_, ct_iter_, out_i_n, out_j_n, out_yg_max, out_yg_min);
    }
    else { // y_[k_o] == -1
      for (t_n = 0; t_n < n_active_; t_n++) {
	t_o = active_set_[t_n];
	if ( y_[t_o] == 1 ) {
	  if ( !IsUpperBounded(t_o) ) {
	    if (grad_[t_o] > y_grad_max ) {
	      y_grad_max = grad_[t_o];
	      i_n = t_n;
	    }
	  }
	}
	else { // y_[t_o] == -1
	  if ( !IsLowerBounded(t_o) ) {
	    if (grad_[t_o] + y_grad_max < 0) {
	      y_grad_max = -grad_[t_o];
	      i_n = t_n;
	    }
	  }
	}
      }
      out_i_n = i_n;
      out_j_n = k_n;
      
      out_yg_max = y_grad_max;
      out_yg_min = -grad_[k_o];
      //printf("epo:%d, iter:%d. in=%d, jn=%d, ygmax=%f, yg_min=%f\n", ct_epo_, ct_iter_, out_i_n, out_j_n, out_yg_max, out_yg_min);
    }
  }
  else { // for REPROCESS, dummy k==-1
    for (t_n = 0; t_n < n_active_; t_n++) { // find argmax(y*grad), t\in I_up
      t_o = active_set_[t_n];
      if (y_[t_o] == 1) {
	if ( !IsUpperBounded(t_o) ) // t\in I_up, y==1: y[t]alpha[t] < C
	  if ( grad_[t_o] > y_grad_max ) { // y==1
	    y_grad_max = grad_[t_o];
	    i_n = t_n;
	  }
      }
      else { // y[t] == -1
	if ( !IsLowerBounded(t_o) ) // t\in I_up, y==-1: y[t]alpha[t] < 0
	  if (grad_[t_o] + y_grad_max < 0) { // y==-1... <=> -grad_[t] > y_grad_max
	    y_grad_max = -grad_[t_o];
	    i_n = t_n;
	  }
      }
    }
    for (t_n = 0; t_n < n_active_; t_n++) { // find argmin(y*grad), t\in I_down
      t_o = active_set_[t_n];
      if (y_[t_o] == 1) {
	if ( !IsLowerBounded(t_o) ) // t\in I_down, y==1: y[t]alpha[t] > 0
	  if ( grad_[t_o] < y_grad_min ) { // y==1
	    y_grad_min = grad_[t_o];
	    j_n = t_n;
	  }
      }
      else { // y[t] == -1
	if ( !IsUpperBounded(t_o) ) // t\in I_down, y==-1: y[t]alpha[t] > -C
	  if ( grad_[t_o] + y_grad_min > 0 ) { // y==-1...<=>  -grad_[t] < y_grad_min
	    y_grad_min = -grad_[t_o];
	    j_n = t_n;
	  }
      }
    }
    out_i_n = i_n;
    out_j_n = j_n;
    
    out_yg_max = y_grad_max; 
    out_yg_min = y_grad_min;
  }
}


/**
* Search direction; Update gradient, alphas and bias term
* 
* @param: indicies (new) of a working set (i,j) found by working set selection
*
*/
template<typename TKernel>
void LASVM<TKernel>::UpdateGradientAlpha_(index_t i_n, index_t j_n) {
  index_t i_o = active_set_[i_n];
  index_t j_o = active_set_[j_n];
  index_t t_n, t_o;

  double a_i = alpha_[i_o]; // old alphas
  double a_j = alpha_[j_o];
  int y_i = y_[i_o];
  int y_j = y_[j_o];
  double C_i = GetC_(i_o);
  double C_j = GetC_(j_o);

  double K_ii, K_ij, K_jj;
  K_ii = CalcKernelValue_(i_o, i_o);
  K_ij = CalcKernelValue_(i_o, j_o);
  K_jj = CalcKernelValue_(j_o, j_o);

  // Calculate step size
  double first_order_diff = y_i * grad_[i_o] - y_j * grad_[j_o];
  double second_order_diff = K_ii + K_jj - 2 * K_ij;
  if (second_order_diff <= 0) { // handle non-positive definite kernels
    second_order_diff = TAU;
  }
  double newton_step = first_order_diff / second_order_diff;

  double step_B, step_A;
  if (y_i == 1) {
    step_B = C_i - a_i;
  }
  else { // y_i == -1
    step_B = a_i; // 0-(-1)a_i
  }
  if (y_j == 1) {
    step_A = a_j;
  }
  else { // y_j == -1
    step_A = C_j - a_j; // (-1)a_j - (-C_j)
  }
  double lambda_temp = min(step_B, step_A);
  double lambda = min(lambda_temp, newton_step); // step size
  //printf("epo:%d, iter:%d, step size=%f\n", ct_epo_, ct_iter_, lambda);

  // Update alphas
  alpha_[i_o] = a_i + y_i * lambda;
  alpha_[j_o] = a_j - y_j * lambda;

  // Update gradient
  double diff_i = alpha_[i_o] - a_i;
  double diff_j = alpha_[j_o] - a_j;
  for (t_n = 0; t_n < n_active_sv_; t_n++) {
    t_o = active_set_[t_n];
    grad_[t_o] = grad_[t_o] - y_[t_o] * (y_[i_o] * diff_i * CalcKernelValue_(i_o, t_o) + y_[j_o] * diff_j * CalcKernelValue_(j_o, t_o));
  }
  
  // Update alpha active status
  UpdateAlphaStatus_(i_o);
  UpdateAlphaStatus_(j_o);
  
}

/**
* Calcualte bias term
* 
* @return: the bias
*
*/
template<typename TKernel>
void LASVM<TKernel>::CalcBias_() {
  double b;
  index_t i, n_free_alpha = 0;
  double ub = INFINITY, lb = -INFINITY, sum_free_yg = 0.0;
  
  for (index_t i_n=0; i_n<n_active_; i_n++){
    i = active_set_[i_n];
    double yg = y_[i] * grad_[i];
      
    if (IsUpperBounded(i)) { // bounded: alpha_i >= C
      if(y_[i] == 1)
	lb = max(lb, yg);
      else
	ub = min(ub, yg);
    }
    else if (IsLowerBounded(i)) { // bounded: alpha_i <= 0
      if(y_[i] == -1)
	lb = max(lb, yg);
      else
	ub = min(ub, yg);
    }
    else { // free: 0< alpha_i <C
      n_free_alpha++;
      sum_free_yg += yg;
    }
  }
  
  if(n_free_alpha>0)
    b = sum_free_yg / n_free_alpha;
  else
    b = (ub + lb) / 2;
  
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
void LASVM<TKernel>::GetSV(ArrayList<index_t> &dataset_index, ArrayList<double> &coef, ArrayList<bool> &sv_indicator) {

  if (learner_typeid_ == 0) {// SVM_C
    for (index_t i = 0; i < n_data_; i++) {
      if (alpha_[i] >= LASVM_ALPHA_ZERO) { // support vectors found
	//printf("%f\n", alpha_[i] * y_[i]);
	coef.PushBack() = alpha_[i] * y_[i];
	sv_indicator[dataset_index[i]] = true;
	n_sv_++;
      }
      else {
	coef.PushBack() = 0;
      }
    }
    printf("Number of SVs: %d\n", n_sv_);
  }
  else if (learner_typeid_ == 1) {// SVM_R
    // TODO
  }
}

#endif
