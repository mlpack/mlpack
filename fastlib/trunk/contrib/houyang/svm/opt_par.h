#ifndef U_SVM_OPT_PAR_H
#define U_SVM_OPT_PAR_H

#include "fastlib/fastlib.h"

// maximum # of interations for PAR training
const index_t MAX_NUM_ITER_PAR = 10000000;
// after # of iterations to do shrinking
const index_t PAR_NUM_FOR_SHRINKING = 1000;
// threshold that determines whether need to do unshrinking
const double PAR_UNSHRINKING_FACTOR = 10;
// threshold that determines whether an alpha is a SV or not
const double PAR_ALPHA_ZERO = 1.0e-7;

template<typename TKernel>
class PAR {
  FORBID_ACCIDENTAL_COPIES(PAR);

 public:
  typedef TKernel Kernel;

 private:
  int learner_typeid_;
  int regularization_; // do L2-SVM or L1-SVM, default: L1

  index_t ct_iter_; /* counter for the number of iterations */
  index_t ct_shrinking_; /* counter for doing shrinking  */
  bool do_shrinking_; // 1(default): do shrinking after 1000 iterations; 0: don't do shrinking

  Kernel kernel_;
  index_t n_data_; /* number of data samples */
  index_t n_features_; /* # of features == # of row - 1, exclude the last row (for labels) */
  Matrix datamatrix_; /* alias for the data matrix, including labels in the last row */
  //Matrix datamatrix_samples_only_; /* alias for the data matrix excluding labels */

  Vector alpha_; /* the alphas, to be optimized */
  index_t n_sv_; /* number of support vectors */
  //index_t n_pos_alpha_;
  
  index_t n_alpha_; /* number of variables to be optimized */
  index_t n_active_; /* number of samples in the active set */
  ArrayList<index_t> active_set_; /* list that stores the old indices of active alphas followed by inactive alphas. == old_from_new*/
  bool reconstructed_; /* indicator: where unshrinking has been carried out  */
  index_t i_cache_, j_cache_; /* indices for the most recently cached kernel value */
  double cached_kernel_value_; /* cache */

  ArrayList<int> y_; /* list that stores "labels" */

  double bias_;

  Vector grad_; /* gradient value */
  Vector grad_bar_; /* gradient value when treat un-upperbounded variables as 0: grad_bar_i==C\sum_{j:a_j=C} y_i y_j K_ij */

  // parameters
  double Cp_; // C_+, for SVM_C, y==1
  double Cn_; // C_-, for SVM_C, y==-1
  double C_;
  double inv_C_; // 1/C
  double inv_two_C_; // 1/2C
  double epsilon_; // for SVM_R
  int wss_; // working set selection scheme, 1 for 1st order expansion; 2 for 2nd order expansion
  index_t n_iter_; // number of iterations
  double accuracy_; // accuracy for stopping creterion
  double gap_; // for stopping criterion

 public:
  PAR() {}
  ~PAR() {}

  /**
   * Initialization for parameters
   */
  void InitPara(int learner_typeid, ArrayList<double> &param_) {
    // init parameters
    wss_ = (int) param_[4];
    regularization_ = (int) param_[3];
    n_iter_ = (index_t) param_[5];
    n_iter_ = n_iter_ < MAX_NUM_ITER_PAR ? n_iter_: MAX_NUM_ITER_PAR;
    accuracy_ = param_[6];
    if (learner_typeid == 0) { // SVM_C
      if (regularization_ == 2) { // L2-SVM
	Cp_ = INFINITY;
	Cn_ = INFINITY;
	C_ = param_[1];
	inv_C_ = 1/C_;
	inv_two_C_ = 1/(2*C_);
      }
      else {
	Cp_ = param_[1];
	Cn_ = param_[2];
      }
    }
    else if (learner_typeid == 1) { // SVM_R
      Cp_ = param_[1];
      Cn_ = Cp_;
      epsilon_ = param_[2];
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

  int PARIterations_();

  void ReconstructGradient_(int learner_typeid);
  
  bool TestShrink_(index_t i, double y_grad_max, double y_grad_min);

  void Shrinking_();

  bool WorkingSetSelection_(index_t &i, index_t &j);

  void UpdateGradientAlpha_(index_t i, index_t j);

  void CalcBias_();

  /*  void GetVector_(index_t i, Vector *v) const {
    datamatrix_.MakeColumnSubvector(i, 0, datamatrix_.n_rows()-1, v);
  }
  */

  /**
   * Instead of C, we use C_+ and C_- to handle unbalanced data
   */
  double GetC_(index_t i) {
    if (regularization_ == 2) { // L2-SVM
      return C_;
    }
    else {
      return (y_[i] > 0 ? Cp_ : Cn_);
    }
  }

  /**
   * Calculate kernel values
   */
  double CalcKernelValue_(index_t i, index_t j) {
    double *v_i, *v_j;
    v_i = datamatrix_.GetColumnPtr(i);
    v_j = datamatrix_.GetColumnPtr(j);

    return kernel_.Eval(v_i, v_j, n_features_);
  }
};


/**
* Reconstruct inactive elements of G from G_bar and free variables 
*
* @param: learner type id
*/
/*
template<typename TKernel>
void PAR<TKernel>::ReconstructGradient_(int learner_typeid) {
  index_t i, j;
  if (n_active_ == n_alpha_)
    return;
  if (learner_typeid == 0) { // SVM_C
    for (i=n_active_; i<n_alpha_; i++) {
      grad_[i] = 1 - grad_bar_[i];
    }
  }
  else if (learner_typeid == 1) { // SVM_R
    for (i=n_active_; i<n_alpha_; i++) {
      j = i >= n_data_ ? (i-n_data_) : i;
      grad_[j] = grad_bar_[j] + datamatrix_.get(datamatrix_.n_rows()-1, active_set_[j]) - epsilon_; // TODO
    }
  }

  for (i=0; i<n_active_; i++) {
    if (alpha_status_[i] == ID_FREE) {
      for (j=n_active_; j<n_alpha_; j++) {
	grad_[j] = grad_[j] - y_[j] * alpha_[i] * y_[i] * CalcKernelValue_(i,j);
      }
    }
  }

}


template<typename TKernel>
bool PAR<TKernel>::TestShrink_(index_t i, double y_grad_max, double y_grad_min) {
  if (IsUpperBounded(i)) { // alpha_[i] = C
    if (y_[i] == 1) {
      return (grad_[i] > y_grad_max);
    }
    else { // y_[i] == -1
      return (grad_[i] + y_grad_min > 0); // -grad_[i]<y_grad_min
    }
  }
  else if (IsLowerBounded(i)) {
    if (y_[i] == 1) {
      return (grad_[i] < y_grad_min);
    }
    else { // y_[i] == -1
      return (grad_[i] + y_grad_max < 0); // -grad_[i]>y_grad_max
    }
  }
  else
    return false;
}


template<typename TKernel>
void PAR<TKernel>::Shrinking_() {
  index_t t;

  // Find m(a) == y_grad_max(i\in I_up) and M(a) == y_grad_min(j\in I_down)
  double y_grad_max = -INFINITY;
  double y_grad_min =  INFINITY;
  for (t=0; t<n_active_; t++) { // find argmax(y*grad), t\in I_up
    if (y_[t] == 1) {
      if (!IsUpperBounded(t)) // t\in I_up, y==1: y[t]alpha[t] < C
	if (grad_[t] > y_grad_max) { // y==1
	  y_grad_max = grad_[t];
	}
    }
    else { // y[t] == -1
      if (!IsLowerBounded(t)) // t\in I_up, y==-1: y[t]alpha[t] < 0
	if (grad_[t] + y_grad_max < 0) { // y==-1... <=> -grad_[t] > y_grad_max
	  y_grad_max = -grad_[t];
	}
    }
  }
  for (t=0; t<n_active_; t++) { // find argmin(y*grad), t\in I_down
    if (y_[t] == 1) {
      if (!IsLowerBounded(t)) // t\in I_down, y==1: y[t]alpha[t] > 0
	if (grad_[t] < y_grad_min) { // y==1
	  y_grad_min = grad_[t];
	}
    }
    else { // y[t] == -1
      if (!IsUpperBounded(t)) // t\in I_down, y==-1: y[t]alpha[t] > -C
	if (grad_[t] + y_grad_min > 0) { // y==-1...<=>  -grad_[t] < y_grad_min
	  y_grad_min = -grad_[t];
	}
    }
  }

  // Find the alpha to be shrunk
  printf("Shrinking...\n");
  for (t=0; t<n_active_; t++) {
    // Shrinking: put inactive alphas behind the active set
    if (TestShrink_(t, y_grad_max, y_grad_min)) {
      n_active_ --;
      while (n_active_ > t) {
	if (!TestShrink_(n_active_, y_grad_max, y_grad_min)) {
	  swap(active_set_[t], active_set_[n_active_]);
	  swap(alpha_[t], alpha_[n_active_]);
	  swap(alpha_status_[t], alpha_status_[n_active_]);
	  swap(y_[t], y_[n_active_]);
	  swap(grad_[t], grad_[n_active_]);
	  swap(grad_bar_[t], grad_bar_[n_active_]);
	  break;
	}
	n_active_ --;
      }
    }
  }
  
  double gap = y_grad_max - y_grad_min;
  printf("%d: gap:%f, n_active:%d\n", ct_iter_, gap, n_active_);
  // do unshrinking for the first time when y_grad_max - y_grad_min <= PAR_UNSHRINKING_FACTOR * accuracy_
  if ( reconstructed_==false && gap <= PAR_UNSHRINKING_FACTOR * accuracy_ ) {
    printf("Unshrinking...\n");
    // Unshrinking: put shrinked alphas back to active set
    // 1.recover gradient
    ReconstructGradient_(learner_typeid_);
    // 2.recover active status
    for (t=n_alpha_-1; t>n_active_; t--) {
      if (!TestShrink_(t, y_grad_max, y_grad_min)) {
	while (n_active_ < t) {
	  if (TestShrink_(n_active_, y_grad_max, y_grad_min)) {
	    swap(active_set_[t], active_set_[n_active_]);
	    swap(alpha_[t], alpha_[n_active_]);
	    swap(alpha_status_[t], alpha_status_[n_active_]);
	    swap(y_[t], y_[n_active_]);
	    swap(grad_[t], grad_[n_active_]);
	    swap(grad_bar_[t], grad_bar_[n_active_]);
	    break;
	  }
	  n_active_ ++;
	}
	n_active_ ++;
      }
    }
    reconstructed_ = true; // indicator: unshrinking has been carried out in this round
  }

}

*/


/**
 * Initialization according to different SVM learner types
 *
 * @param: learner type id 
 */
template<typename TKernel>
void PAR<TKernel>::LearnersInit_(int learner_typeid) {
  index_t i;
  learner_typeid_ = learner_typeid;
  
  if (learner_typeid_ == 0) { // SVM_C
    n_alpha_ = n_data_;
    active_set_.Init(n_alpha_);
    for (i=0; i<n_alpha_; i++) {
      active_set_[i] = i;
    }
    
    y_.Init(n_alpha_);
    for (i = 0; i < n_alpha_; i++) {
      y_[i] = datamatrix_.get(datamatrix_.n_rows()-1, i) > 0 ? 1 : -1;
    }

    // initialize alpha
    alpha_.Init(n_alpha_);
    alpha_.SetZero();
    //    n_pos_alpha_ = 0;
    // randomly choose a point as initial feasible solution
    index_t p = fx_param_int(NULL, "p_rand", rand() % n_alpha_);
    p = p>n_data_ ? (rand() % n_alpha_): p;
    alpha_[p] = 1;
    //swap(active_set_[p], active_set_[n_pos_alpha_]);
    //n_pos_alpha_ ++;
    //printf("p_rand=%d\n", p);

    // initialize gradient
    grad_.Init(n_alpha_);
    grad_.SetZero();
    for (i=0; i<n_alpha_; i++) {
      grad_[i] = -2 * y_[i] * y_[p] * ( CalcKernelValue_(i, p) + 1 );
    }
    grad_[p] = grad_[p] - inv_C_ ;
    
    printf("PAR L2-SVM initialization done!\n");
  }
  else if (learner_typeid_ == 1) { // SVM_R
    // TODO
  }
  else if (learner_typeid_ == 2) { // SVM_DE
    // TODO
  }

}


/**
* PAR training for 2-classes
*
* @param: input 2-classes data matrix with labels (1,-1) in the last row
*/
template<typename TKernel>
void PAR<TKernel>::Train(int learner_typeid, const Dataset* dataset_in) {
  // Load data
  datamatrix_.Alias(dataset_in->matrix());
  n_data_ = datamatrix_.n_cols();
  n_features_ = datamatrix_.n_rows() - 1; // excluding the last row for labels
  //datamatrix_samples_only_.Alias(datamatrix_.ptr(), n_features_, n_data_);

  // Learners initialization
  LearnersInit_(learner_typeid);

  // General learner-independent initializations
  n_active_ = n_alpha_;

  bias_ = 0.0;
  n_sv_ = 0;

  i_cache_ = -1; j_cache_ = -1;
  cached_kernel_value_ = INFINITY;

  do_shrinking_ = fx_param_int(NULL, "shrink", 0);
  ct_shrinking_ = min(n_data_, PAR_NUM_FOR_SHRINKING);
  reconstructed_ = false;

  // Begin PAR iterations
  ct_iter_ = 0;
  int stop_condition = 0;
  while (1) {
    //for(index_t i=0; i<n_alpha_; i++)
    //  printf("%f.\n", y_[i]*alpha_[i]);
    //printf("\n\n");
      
    // for every min(n_data_, 1000) iterations, do shrinking
    if (do_shrinking_) {
      if ( --ct_shrinking_ == 0) {
	/*
	Shrinking_();
	ct_shrinking_ = min(n_data_, PAR_NUM_FOR_SHRINKING);
	*/
      }
    }

    // Find working set, check stopping criterion, update gradient and alphas
    stop_condition = PARIterations_();
    // Termination check, if stop_condition==1 or ==2 => PAR terminates
    if (stop_condition == 1) {// optimality reached
      // Calculate the bias term
      CalcBias_();
      printf("PAR terminates since the accuracy %f achieved!!! Number of iterations: %d.\n", accuracy_, ct_iter_);
      break;
    }
    else if (stop_condition == 2) {// max num of iterations exceeded
      // Calculate the bias term
      CalcBias_();
      fprintf(stderr, "PAR terminates since the number of iterations %d exceeded !!! Gap: %f.\n", n_iter_, gap_);
      break;
    }
  }
}

/**
* PAR training iterations
* 
* @return: stopping condition id
*/
template<typename TKernel>
int PAR<TKernel>::PARIterations_() {
  ct_iter_ ++;
  index_t i,j;
  if (WorkingSetSelection_(i,j) == true) {
    if (!do_shrinking_) { // no shrinking, optimality reached
      return 1;
    }
    /*
    else { // shrinking, need to check whether optimality really reached
      ReconstructGradient_(learner_typeid_); // restore the inactive alphas and reconstruct gradients
      n_active_ = n_alpha_;
      if (WorkingSetSelection_(i,j) == true) { // optimality reached
	return 1;
      }
      else {
	ct_shrinking_ = 1; // do shrinking in the next iteration
	return 0;
      }
    }*/
    return 1;
  }
  else if (ct_iter_ >= n_iter_) { // number of iterations exceeded
    if (!do_shrinking_) { // no shrinking, optimality reached
      return 2;
    }
    else if ( ct_iter_ >= min(n_data_, PAR_NUM_FOR_SHRINKING) ) { // shrinking has been carried out, need to calculate the true gap
      /*
      ReconstructGradient_(learner_typeid_); // restore the inactive alphas and reconstruct gradients
      n_active_ = n_alpha_;
      WorkingSetSelection_(i,j);
      */
      return 2;
    }
    else {
      return 2;
    }
  }
  else{ // update gradient, alphas and bias term, and continue iterations
    UpdateGradientAlpha_(i, j);
    return 0;
  }
}

/**
* Try to find a working set (i,j). Both 1st(default) and 2nd order approximations of 
* the objective function Z(\alpha+\lambda u_ij)-Z(\alpha) are implemented.
*
* @param: reference to working set (i, j)
*
* @return: working set (i, j); indicator of whether the optimal solution is reached (true:reached)
*/
template<typename TKernel>
bool PAR<TKernel>::WorkingSetSelection_(index_t &out_i, index_t &out_j) {
  double grad_max = -INFINITY; // for wss
  double grad_min =  INFINITY;
  int idx_i = -1;
  int idx_j = -1;
  double max_grad_zero = -INFINITY; // for l2-svm optimiality check
  double min_gradinvCalpha_pos = INFINITY; // for l2-svm optimiality check
  
  // Find i using maximal violating pair scheme
  index_t t;
  for (t=0; t<n_active_; t++) { // find argmax(grad)
    if (grad_[t] > grad_max) {
      grad_max = grad_[t];
      idx_i = t;
    }
    if (alpha_[t] <= PAR_ALPHA_ZERO ) {
      if (grad_[t] > max_grad_zero) {
	max_grad_zero = grad_[t];
      }
    }
  }
  out_i = idx_i; // i found

  for (t=0; t<n_active_; t++) { // find argmin(grad)
    if (grad_[t] < grad_min && alpha_[t] > PAR_ALPHA_ZERO ) {
      //if (grad_[t] < grad_min) {
      grad_min = grad_[t];
      idx_j = t;
    }
  }
  out_j = idx_j; // j found

  double min_tmp;
  for (t=0; t<n_active_; t++) {
    if (alpha_[t] > PAR_ALPHA_ZERO) {
      min_tmp = grad_[t] + inv_C_ * alpha_[t];
      if ( min_tmp < min_gradinvCalpha_pos ) {
	min_gradinvCalpha_pos = min_tmp;
      }
    }
  }
  // Stopping Criterion check
  gap_ = max_grad_zero - min_gradinvCalpha_pos;
  printf("%d: gap=%f, i=%d: %f, j=%d: %f\n", ct_iter_, gap_ , out_i, alpha_[out_i], out_j,  alpha_[out_j]);
  if (gap_ <= accuracy_) {
    return true; // optimality reached
  }

  return false;
}

/**
* Search direction; Update gradient, alphas and bias term
* 
* @param: a working set (i,j) found by working set selection
*
*/
template<typename TKernel>
void PAR<TKernel>::UpdateGradientAlpha_(index_t i, index_t j) {
  index_t t;

  double a_i = alpha_[i]; // old alphas
  double a_j = alpha_[j];
  int y_i = y_[i];
  int y_j = y_[j];

  // cached kernel values
  double A_ii, A_ij, A_jj;
  A_ii = CalcKernelValue_(i, i) + 1 + inv_two_C_;
  A_ij = y_i * y_j * (CalcKernelValue_(i, j) + 1);
  A_jj = CalcKernelValue_(j, j) + 1 + inv_two_C_;

  /*
  double first_order_diff = y_i * grad_[i] - y_j * grad_[j];
  double second_order_diff = K_ii + K_jj - 2 * K_ij;
  if (second_order_diff <= 0) // handle non-positive definite kernels
    second_order_diff = TAU;
  double newton_step = first_order_diff / second_order_diff;
  */
  
  double lambda; // step size
  //printf("gi=%f, gj=%f, aii=%f, ajj=%f, aij=%f\n", grad_[i], grad_[j], A_ii, A_jj, A_ij);
  lambda = (grad_[i] - grad_[j] ) / (2*( A_ii + A_jj - 2 * A_ij ));
  //printf("%d: step size=%f\n", ct_iter_, lambda);
  /*
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
  double min_step_temp = min(step_B, step_A);
  double min_step = min(min_step_temp, newton_step);
  */

  // Update alphas
  alpha_[i] = a_i + lambda;
  alpha_[j] = a_j - lambda;
  
  // handle bounds for updated alphas
  if (alpha_[i] < PAR_ALPHA_ZERO) {
    alpha_[i] = 0;
    printf("i=%d bounded!\n", i);
  }
  if (alpha_[j] < PAR_ALPHA_ZERO) {
    alpha_[j] = 0;
    alpha_[i] = a_i + a_j;
    printf("j=%d bounded!\n", j);
  }

  /*
  if (y_i != y_j) {
    double alpha_old_diff = a_i - a_j;
    if (alpha_old_diff > 0) {
      if (alpha_[i] < alpha_old_diff) {
	alpha_[i] = alpha_old_diff;
      }
      else if (alpha_[i] > C_i) {
	alpha_[i] = C_i;
      }
    }
    else { // alpha_old_diff <= 0
      if (alpha_[i] < 0) {
	alpha_[i] = 0;
      }
      else if (alpha_[i] > C_i + alpha_old_diff) {
	alpha_[i] = C_i + alpha_old_diff;
      }
    }
  }
  else { // y_i == y_j
    double alpha_old_sum = a_i + a_j;
    if (alpha_old_sum > C_i) {
      if (alpha_[i] < alpha_old_sum - C_i) {
	alpha_[i] =  alpha_old_sum - C_i;
      }
      else if (alpha_[i] > C_i) {
	alpha_[i] = C_i;
      }
    }
    else { //alpha_old_sum <= C_i
      if (alpha_[i] < 0) {
	alpha_[i] = 0;
      }
      else if (alpha_[i] > alpha_old_sum) {
	alpha_[i] = alpha_old_sum;
      }
    }
  }
  alpha_[j] = a_j + y_i * y_j * (a_i - alpha_[i]);
  */

  // Handle bounds for updated alphas


  /*
  if (y_i != y_j) {
    double alpha_old_diff = a_i - a_j;
    if (alpha_old_diff > 0) {
      if (alpha_[j] < 0) {
	alpha_[j] = 0;
	alpha_[i] = alpha_old_diff;
      }
    }
    else { // alpha_old_diff <= 0
      if (alpha_[i] < 0) {
	alpha_[i] = 0;
	alpha_[j] = - alpha_old_diff;
      }
    }
    if (alpha_old_diff > C_i - C_j) {
      if (alpha_[i] > C_i) {
	alpha_[i] = C_i;
	alpha_[j] = C_i - alpha_old_diff;
      }
    }
    else {
      if (alpha_[j] > C_j) {
	alpha_[j] = C_j;
	alpha_[i] = C_j + alpha_old_diff;
      }
    }
  }
  else { // y_i == y_j
    double alpha_old_sum = a_i + a_j;
    if (alpha_old_sum > C_i) {
      if (alpha_[i] > C_i) {
	alpha_[i] = C_i;
	alpha_[j] = alpha_old_sum - C_i;
      }
    }
    else {
      if (alpha_[j] < 0) {
	alpha_[j] = 0;
	alpha_[i] = alpha_old_sum;
      }
    }
    if (alpha_old_sum > C_j) {
      if (alpha_[j] > C_j) {
	alpha_[j] = C_j;
	alpha_[i] = alpha_old_sum - C_j;
      }
    }
    else {
      if (alpha_[i] < 0) {
	alpha_[i] = 0;
	alpha_[j] = alpha_old_sum;
      }
    }
  }
  */

  // Update gradient
  double diff_i = alpha_[i] - a_i;
  double diff_j = alpha_[j] - a_j;
  for (t=0; t<n_active_; t++) {
    grad_[t] = grad_[t] - 2* y_[t] * (   y_[i] * diff_i * (CalcKernelValue_(i, t)+1) + y_[j] * diff_j * (CalcKernelValue_(j, t)+1)   );
  }
  grad_[i] = grad_[i] - diff_i / C_;
  grad_[j] = grad_[j] - diff_j / C_;

  /*
  bool ub_i = IsUpperBounded(i);
  bool ub_j = IsUpperBounded(j);
  
  // Update alpha active status
  UpdateAlphaStatus_(i);
  UpdateAlphaStatus_(j);


  if (do_shrinking_) {
    // Update gradient_bar
    if( ub_i != IsUpperBounded(i) ) { // updated_alpha_i >= C
      if(ub_i) // old_alpha_i >= C, new_alpha_i < C
	for(t=0; t<n_alpha_; t++)
	  grad_bar_[t] = grad_bar_[t] - C_i * y_[i] * y_[t] * CalcKernelValue_(i, t);
      else // old_alpha_i < C, new_alpha_i >= C
	for(t=0; t<n_alpha_; t++)
	  grad_bar_[t] = grad_bar_[t] + C_i * y_[i] * y_[t] * CalcKernelValue_(i, t);
    }
    if( ub_j != IsUpperBounded(j) ) {
      if(ub_j) // old_alpha_j >= C, new_alpha_j < C
	for(t=0; t<n_alpha_; t++)
	  grad_bar_[t] = grad_bar_[t] - C_j * y_[j] * y_[t] * CalcKernelValue_(j, t);
      else // old_alpha_j < C, new_alpha_j >= C
	for(t=0; t<n_alpha_; t++)
	  grad_bar_[t] = grad_bar_[t] + C_j * y_[j] * y_[t] * CalcKernelValue_(j, t);
    }
  }
  */
  
}

/**
* Calcualte bias term
* 
* @return: the bias
*
*/
template<typename TKernel>
void PAR<TKernel>::CalcBias_() {
  index_t op_pot;
  bias_ = 0;
  for (index_t i=0; i<n_alpha_; i++) {
    op_pot = active_set_[i];
    bias_ = bias_ + y_[op_pot] * alpha_[op_pot];
  }
}

/* Get SVM results:coefficients, number and indecies of SVs
*
* @param: sample indices of the training (sub)set in the total training set
* @param: support vector coefficients: alpha*y
* @param: bool indicators  FOR THE TRAINING SET: is/isn't a support vector
*
*/
template<typename TKernel>
void PAR<TKernel>::GetSV(ArrayList<index_t> &dataset_index, ArrayList<double> &coef, ArrayList<bool> &sv_indicator) {
  if (learner_typeid_ == 0) {// SVM_C
    for (index_t i = 0; i < n_data_; i++) {
      if (alpha_[i] >= FW_ALPHA_ZERO) { // support vectors found
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
