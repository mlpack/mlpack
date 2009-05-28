/**
 * @author Hua Ouyang
 *
 * @file opt_sfw.h
 *
 * This head file contains functions for performing Stochastic Frank-Wolfe Algorithm for Large Scale SVMs
 *
 * The algorithms in the following papers are implemented:
 *
 *
 * @see svm.h
 */

#ifndef U_SVM_OPT_SFW_H
#define U_SVM_OPT_SFW_H

#include "fastlib/fastlib.h"
#include "fastlib/base/test.h"

// maximum # of iterations for SFW training
const index_t MAX_NUM_ITER_SFW = 1000000;
// threshold that determines whether an alpha is a SV or not
const double SFW_ALPHA_ZERO = 1.0e-7;
// for inv_mu
const double SFW_ZERO = 1.0e-12;

template<typename TKernel>
class SFW {
  FORBID_ACCIDENTAL_COPIES(SFW);

 public:
  typedef TKernel Kernel;

 private:
  int learner_typeid_;
  index_t ct_iter_; /* counter for the number of iterations */

  Kernel kernel_;
  index_t n_data_; /* number of data samples */
  index_t n_data_pos_; /* number of data samples with label +1 */
  index_t n_features_; /* # of features == # of row - 1, exclude the last row (for labels) */
  Matrix datamatrix_; /* alias for the matrix of all data, including last label row */
  
  Vector alpha_; /* the alphas, to be optimized */

  index_t n_sv_; /* number of support vectors */

  double q_;
  double r_;
  double lambda_; // optimal step length
  index_t p_; // optimal index of the subgradient

  index_t n_alpha_; /* number of variables to be optimized */
  //index_t n_active_; /* number of samples in the active set */
  index_t n_active_pos_; /* number of samples in the active set with label +1 */
  index_t n_active_neg_; /* number of samples in the active set with label -1 */
  // n_active + n_inactive == n_alpha;
  /* list that stores the old indices of active alphas followed by inactive alphas. 1st half for +1 class, 2nd half for -1 class */
  // active_set = [active_+1, inactive_+1, active_-1, inactive_-1]
  ArrayList<index_t> active_set_;

  ArrayList<int> y_; /* list that stores "labels" */

  double bias_;

  Vector grad_; /* gradient value */
  Vector kernel_cache_; /* cache for kernel values */

  // parameters
  //double nu_; // for nu-svm
  //double mu_; // for bias term regularization
  double C_; // weight for regularization

  //double sq_nu_; // nu_^2
  //double inv_mu_; // 1/mu
  double inv_C_; // 1/C
  double inv_two_C_; // 1/2C
  
  //double epsilon_; // for SVM_R
  index_t n_iter_; // number of iterations
  double accuracy_; // accuracy for stopping creterion

  // indicator for balanced sampling
  bool sample_pos_;
 public:
  SFW() {}
  ~SFW() {}

  /**
   * Initialization for parameters
   */
  void InitPara(int learner_typeid, ArrayList<double> &param_) {
    // init parameters
    //nu_ = param_[0];
    //sq_nu_ = nu_ * nu_;
    /*
    mu_ = param_[1];
    if (mu_ > SFW_ZERO) {
      inv_mu_ = 1 / mu_;
    }
    else {
      fprintf(stderr, " User-provided mu=%f is either negative or too small!!! Use default mu=1.\n", mu_);
      inv_mu_ = 1;
    }
    */
    C_ = param_[0];
    if (C_ > SFW_ZERO) {
      inv_two_C_ = 1 / (2 * C_);
      inv_C_ = 1 / C_;
    }
    else {
      fprintf(stderr, " User-provided C=%f is either negative or too small!!! Use default C=1.\n", C_);
      inv_two_C_ = 0.5;
      inv_C_ = 1;
    }
    n_iter_ = (index_t) param_[1];
    n_iter_ = n_iter_ < MAX_NUM_ITER_SFW ? n_iter_: MAX_NUM_ITER_SFW;
    accuracy_ = param_[2];
    n_data_pos_ = (index_t)param_[3];
    if (learner_typeid == 0) { // SVM_C
      //Cp_ = param_[1];
      //Cn_ = param_[2];
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

  int SFWIterations_();

  int GreedyVectorSelection_();

  void UpdateGradientAlpha_();

  void CalcBias_();

  /**
   * Calculate kernel values
   */
  double CalcKernelValue_(index_t i, index_t j) {
    // for SVM_R where max_n_alpha_==2*n_data_
    /*
    if (learner_typeid_ == 1) {
      i = i >= n_data_ ? (i-n_data_) : i;
      j = j >= n_data_ ? (j-n_data_) : j;
      }*/

    // Check cache
    //if (i == i_cache_ && j == j_cache_) {
    //  return cached_kernel_value_;
    //}

    double *v_i, *v_j;
    v_i = datamatrix_.GetColumnPtr(i);
    v_j = datamatrix_.GetColumnPtr(j);

    return kernel_.Eval(v_i, v_j, n_features_);
  }
};


/**
 * Initialization according to different SVM learner types
 *
 * @param: learner type id 
 */
template<typename TKernel>
void SFW<TKernel>::LearnersInit_(int learner_typeid) {
  index_t i;
  learner_typeid_ = learner_typeid;
  
  if (learner_typeid_ == 0) { // SVM_C
    n_alpha_ = n_data_;
    active_set_.Init(n_alpha_);
    for (i=0; i<n_alpha_; i++)
      active_set_[i] = i;

    y_.Init(n_data_);
    for (i = 0; i < n_data_; i++) {
      y_[i] = datamatrix_.get(datamatrix_.n_rows()-1, i) > 0 ? 1 : -1;
    }

    n_active_pos_ = 0;
    n_active_neg_ = 0;
    //p_ = rand() % n_alpha_; // randomly choose a point for opt
    p_ = fx_param_int(NULL, "p_rand", rand() % n_alpha_);
    printf("p_rand=%d\n", p_);
    if (y_[p_] == 1) {
      swap(active_set_[p_], active_set_[n_active_pos_]);
      n_active_pos_ ++;
      sample_pos_ = true;
    }
    else { // y_[p_] == -1
      swap(active_set_[p_], active_set_[n_data_pos_ + n_active_neg_]);
      n_active_neg_ ++;
      sample_pos_ = false;
    }

    // initialize alpha
    alpha_.Init(n_alpha_);
    alpha_.SetZero();
    alpha_[p_] = 1;

    // initialize gradient
    grad_.Init(n_alpha_);
    grad_.SetZero();
    grad_[p_] = -2 * ( CalcKernelValue_(p_, p_) + 1 + inv_two_C_ );
    /*
    for (i=0; i<n_active_; i++) {
      op_pos = active_set_[i];
      grad_[op_pos] = -2 * nu_ * y_[op_pos] * y_[p_] * ( CalcKernelValue_(op_pos, p_) + inv_mu_ );
    }
    */

    kernel_cache_.Init(n_alpha_);
    kernel_cache_.SetZero();

  }
  else if (learner_typeid_ == 1) { // SVM_R
    // TODO
  }
  else if (learner_typeid_ == 2) { // SVM_DE
    // TODO
  }
  
}

/**
* SFW training for 2-classes
*
* @param: input 2-classes data matrix with labels (1,-1) in the last row
*/
template<typename TKernel>
void SFW<TKernel>::Train(int learner_typeid, const Dataset* dataset_in) {
  srand(time(NULL));
  // Load data
  datamatrix_.Alias(dataset_in->matrix());
  n_data_ = datamatrix_.n_cols();
  n_features_ = datamatrix_.n_rows() - 1; // excluding the last row for labels

  // General learner-independent initializations
  r_ = 0.0;
  lambda_ = 1.0; // optimal step length  
  // Learners initialization
  LearnersInit_(learner_typeid);
  // General learner-independent initializations
  q_ = CalcKernelValue_(p_, p_) + 1 + inv_two_C_;

  bias_ = 0.0;
  n_sv_ = 0;

  
  // Begin SFW iterations
  ct_iter_ = 0;
  int stop_condition = 0;
  while (1) {
    //for(index_t i=0; i<n_alpha_; i++)
    //  printf("%f.\n", y_[i]*alpha_[i]);
    //printf("\n\n");

    // Find working set, check stopping criterion, update gradient and alphas
    stop_condition = SFWIterations_();
    // Termination check, if stop_condition==1 or ==2 => SFW terminates
    if (stop_condition == 1) {// optimality reached
      // Calculate the bias term
      CalcBias_();
      printf("SFW terminates since the accuracy %f achieved!!! Number of iterations: %d\n.", accuracy_, ct_iter_);
      printf("n_act_pos=%d, n_act_neg=%d, n_act=%d\n", n_active_pos_, n_active_neg_, n_active_pos_+n_active_neg_);
      break;
    }
    else if (stop_condition == 2) {// max num of iterations exceeded
      // Calculate the bias term
      CalcBias_();
      printf("SFW terminates since the number of iterations %d exceeded !!!\n", n_iter_);
      printf("n_act_pos=%d, n_act_neg=%d, n_act=%d\n", n_active_pos_, n_active_neg_, n_active_pos_+n_active_neg_);
      break;
    }
  }
}

/**
* SFW training iterations
* 
* @return: stopping condition id
*/
template<typename TKernel>
int SFW<TKernel>::SFWIterations_() {
  ct_iter_ ++;
  if (ct_iter_ >= n_iter_) { // number of iterations exceeded
    return 2;
  }
  else if (GreedyVectorSelection_() == 1) // optimality reached
    return 1;
  //else if (GreedyVectorSelection_() == 2) { // new sample not included in the candidate set, find another
  //  return 0;
  //}
  else{ // new sample included in the candidate set, update gradient, alphas and bias term, and continue iterations
    UpdateGradientAlpha_();
    return 0;
  }
}

/**
* Greedy searching a working vector i with largest gradient
*
* @param: reference to the index i of the working vector
*
* @return: indicator of whether the optimal solution is reached (true:reached)
*/
template<typename TKernel>
int SFW<TKernel>::GreedyVectorSelection_() {
  double grad_max = -INFINITY;
  //double grad_min =  INFINITY;
  index_t idx_i_grad_max = -1;
  //index_t idx_i_grad_min = -1;

  index_t k, op_pos;

  //double max_grad_inact = -INFINITY; // for optimiality check
  //double min_gradinvCalpha_act = INFINITY; // for optimiality check

  index_t p_rnd, p_rnd_idx;

  // balanced sampling (for unbalanced classes)
  index_t n_inactive_pos, n_inactive_neg;
  if (sample_pos_) { // last sample is +1, this sample should be -1
    n_inactive_neg = n_alpha_ - n_data_pos_ - n_active_neg_;
    if (n_inactive_neg >0) {
      p_rnd = n_data_pos_ + n_active_neg_ + rand() % n_inactive_neg; // choose a random sample x_{p_rnd} from the inactive set
      sample_pos_ = false;
    }
    else { // no more neg points to sample, use positive points
      n_inactive_pos = n_data_pos_ - n_active_pos_;
      p_rnd = n_active_pos_ + rand() % n_inactive_pos; // choose a random sample x_{p_rnd} from the inactive set
      sample_pos_ = true;
    }
    //printf("n_act_pos=%d, n_act_neg=%d, p_rnd_neg=%d\n", n_active_pos_, n_active_neg_, p_rnd);
  }
  else { // last sample is -1, this sample should be +1
    n_inactive_pos = n_data_pos_ - n_active_pos_;
    if (n_inactive_pos >0) {
      p_rnd = n_active_pos_ + rand() % (n_data_pos_ - n_active_pos_); // choose a random sample x_{p_rnd} from the inactive set
      sample_pos_ = true;
    }
    else{ // no more positive points to sample, use negative points
      n_inactive_neg = n_alpha_ - n_data_pos_ - n_active_neg_;
      p_rnd = n_data_pos_ + n_active_neg_ + rand() % n_inactive_neg; // choose a random sample x_{p_rnd} from the inactive set
      sample_pos_ = false;
    }
    //printf("n_act_pos=%d, n_act_neg=%d, p_rnd_pos=%d\n", n_active_pos_, n_active_neg_, p_rnd);
  }
  p_rnd_idx = active_set_[p_rnd];
  

  // caculate p_rnd th component of the gradient
  double grad_tmp = 0;
  for (k=0; k<n_active_pos_; k++) {
    op_pos = active_set_[k];
    grad_tmp = grad_tmp - y_[op_pos] * alpha_[op_pos] * ( CalcKernelValue_(p_rnd_idx, op_pos) + 1 );
  }
  for (k=0; k<n_active_neg_; k++) {
    op_pos = active_set_[n_data_pos_ + k];
    grad_tmp = grad_tmp - y_[op_pos] * alpha_[op_pos] * ( CalcKernelValue_(p_rnd_idx, op_pos) + 1 );
  }
  grad_tmp = 2 * y_[p_rnd_idx] * grad_tmp;
  grad_tmp = grad_tmp - alpha_[p_rnd_idx] * inv_C_;

  //max_grad_inact = grad_tmp;

  // Find working vector using greedy search: idx_i = argmax_k(grad_k), k \in active_set
  for (k=0; k<n_active_pos_; k++) {
    op_pos = active_set_[k];
    if (grad_[op_pos] > grad_max) {
      grad_max = grad_[op_pos];
      idx_i_grad_max = op_pos;
    }
  }
  for (k=0; k<n_active_neg_; k++) {
    op_pos = active_set_[n_data_pos_ + k];
    if (grad_[op_pos] > grad_max) {
      grad_max = grad_[op_pos];
      idx_i_grad_max = op_pos;
    }
  }

  //printf("grad_max=%f, grad_[p_rnd_idx]=%f\n", grad_max, grad_[p_rnd_idx]);
  if ( grad_tmp >= grad_max) { // include new ramdom sample in active_set
    grad_[p_rnd_idx] = grad_tmp;
    p_ = p_rnd_idx;
    //printf("new:%d\n", p_);
    // update active_set
    if (sample_pos_) { // this sample is +1
      swap(active_set_[p_rnd], active_set_[n_active_pos_]);
      n_active_pos_ ++;
    }
    else { // this sample is -1
      swap(active_set_[p_rnd], active_set_[n_data_pos_ + n_active_neg_]);
      n_active_neg_ ++;
    }
  }
  else { // new random sample not selected for opt
    p_ = idx_i_grad_max;
    //printf("old:%d\n", p_);
  }

  /*
  double min_tmp;
  for (k=0; k<n_active_pos_; k++) {
    op_pos = active_set_[k];
    min_tmp = grad_[op_pos] + inv_C_ * alpha_[op_pos];
    if ( min_tmp <min_gradinvCalpha_act ) {
      min_gradinvCalpha_act = min_tmp;
    }
  }
  for (k=0; k<n_active_neg_; k++) {
    op_pos = active_set_[n_data_pos_ + k];
    min_tmp = grad_[op_pos] + inv_C_ * alpha_[op_pos];
    if ( min_tmp <min_gradinvCalpha_act ) {
      min_gradinvCalpha_act = min_tmp;
    }
  }
  
  // Stopping Criterion check
  double gap = max_grad_inact - min_gradinvCalpha_act;
  printf("%d: gap=%f\n", ct_iter_, gap);
  if (gap <= accuracy_) {
    return 1; // optimality reached
  }
  */

  return -1;
}


/**
* Search direction; Update gradient, alphas and bias term
* 
* @param: index p found by greedy selection
*
*/
template<typename TKernel>
void SFW<TKernel>::UpdateGradientAlpha_() {
  index_t i, op_pos;
  double one_m_lambda;
  double App = CalcKernelValue_(p_, p_) + 1 + inv_two_C_;
  
  // update r
  r_ = 0;
  for (i=0; i<n_active_pos_; i++) {
    op_pos = active_set_[i];
    kernel_cache_[i] = y_[op_pos] * y_[p_] * ( CalcKernelValue_(p_,op_pos) + 1 );
    r_ = r_ + alpha_[op_pos] * kernel_cache_[i];
  }
  for (i=0; i<n_active_neg_; i++) {
    op_pos = active_set_[n_data_pos_ + i];
    kernel_cache_[n_active_pos_+i] = y_[op_pos] * y_[p_] * ( CalcKernelValue_(p_,op_pos) + 1 );
    r_ = r_ + alpha_[op_pos] * kernel_cache_[n_active_pos_+i];
  }
  r_ = r_ + alpha_[p_] * inv_two_C_;

  //printf("lambda=%f\n", lambda_);
  //printf("p_star_=%d\n", p_);

  // update step length: lambda
  lambda_ = 1 + ( r_ - App ) / ( q_ - 2 * r_ + App );
  if (lambda_ > 1) // clamp to [0 1]
    lambda_ = 1;
  else if (lambda_ < 0)
    lambda_ = 0;

  //printf("%d: lambda =%f\n", ct_iter_, lambda_);

  one_m_lambda = 1 - lambda_;

  // update q
  q_ = one_m_lambda * one_m_lambda * q_ + 2 * lambda_ * one_m_lambda * r_ + lambda_ * lambda_ * App;

  // update gradients
  for (i=0; i<n_active_pos_; i++) {
    op_pos = active_set_[i];
    grad_[op_pos] = one_m_lambda * grad_[op_pos] - 2 * lambda_ * kernel_cache_[i];
  }
  for (i=0; i<n_active_neg_; i++) {
    op_pos = active_set_[n_data_pos_ + i];
    grad_[op_pos] = one_m_lambda * grad_[op_pos] - 2 * lambda_ * kernel_cache_[n_active_pos_+i];
  }
  grad_[p_] = grad_[p_] - lambda_ * inv_C_;

  // update alphas
  for (i=0; i<n_active_pos_; i++) {
    op_pos = active_set_[i];
    alpha_[op_pos] = one_m_lambda * alpha_[op_pos];
  }
  for (i=0; i<n_active_neg_; i++) {
    op_pos = active_set_[n_data_pos_ + i];
    alpha_[op_pos] = one_m_lambda * alpha_[op_pos];
  }
  alpha_[p_] = alpha_[p_] + lambda_;
}


/**
* Calcualte bias term
* 
* @return: the bias
*
*/
template<typename TKernel>
void SFW<TKernel>::CalcBias_() {
  index_t op_pos;
  
  bias_ = 0;
  for (index_t i=0; i<n_active_pos_; i++) {
    op_pos = active_set_[i];
    bias_ = bias_ + y_[op_pos] * alpha_[op_pos];
  }
  for (index_t i=0; i<n_active_neg_; i++) {
    op_pos = active_set_[n_data_pos_ + i];
    bias_ = bias_ + y_[op_pos] * alpha_[op_pos];
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
void SFW<TKernel>::GetSV(ArrayList<index_t> &dataset_index, ArrayList<double> &coef, ArrayList<bool> &sv_indicator) {

  if (learner_typeid_ == 0) {// SVM_C
    for (index_t i = 0; i < n_data_; i++) {
      if (alpha_[i] >= SFW_ALPHA_ZERO) { // support vectors found
	//printf("%f\n", alpha_[i] * y_[i]);
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
    // TODO
  }
}

#endif
