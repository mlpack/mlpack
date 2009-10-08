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
const index_t MAX_NUM_ITER_SFW = 100000000;
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
  double lambda_; // optimal step length
  index_t p_; // optimal index of the subgradient

  index_t n_alpha_; /* number of variables to be optimized */

  index_t n_working_pos_; /* number of samples in the working set with label +1 */
  index_t n_working_neg_; /* number of samples in the working set with label -1 */

  // List that stores the old indices of working alphas followed by nonworking alphas;
  // 1st half for +1 class, 2nd half for -1 class: [working_+1, nonworking_+1, working_-1, nonworking_-1]
  ArrayList<index_t> old_from_new_;

  ArrayList<int> y_; /* list that stores "labels" */

  double bias_;

  double alpha_scale_; // scaler for alpha, using it will not need to scale all alphas during every iteration

  Vector grad_; /* gradient value */
  Vector A_cache_; /* cache for kernel values */
  bool b_A_cached_;

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

  int step_type_; // 1: Toward Step; -1: Away Step
  bool b_away_; // whether use both toward and away steps
  index_t n_away_steps_;

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
    n_iter_ = (index_t) param_[2];
    n_iter_ = n_iter_ < MAX_NUM_ITER_SFW ? n_iter_: MAX_NUM_ITER_SFW;
    accuracy_ = param_[3];
    n_data_pos_ = (index_t)param_[4];
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
    old_from_new_.Init(n_alpha_);
    for (i=0; i<n_alpha_; i++)
      old_from_new_[i] = i;

    y_.Init(n_data_);
    for (i = 0; i < n_data_; i++) {
      y_[i] = datamatrix_.get(datamatrix_.n_rows()-1, i) > 0 ? 1 : -1;
    }

    n_working_pos_ = 0;
    n_working_neg_ = 0;
    //p_ = rand() % n_alpha_; // randomly choose a point for opt
    p_ = fx_param_int(NULL, "p_rand", rand() % n_alpha_);
    printf("p_rand=%d\n", p_);
    if (y_[p_] == 1) {
      swap(old_from_new_[p_], old_from_new_[n_working_pos_]);
      n_working_pos_ ++;
      sample_pos_ = true;
    }
    else { // y_[p_] == -1
      swap(old_from_new_[p_], old_from_new_[n_data_pos_ + n_working_neg_]);
      n_working_neg_ ++;
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

    // init q
    q_ = -grad_[p_] / 2;

    A_cache_.Init(n_alpha_);
    A_cache_.SetZero();

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
  lambda_ = 1.0; // optimal step length  
  // Learners initialization
  LearnersInit_(learner_typeid);
  // General learner-independent initializations
  b_away_ = fx_param_int(NULL, "away", 1); // default use both toward and away steps
  alpha_scale_ = 1.0;
  step_type_ = 1;
  n_away_steps_ = 0;
  bias_ = 0.0;
  n_sv_ = 0;

  
  // Begin SFW iterations
  ct_iter_ = 0;
  int stop_condition = 0;
  while (1) {
    // Find working set, check stopping criterion, update gradient and alphas
    stop_condition = SFWIterations_();
    // Termination check, if stop_condition==1 or ==2 => SFW terminates
    if (stop_condition == 1) {// optimality reached
      // Calculate the bias term
      CalcBias_();
      printf("SFW terminates since the accuracy %f achieved!!! Number of iterations: %d. Portion of Away steps:%f\n.", accuracy_, ct_iter_, double(n_away_steps_)/double(ct_iter_));
      printf("n_work_pos=%d, n_work_neg=%d, n_work=%d, alpha_scale=%g\n", n_working_pos_, n_working_neg_, n_working_pos_+n_working_neg_, alpha_scale_);
      break;
    }
    else if (stop_condition == 2) {// max num of iterations exceeded
      // Calculate the bias term
      CalcBias_();
      printf("SFW terminates since the number of iterations %d exceeded !!! Portion of Away steps:%f\n", n_iter_, double(n_away_steps_)/double(n_iter_));
      printf("n_work_pos=%d, n_work_neg=%d, n_work=%d, alpha_scale_=%g\n", n_working_pos_, n_working_neg_, n_working_pos_+n_working_neg_, alpha_scale_);
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
  else if (GreedyVectorSelection_() == 1){ // optimality reached
    return 1;
  }
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
  double grad_min =  INFINITY;
  index_t idx_grad_max_o = -1;
  index_t idx_grad_min_o = -1;

  index_t k, op_pos, neg_tmp;

  double alpha_zero_scaled = SFW_ALPHA_ZERO / alpha_scale_;

  //double max_grad_inact = -INFINITY; // for optimiality check
  //double min_gradinvCalpha_act = INFINITY; // for optimiality check

  index_t p_rnd, p_rnd_o;

  // balanced sampling (for unbalanced classes)
  index_t n_nonworking_pos, n_nonworking_neg;
  if (sample_pos_) { // last sample is +1, this sample should be -1
    n_nonworking_neg = n_alpha_ - n_data_pos_ - n_working_neg_;
    if (n_nonworking_neg >0) {
      p_rnd = n_data_pos_ + n_working_neg_ + rand() % n_nonworking_neg; // choose a random sample x_{p_rnd} from the nonworking set
      sample_pos_ = false;
    }
    else { // no more neg points to sample, use positive points
      printf("No more -1 new samples\n");
      n_nonworking_pos = n_data_pos_ - n_working_pos_;
      p_rnd = n_working_pos_ + rand() % n_nonworking_pos; // choose a random sample x_{p_rnd} from the nonworking set
      sample_pos_ = true;
    }
    //printf("n_act_pos=%d, n_act_neg=%d, p_rnd_neg=%d\n", n_working_pos_, n_working_neg_, p_rnd);
  }
  else { // last sample is -1, this sample should be +1
    n_nonworking_pos = n_data_pos_ - n_working_pos_;
    if (n_nonworking_pos >0) {
      p_rnd = n_working_pos_ + rand() % (n_data_pos_ - n_working_pos_); // choose a random sample x_{p_rnd} from the nonworking set
      sample_pos_ = true;
    }
    else{ // no more positive points to sample, use negative points
      printf("No more +1 new samples\n");
      n_nonworking_neg = n_alpha_ - n_data_pos_ - n_working_neg_;
      p_rnd = n_data_pos_ + n_working_neg_ + rand() % n_nonworking_neg; // choose a random sample x_{p_rnd} from the nonworking set
      sample_pos_ = false;
    }
    //printf("n_act_pos=%d, n_act_neg=%d, p_rnd_pos=%d\n", n_working_pos_, n_working_neg_, p_rnd);
  }
  p_rnd_o = old_from_new_[p_rnd];
  

  // Caculate p_rnd th component of the gradient
  double grad_k = 0;
  neg_tmp = n_data_pos_+n_working_neg_;
  for (k=0; k<n_working_pos_; k++) {
    op_pos = old_from_new_[k];
    if (alpha_[op_pos] > alpha_zero_scaled) {
      A_cache_[k] = y_[op_pos] * y_[p_rnd_o] * ( CalcKernelValue_(p_rnd_o,op_pos) + 1 );
      grad_k = grad_k - alpha_[op_pos] * A_cache_[k];
    }
    else {
      A_cache_[k] = INFINITY;
    }
  }
  for (k=n_data_pos_; k<neg_tmp; k++) {
    op_pos = old_from_new_[k];
    if (alpha_[op_pos] > alpha_zero_scaled) {
      A_cache_[k] = y_[op_pos] * y_[p_rnd_o] * ( CalcKernelValue_(p_rnd_o,op_pos) + 1 );
      grad_k = grad_k - alpha_[op_pos] * A_cache_[k];
    }
    else {
      A_cache_[k] = INFINITY;
    }
  }
  grad_k = 2 * grad_k;
  grad_k = alpha_scale_ * (grad_k - alpha_[p_rnd_o] * inv_C_);

  /*
  for (k=0; k<n_working_pos_; k++) {
    op_pos = old_from_new_[k];
    grad_k = grad_k - y_[op_pos] * alpha_[op_pos] * ( CalcKernelValue_(p_rnd_o, op_pos) + 1 );
  }
  for (k=0; k<n_working_neg_; k++) {
    op_pos = old_from_new_[n_data_pos_ + k];
    grad_k = grad_k - y_[op_pos] * alpha_[op_pos] * ( CalcKernelValue_(p_rnd_o, op_pos) + 1 );
  }
  grad_k = 2 * y_[p_rnd_o] * grad_k;
  grad_k = grad_k - alpha_[p_rnd_o] * inv_C_;
  */

  //max_grad_inact = grad_k;

  // Find grad_max
  for (k=0; k<n_working_pos_; k++) {
    op_pos = old_from_new_[k];
    if (grad_[op_pos] > grad_max) {
      grad_max = grad_[op_pos];
      idx_grad_max_o = op_pos;
    }
  }
  for (k=n_data_pos_; k<neg_tmp; k++) {
    op_pos = old_from_new_[k];
    if (grad_[op_pos] > grad_max) {
      grad_max = grad_[op_pos];
      idx_grad_max_o = op_pos;
    }
  }

  //printf("grad_k=%g, grad_max=%g\n", grad_k, grad_max);
  if ( grad_k >= grad_max) { // include new ramdom sample in active_set
    grad_[p_rnd_o] = grad_k;
    p_ = p_rnd_o;
    step_type_ = 1; // Toward step
    b_A_cached_ = 1;
    //printf("Use New Sample:%d!!!!!!!\n", p_);
    // update active_set
    if (sample_pos_) { // this sample is +1
      swap(old_from_new_[p_rnd], old_from_new_[n_working_pos_]);
      //A_cache_[n_working_pos_] = CalcKernelValue_(p_,p_) + 1;
      n_working_pos_ ++;
    }
    else { // this sample is -1
      swap(old_from_new_[p_rnd], old_from_new_[neg_tmp]);
      //A_cache_[neg_tmp] = CalcKernelValue_(p_,p_) + 1;
      n_working_neg_ ++;
    }
  }
  else { // new random sample not selected for opt
    if (b_away_) { // consider both toward and away steps
      // find grad_min
      for (k=0; k<n_working_pos_; k++) {
	op_pos = old_from_new_[k];
	if (  (grad_[op_pos] < grad_min) && (alpha_[op_pos] > alpha_zero_scaled)  ) {
	  grad_min = grad_[op_pos];
	  idx_grad_min_o = op_pos;
	}
      }
      for (k=n_data_pos_; k<neg_tmp; k++) {
	op_pos = old_from_new_[k];
	if (  (grad_[op_pos] < grad_min) && (alpha_[op_pos] > alpha_zero_scaled)  ) {
	  grad_min = grad_[op_pos];
	  idx_grad_min_o = op_pos;
	}
      }
      // Calc alpha^T * grad
      double alpha_grad = 0.0;
      for (k=0; k<n_working_pos_; k++) {
	op_pos = old_from_new_[k];
	alpha_grad += alpha_[op_pos] * grad_[op_pos];
      }
      for (k=n_data_pos_; k<neg_tmp; k++) {
	op_pos = old_from_new_[k];
	alpha_grad += alpha_[op_pos] * grad_[op_pos];
      }
      alpha_grad *= alpha_scale_;
      
      //printf("ct_iter:%d, max=%g, min=%g, max+min=%g, 2*alpha_grad=%g\n",ct_iter_, grad_max, grad_min, grad_max + grad_min, 2 * alpha_grad);
      //printf("ct_iter:%d, max=%g, min=%g, |max|=%g, |min|=%g\n",ct_iter_, grad_max, grad_min, fabs(grad_max), fabs(grad_min));
      if ( (grad_max + grad_min) >= (2 * alpha_grad) ) { // Guelat's MFW
      //if (fabs(grad_max) >= fabs(grad_min)) { // Wolfe's MFW
	step_type_ = 1; // Toward step
	p_ = idx_grad_max_o;
	b_A_cached_ = 0;
      }
      else {
	step_type_ = -1; // Away step
	n_away_steps_ ++; 
	p_ = idx_grad_min_o;
	b_A_cached_ = 0;
      }
    }
    else { // only use toward steps
      p_ = idx_grad_max_o;
      b_A_cached_ = 0;
      //printf("Use Old Sample:%d\n", p_);
    }
  }

  /*
  double min_tmp;
  for (k=0; k<n_working_pos_; k++) {
    op_pos = old_from_new_[k];
    min_tmp = grad_[op_pos] + inv_C_ * alpha_[op_pos];
    if ( min_tmp <min_gradinvCalpha_act ) {
      min_gradinvCalpha_act = min_tmp;
    }
  }
  for (k=0; k<n_working_neg_; k++) {
    op_pos = old_from_new_[n_data_pos_ + k];
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
  index_t i, op_pos, neg_tmp;
  double one_m_lambda, one_p_lambda, two_lambda;
  double alpha_zero_scaled = SFW_ALPHA_ZERO / alpha_scale_;
  double App = CalcKernelValue_(p_, p_) + 1 + inv_two_C_;

  if (step_type_ == 1) { // Make a Toward Step
    // update step length: lambda
    lambda_ = 1 - ( grad_[p_]/2 + App ) / ( q_ + grad_[p_] + App );
    //printf("TOWARD: ct_iter:%d, lambda=%g, p=%d, scale:%g, alpha_p=%g ", ct_iter_, lambda_, p_, alpha_scale_, alpha_[p_]*alpha_scale_);
    if (lambda_ > 1) { // clamp to [0 1]
      lambda_ = 1;
    }
    else if (lambda_ < 0) {
      lambda_ = 0;
    }
    
    if (lambda_ > 0) {
      one_m_lambda = 1 - lambda_;
      two_lambda = 2 * lambda_;
      
      // update q
      q_ = one_m_lambda * one_m_lambda * q_ - lambda_ * one_m_lambda * grad_[p_] + lambda_ * lambda_ * App;
      
      // update gradients
      neg_tmp = n_data_pos_+n_working_neg_;
      if (b_A_cached_) { // p_ is new, using cache
	for (i=0; i<n_working_pos_; i++) {
	  op_pos = old_from_new_[i];
	  if (alpha_[op_pos] > alpha_zero_scaled) {
	    grad_[op_pos] = one_m_lambda * grad_[op_pos] - two_lambda * A_cache_[i];
	  }
	  else {
	    grad_[op_pos] = one_m_lambda * grad_[op_pos] - two_lambda * y_[op_pos] * y_[p_] * ( CalcKernelValue_(op_pos, p_) + 1 );
	  }
	}
	for (i=n_data_pos_; i<neg_tmp; i++) {
	  op_pos = old_from_new_[i];
	  if (alpha_[op_pos] > alpha_zero_scaled) {
	    grad_[op_pos] = one_m_lambda * grad_[op_pos] - two_lambda * A_cache_[i];
	  }
	  else {
	    grad_[op_pos] = one_m_lambda * grad_[op_pos] - two_lambda * y_[op_pos] * y_[p_] * ( CalcKernelValue_(op_pos, p_) + 1 );
	  }
	}
	grad_[p_] = grad_[p_] - lambda_ * inv_C_;
      }
      else { // p_ is old
	for (i=0; i<n_working_pos_; i++) {
	  op_pos = old_from_new_[i];
	  grad_[op_pos] = one_m_lambda * grad_[op_pos] - two_lambda * y_[op_pos] * y_[p_] * ( CalcKernelValue_(op_pos, p_) + 1 );
	}
	for (i=n_data_pos_; i<neg_tmp; i++) {
	  op_pos = old_from_new_[i];
	  grad_[op_pos] = one_m_lambda * grad_[op_pos] - two_lambda * y_[op_pos] * y_[p_] * ( CalcKernelValue_(op_pos, p_) + 1 );
	}
	grad_[p_] = grad_[p_] - lambda_ * inv_C_;
      }
      
      // update alphas
      alpha_scale_ *= one_m_lambda;
      alpha_[p_] = alpha_[p_] + lambda_ / alpha_scale_;
    }
    //printf("after_p=%g\n", alpha_[p_]*alpha_scale_);
  }
  else { // Make an Away step
    // update step length: lambda
    lambda_ = ( grad_[p_]/2 + App ) / ( q_ + grad_[p_] + App ) - 1;
    // clamp to [0 eta]
    double eta = alpha_[p_]*alpha_scale_ / (1 - alpha_[p_]*alpha_scale_);
    //printf("AWAY: ct_iter:%d, lambda=%g, eta=%g, p=%d, scale:%g, alpha_p=%g ", ct_iter_, lambda_, eta, p_, alpha_scale_, alpha_[p_]*alpha_scale_);
    if (lambda_ > eta) {
      lambda_ = eta;
    }
    else if (lambda_ < 0) {
      lambda_ = 0;
    }
    
    if (lambda_ > 0) {
      one_p_lambda = 1 + lambda_;
      two_lambda = 2 * lambda_;
      
      // update q
      q_ = one_p_lambda * one_p_lambda * q_ + lambda_ * one_p_lambda * grad_[p_] + lambda_ * lambda_ * App;
      
      // update gradients
      neg_tmp = n_data_pos_+n_working_neg_;
      if (b_A_cached_) { // p_ is new, using cache
	for (i=0; i<n_working_pos_; i++) {
	  op_pos = old_from_new_[i];
	  if (alpha_[op_pos] > alpha_zero_scaled) {
	    grad_[op_pos] = one_p_lambda * grad_[op_pos] + two_lambda * A_cache_[i];
	  }
	  else {
	    grad_[op_pos] = one_p_lambda * grad_[op_pos] + two_lambda * y_[op_pos] * y_[p_] * ( CalcKernelValue_(op_pos, p_) + 1 );
	  }
	}
	for (i=n_data_pos_; i<neg_tmp; i++) {
	  op_pos = old_from_new_[i];
	  if (alpha_[op_pos] > alpha_zero_scaled) {
	    grad_[op_pos] = one_p_lambda * grad_[op_pos] + two_lambda * A_cache_[i];
	  }
	  else {
	    grad_[op_pos] = one_p_lambda * grad_[op_pos] + two_lambda * y_[op_pos] * y_[p_] * ( CalcKernelValue_(op_pos, p_) + 1 );
	  }
	}
	grad_[p_] = grad_[p_] + lambda_ * inv_C_;
      }
      else { // p_ is old
	for (i=0; i<n_working_pos_; i++) {
	  op_pos = old_from_new_[i];
	  grad_[op_pos] = one_p_lambda * grad_[op_pos] + two_lambda * y_[op_pos] * y_[p_] * ( CalcKernelValue_(op_pos, p_) + 1 );
	}
	for (i=n_data_pos_; i<neg_tmp; i++) {
	  op_pos = old_from_new_[i];
	  grad_[op_pos] = one_p_lambda * grad_[op_pos] + two_lambda * y_[op_pos] * y_[p_] * ( CalcKernelValue_(op_pos, p_) + 1 );
	}
	grad_[p_] = grad_[p_] + lambda_ * inv_C_;
      }
      
      // update alphas
      alpha_scale_ *= one_p_lambda;
      alpha_[p_] = alpha_[p_] - lambda_ / alpha_scale_;
    }
    //printf("after_p=%g\n", alpha_[p_]*alpha_scale_);
  }

  //printf("ct_iter:%d, p:%d, y_p:%d, step_size:%g, n_work=%d, alpha_scale_=%g, lambda_=%g\n\n", ct_iter_, p_, y_[p_], lambda_,  n_working_pos_ + n_working_neg_, alpha_scale_, lambda_);
}


/**
* Calcualte bias term
* 
* @return: the bias
*
*/
template<typename TKernel>
void SFW<TKernel>::CalcBias_() {
  index_t i, op_pos, neg_tmp;
  
  bias_ = 0;
  for (i=0; i<n_working_pos_; i++) {
    op_pos = old_from_new_[i];
    bias_ = bias_ + y_[op_pos] * alpha_[op_pos];
  }
  neg_tmp = n_data_pos_+n_working_neg_;
  for (i=n_data_pos_; i<neg_tmp; i++) {
    op_pos = old_from_new_[i];
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
  n_sv_ = 0;
  double alpha_zero_scaled = SFW_ALPHA_ZERO / alpha_scale_;
  if (learner_typeid_ == 0) {// SVM_C
    for (index_t i = 0; i < n_data_; i++) {
      if (alpha_[i] > alpha_zero_scaled) { // support vectors found
	//printf("%f\n", alpha_[i] * y_[i]);
	coef.PushBack() = alpha_[i] * y_[i];
	sv_indicator[dataset_index[i]] = true;
	n_sv_++;
      }
      else {
	coef.PushBack() = 0;
      }
    }
    printf("Number of SVs: %d.................\n", n_sv_);
  }
  else if (learner_typeid_ == 1) {// SVM_R
    // TODO
  }
}

#endif
// ./svm_main --learner_name=svm_c --mode=train_test --train_data=w3a_train_sort.csv --test_data=w3a_test_sort.csv --kernel=gaussian --normalize=0 --opt=sfw --sigma=4 --c=5 --p_rand=680 --away=1 --n_iter=9824
