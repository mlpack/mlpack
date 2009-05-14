/**
 * @author Hua Ouyang
 *
 * @file opt_sga.h
 *
 * This head file contains functions for performing Sparse Greedy Approximation for Large Scale SVMs
 *
 * The algorithms in the following papers are implemented:
 *
 *
 * @see svm.h
 */

#ifndef U_SVM_OPT_SGA_H
#define U_SVM_OPT_SGA_H

#include "fastlib/fastlib.h"
#include "fastlib/base/test.h"

// maximum # of iterations for SGA training
const index_t MAX_NUM_ITER_SGA = 1000000;
// threshold that determines whether an alpha is a SV or not
const double SGA_ALPHA_ZERO = 1.0e-4;
// for inv_mu
const double SGA_ZERO = 1.0e-12;


template<typename TKernel>
class SGA {
  FORBID_ACCIDENTAL_COPIES(SGA);

 public:
  typedef TKernel Kernel;

 private:
  int learner_typeid_;
  index_t ct_iter_; /* counter for the number of iterations */

  Kernel kernel_;
  index_t n_data_; /* number of data samples */
  index_t n_features_; /* # of features == # of row - 1, exclude the last row (for labels) */
  Matrix datamatrix_; /* alias for the matrix of all data, including last label row */
  
  Vector alpha_; /* the alphas, to be optimized */

  index_t n_sv_; /* number of support vectors */

  double q_;
  double r_;
  double lambda_; // optimal step length
  index_t p_; // optimal index of the subgradient

  index_t n_alpha_; /* number of variables to be optimized */
  index_t n_active_; /* number of samples in the active set */
  // n_active + n_inactive == n_alpha;
  ArrayList<index_t> active_set_; /* list that stores the old indices of active alphas followed by inactive alphas */

  ArrayList<int> y_; /* list that stores "labels" */

  double bias_;

  Vector grad_; /* gradient value */
  Vector kernel_cache_; /* cache for kernel values */

  // parameters
  double nu_; // for nu-svm
  double mu_; // for bias term regularization

  double sq_nu_; // nu_^2
  double inv_mu_; // 1/mu
  
  //double epsilon_; // for SVM_R
  index_t n_iter_; // number of iterations
  double accuracy_; // accuracy for stopping creterion

 public:
  SGA() {}
  ~SGA() {}

  /**
   * Initialization for parameters
   */
  void InitPara(int learner_typeid, ArrayList<double> &param_) {
    // init parameters
    nu_ = param_[0];
    sq_nu_ = nu_ * nu_;
    mu_ = param_[1];
    if (mu_ > SGA_ZERO) {
      inv_mu_ = 1 / mu_;
    }
    else {
      fprintf(stderr, " User-provided mu=%f is either negative or too small!!! Use default mu=1.\n", mu_);
      inv_mu_ = 1;
    }
    n_iter_ = (index_t) param_[2];
    n_iter_ = n_iter_ < MAX_NUM_ITER_SGA ? n_iter_: MAX_NUM_ITER_SGA;
    accuracy_ = param_[3];
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

  int SGAIterations_();

  bool GreedyVectorSelection_();

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
void SGA<TKernel>::LearnersInit_(int learner_typeid) {
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

    srand(time(NULL));
    p_ = rand() % n_alpha_; // randomly choose a point for opt

    n_active_ = 0;
    swap(active_set_[p_], active_set_[n_active_]);
    //active_set_[0] = p_;
    n_active_ ++;

    // initialize alpha
    alpha_.Init(n_alpha_);
    alpha_.SetZero();
    alpha_[p_] = nu_;

    // initialize gradient
    grad_.Init(n_alpha_);
    grad_.SetZero();
    for (i=0; i<n_alpha_; i++) {
      grad_[i] = -2 * nu_ * y_[i] * y_[p_] * ( CalcKernelValue_(i, p_) + inv_mu_ );
    }

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
* SGA training for 2-classes
*
* @param: input 2-classes data matrix with labels (1,-1) in the last row
*/
template<typename TKernel>
void SGA<TKernel>::Train(int learner_typeid, const Dataset* dataset_in) {
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
  q_ = sq_nu_ * ( CalcKernelValue_(p_, p_) + inv_mu_ );

  bias_ = 0.0;
  n_sv_ = 0;

  
  // Begin SGA iterations
  ct_iter_ = 0;
  int stop_condition = 0;
  while (1) {
    //for(index_t i=0; i<n_alpha_; i++)
    //  printf("%f.\n", y_[i]*alpha_[i]);
    //printf("\n\n");

    // Find working set, check stopping criterion, update gradient and alphas
    stop_condition = SGAIterations_();
    // Termination check, if stop_condition==1 or ==2 => SGA terminates
    if (stop_condition == 1) {// optimality reached
      // Calculate the bias term
      CalcBias_();
      printf("SGA terminates since the accuracy %f achieved!!! Number of iterations: %d\n.", accuracy_, ct_iter_);
      break;
    }
    else if (stop_condition == 2) {// max num of iterations exceeded
      // Calculate the bias term
      CalcBias_();
      printf("SGA terminates since the number of iterations %d exceeded !!!\n", n_iter_);
      printf("n_active=%d\n", n_active_);
      break;
    }
  }
}

/**
* SGA training iterations
* 
* @return: stopping condition id
*/
template<typename TKernel>
int SGA<TKernel>::SGAIterations_() {
  ct_iter_ ++;
  if (GreedyVectorSelection_() == true)
    return 1;
  else if (ct_iter_ >= n_iter_) { // number of iterations exceeded
    return 2;
  }
  else{ // update gradient, alphas and bias term, and continue iterations
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
bool SGA<TKernel>::GreedyVectorSelection_() {
  double grad_max = -INFINITY;
  //double grad_min =  INFINITY;
  index_t idx_i_grad_max = -1;
  //index_t idx_i_grad_min = -1;
  
  // Find working vector using greedy search: idx_i = argmax_k(grad_k), k = 1...N
  index_t k, op_pos;
  index_t p_po_act = -1; // p's position in active set
  for (k=0; k<n_alpha_; k++) {
    op_pos = active_set_[k];
    if (grad_[op_pos] > grad_max) {
      grad_max = grad_[op_pos];
      idx_i_grad_max = op_pos;
      p_po_act = k;
    }
  }
  p_ = idx_i_grad_max; // i found

  if (p_po_act >= n_active_) { // this sample is not in active set, update it
    swap(active_set_[p_po_act], active_set_[n_active_]);
    n_active_ ++;
  }
  
  // Stopping Criterion check
  //if (y_grad_max - y_grad_min <= accuracy_)
  // return true; // optimality reached

  return false;
}


/**
* Search direction; Update gradient, alphas and bias term
* 
* @param: index p found by greedy selection
*
*/
template<typename TKernel>
void SGA<TKernel>::UpdateGradientAlpha_() {
  index_t i, op_pos;
  
  double sq_nu_App = sq_nu_ * ( CalcKernelValue_(p_, p_) + inv_mu_ );
  
  // update r
  r_ = 0;
  for (i=0; i<n_active_; i++) {
    op_pos = active_set_[i];
    kernel_cache_[i] = CalcKernelValue_(p_,op_pos) + inv_mu_;
    r_ = r_ + alpha_[op_pos] * y_[p_] * y_[op_pos] * ( kernel_cache_[i] );
  }
  r_ = r_ * nu_;

  //printf("lambda=%f\n", lambda_);
  //printf("p_star_=%d\n", p_);

  // update step length: lambda
  lambda_ = 1 + ( r_ - sq_nu_App ) / ( q_ - 2 * r_ + sq_nu_App );
  if (lambda_ > 1) // clamp to [0 1]
    lambda_ = 1;
  else if (lambda_ < 0)
    lambda_ = 0;

  double one_m_lambda = 1 - lambda_;
  double nu_lambda = nu_ * lambda_;

  // update q
  q_ = one_m_lambda * one_m_lambda * q_ + 2 * lambda_ * one_m_lambda * r_ + lambda_ * lambda_ * sq_nu_App;

  // update gradients
  //for (i=0; i<n_alpha_; i++) {
  //  grad_[i] = one_m_lambda * grad_[i] - 2 * nu_lambda * y_[i] * y_[p_] * ( CalcKernelValue_(i, p_) + inv_mu_ );
  //}
  for (i=0; i<n_active_; i++) { // for active set
    op_pos = active_set_[i];
    grad_[op_pos] = one_m_lambda * grad_[op_pos] - 2 * nu_lambda * y_[op_pos] * y_[p_] * ( kernel_cache_[i] );
  }
  for (i=n_active_; i<n_alpha_; i++) { // for inactive set
    op_pos = active_set_[i];
    grad_[op_pos] = one_m_lambda * grad_[op_pos] - 2 * nu_lambda * y_[op_pos] * y_[p_] * ( CalcKernelValue_(op_pos, p_) + inv_mu_ );
  }

  // update alphas
  for (i=0; i<n_active_; i++) {
    op_pos = active_set_[i];
    alpha_[op_pos] = one_m_lambda * alpha_[op_pos];
  }
  alpha_[p_] = alpha_[p_] + nu_lambda;
}


/**
* Calcualte bias term
* 
* @return: the bias
*
*/
template<typename TKernel>
void SGA<TKernel>::CalcBias_() {
  index_t op_pos;
  
  for (index_t i=0; i<n_active_; i++) {
    op_pos = active_set_[i];
    bias_ = bias_ + y_[op_pos] * alpha_[op_pos];
  }
  bias_ = bias_ / mu_;
}

/* Get SVM results:coefficients, number and indecies of SVs
*
* @param: sample indices of the training (sub)set in the total training set
* @param: support vector coefficients: alpha*y
* @param: bool indicators  FOR THE TRAINING SET: is/isn't a support vector
*
*/
template<typename TKernel>
void SGA<TKernel>::GetSV(ArrayList<index_t> &dataset_index, ArrayList<double> &coef, ArrayList<bool> &sv_indicator) {

  if (learner_typeid_ == 0) {// SVM_C
    for (index_t i = 0; i < n_data_; i++) {
      if (alpha_[i] >= SGA_ALPHA_ZERO) { // support vectors found
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
