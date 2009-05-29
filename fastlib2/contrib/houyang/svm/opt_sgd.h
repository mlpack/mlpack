/**
 * @author Hua Ouyang
 *
 * @file opt_sgd.h
 *
 * This head file contains functions for performing Stochastic Gradient Descent (SGD) based optimization for SVM
 *
 * The algorithms in the following papers are implemented:
 *
 * 1. SGD for linear SVM
 * @ARTICLE{Zhang_SGD,
 * author = "Tong Zhang",
 * title = "{Solving Large Scale Linear Prediction Problems Using Stochastic Gradient Descent Algorithms}",
 * booktitle = "{International Conference on Machine Learning}",
 * year = 2004,
 * }
 *
 * 2. SGD for nonlinear SVM
 * @ARTICLE{Kivinen_SGD,
 * author = "Jyrki Kivinen",
 * title = "{Online Learning with Kernels}",
 * booktitle = NIPS,
 * number = 14,
 * year = 2001,
 * }
 *
 * @see svm.h
 */

#ifndef U_SVM_OPT_SGD_H
#define U_SVM_OPT_SGD_H

#include "fastlib/fastlib.h"

// max t_
const index_t MAX_NUM_ITER_SGD = 1000000;
// tolerance of sacale_w
const double SCALE_W_TOLERANCE = 1.0e-9;
// threshold that determines whether an alpha is a SV or not
const double SGD_ALPHA_ZERO = 1.0e-7;


template<typename TKernel>
class SGD {
  FORBID_ACCIDENTAL_COPIES(SGD);

 public:
  typedef TKernel Kernel;

 private:
  int learner_typeid_;

  Kernel kernel_;
  const Dataset *dataset_;
  index_t n_data_; /* number of data samples */
  index_t n_features_; /* # of features == # of row - 1, exclude the last row (for labels) */
  Matrix datamatrix_; /* alias for the data matrix */

  Vector coef_; /* alpha*y, to be optimized */
  index_t n_alpha_; /* number of variables to be optimized */
  index_t n_sv_; /* number of support vectors */
  
  index_t i_cache_, j_cache_; /* indices for the most recently cached kernel value */
  double cached_kernel_value_; /* cache */

  ArrayList<int> y_; /* list that stores "labels" */

  Vector w_; /* the slope of the decision hyperplane y=w^T x+b */
  double bias_;
  double scale_w_; // the scale for w

  // parameters
  double C_; // for SVM_C
  double epsilon_; // for SVM_R
  bool b_linear_; // whether it's a linear SVM
  double lambda_; // regularization parameter. lambda = 1/(C*n_data)
  index_t n_iter_; // number of iterations
  double accuracy_; // accuracy for stopping creterion
  double eta_; // step length. eta = 1/(lambda*t)
  double t_;

  double rho_;// for soft margin nonlinear SGD SVM

 public:
  SGD() {}
  ~SGD() {}

  /**
   * Initialization for parameters
   */
  void InitPara(int learner_typeid, ArrayList<double> &param_) {
    // init parameters
    if (learner_typeid == 0) { // SVM_C
      C_ = param_[0];
      b_linear_ = param_[2]>0.0 ? false: true; // whether it's a linear learner
      n_iter_ = (index_t)param_[3];
      n_iter_ = n_iter_ < MAX_NUM_ITER_SGD? n_iter_: MAX_NUM_ITER_SGD;
      accuracy_ = param_[4];
    }
    else if (learner_typeid == 1) { // SVM_R
    }
  }

  void Train(int learner_typeid, const Dataset* dataset_in);

  Kernel& kernel() {
    return kernel_;
  }

  double Bias() const {
    return bias_;
  }

  Vector* W() {
    return &w_;
  }

  double ScaleW() const {
    return scale_w_;
  }

  void GetSV(ArrayList<index_t> &dataset_index, ArrayList<double> &coef, ArrayList<bool> &sv_indicator);

 private:
  /**
   * Loss functions
   */
  double LossFunction_(int learner_typeid, double yy_hat) {
    if (learner_typeid_ == 0) { // SVM_C
      return HingeLoss_(yy_hat);
    }
    else if (learner_typeid_ == 1) { // SVM_R
      return 0.0; // TODO
    }
    else
      return HingeLoss_(yy_hat);
  }

  /**
   * Gradient of loss functions
   */
  double LossFunctionGradient_(int learner_typeid, double yy_hat) {
    if (learner_typeid_ == 0) { // SVM_C
      return HingeLossGradient_(yy_hat);
    }
    else if (learner_typeid_ == 1) { // SVM_R
      return 0.0; // TODO
    }
    else {
      if (yy_hat < 1.0)
	return 1.0;
      else
	return 0.0;
    }
  }

  /**
   * Hinge Loss function
   */
  double HingeLoss_(double yy_hat) {
    if (yy_hat < 1.0)
      return 1.0 - yy_hat;
    else
      return 0.0;
  }
  
  /**
   * Gradient of the Hinge Loss function
   */
  double HingeLossGradient_(double yy_hat) {
    if (yy_hat < 1.0)
      return 1.0;
    else
      return 0.0;
  }

  void LearnersInit_(int learner_typeid);

  int TrainIteration_();

  double GetC_(index_t i) {
    return C_;
  }

  /**
   * Calculate kernel values
   */
  double CalcKernelValue_(index_t i, index_t j) {
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
void SGD<TKernel>::LearnersInit_(int learner_typeid) {
  index_t i;
  learner_typeid_ = learner_typeid;
  rho_ = fx_param_double(NULL, "rho", 1.0); // specify the soft margin. default value 1.0: hard margin
  
  if (learner_typeid_ == 0) { // SVM_C
    if (b_linear_) { // linear SVM
      w_.Init(n_features_);
      w_.SetZero();

      coef_.Init(0); // not used, plain init
    }
    else { // nonlinear SVM
      n_alpha_ = n_data_;
      coef_.Init(n_alpha_);
      coef_.SetZero();

      w_.Init(0); // not used, plain init
    }

    y_.Init(n_data_);
    for (i = 0; i < n_data_; i++) {
      y_[i] = datamatrix_.get(datamatrix_.n_rows()-1, i) > 0 ? 1 : -1;
    }
  }
  else if (learner_typeid_ == 1) { // SVM_R
    // TODO
    n_alpha_ = 2 * n_data_;
    
    coef_.Init(n_alpha_);
    coef_.SetZero();

    y_.Init(n_alpha_);
    for (i = 0; i < n_data_; i++) {
      y_[i] = 1; // -> alpha_i
      y_[i + n_data_] = -1; // -> alpha_i^*
    }
  }
  else if (learner_typeid_ == 2) { // SVM_DE
    // TODO
  }
}


/**
* Steepest descent based SGD training for 2-classes
*
* @param: input 2-classes data matrix with labels (1,-1) in the last row
*/
template<typename TKernel>
void SGD<TKernel>::Train(int learner_typeid, const Dataset* dataset_in) {
  /* general learner-independent initializations */
  dataset_ = dataset_in;
  datamatrix_.Alias(dataset_->matrix());
  n_data_ = datamatrix_.n_cols();
  n_features_ = datamatrix_.n_rows() - 1;
  
  DEBUG_ASSERT(C_ != 0);
  lambda_ = 1.0/(C_*n_data_);
  bias_ = 0.0;
  

  /* learners initialization */
  LearnersInit_(learner_typeid);

  /* Begin SGD iterations */
  if (b_linear_) { // linear SVM, output: w, bias

    double sqrt_n = sqrt(n_data_);
    double eta0 = sqrt_n / max(1.0, LossFunctionGradient_(learner_typeid, -sqrt_n)); // initial step length
    t_ = 1.0 / (eta0 * lambda_);

    scale_w_ = 0.0;
    //for (index_t ct=0; ct<n_data_; ct++) {
    double eta_grad = INFINITY;
    index_t work_idx = 0;
    index_t ct = 0;
    //while (ct<=n_iter_ && fabs(eta_grad)>=accuracy_) {
    while (ct<=n_iter_) {
      work_idx = ct % n_data_;
      eta_ = 1.0 / (lambda_ * t_); // update step length
      scale_w_ = scale_w_ - scale_w_ / t_; // update scale of w
      if (scale_w_ < SCALE_W_TOLERANCE) {
	la::Scale(scale_w_, &w_);
	scale_w_ = 1.0;
      }
      Vector xt;
      datamatrix_.MakeColumnSubvector(work_idx, 0, n_features_, &xt);
      double yt = y_[work_idx];
      double yt_hat = la::Dot(w_, xt) * scale_w_ + bias_;
      double yy_hat = yt * yt_hat;
      if (yy_hat < 1.0) {
	// standard Stochastic Gradient Descent
        eta_grad = eta_ * LossFunctionGradient_(learner_typeid, yy_hat) * yt; // also need *xt, but it's done in next line
	la::AddExpert(eta_grad/ scale_w_, xt, &w_); // update w by Steepest Descent: w_{t+1} = w_t - eta * loss_gradient * xt
	
	// Pegasos Stochastic Gradient Descent
	//eta_grad = ; 
	
	bias_ += eta_grad * 0.01; // update bias
      }
      t_ += 1.0;
      ct ++;
    }
  }
  else { // nonlinear SVM, output: coefs(i.e. alpha*y), bias
    // it's more expensive to calc the accuracy then linear SVM, so we just use n_iter_ as stop criterion
    index_t work_idx = 0;
    index_t ct = 0;

    // initial step length
    //double sqrt_n = sqrt(n_data_);
    //double eta0 = sqrt_n / max(1.0, LossFunctionGradient_(learner_typeid, -sqrt_n)); // initial step length
    //t_ = 1.0 / (eta0 * lambda_);

    //double eta0 = 1.0 / (2*lambda_);
    t_ = 1.0;

    Vector coef_long;
    coef_long.Init(n_iter_);
    coef_long.SetZero();

    double delta;

    while (ct < n_iter_) {
      work_idx = ct % n_data_;
      //work_idx = rand() % n_data_;

      double yt = y_[work_idx];
      double yt_hat = 0.0;
      for (index_t i=0; i<ct; i++) {
	yt_hat += coef_long[i] * CalcKernelValue_(i%n_data_, work_idx);
      }
      yt_hat += bias_;
      double yy_hat = yt * yt_hat;

      // update step length
      eta_ = 1.0 / (lambda_ * t_);
      //eta_ = eta0 / sqrt(t_);

      if (ct >= 1)
	bias_ += coef_long[ct-1];

      // update old coefs (for i<t)
      double one_minus_eta_lambda = 1.0 - eta_ * lambda_;
      //printf("%d: %f\n", ct, one_minus_eta_lambda);
      for (index_t i=0; i<ct; i++)
	coef_long[i] = coef_long[i] * one_minus_eta_lambda; 

      // update current coef (for i==t)

      //coef_long[ct] = eta_ * LossFunctionGradient_(learner_typeid, yy_hat) * yt; // Hard margin SVM
      //printf("%f, %f, %f\n", eta_, LossFunctionGradient_(learner_typeid, yy_hat), yt);

      // soft margin svm
      if (yy_hat <= rho_) {
	//printf("%d: %f, %f\n", ct, yy_hat, yt);
	delta = 1.0;
      }
      else {
	delta = 0.0;
      }
      coef_long[ct] = eta_ * delta * yt;

      // update bias
      //bias_ += coef_long[ct];
 
      t_ += 1.0;
      ct ++;
    }
    // convert coef_long to coef_
    for (index_t i=0; i<n_iter_; i++) {
      work_idx = i % n_data_;
      coef_[work_idx] = coef_[work_idx] + coef_long[i];
    }
  }
}


/* Get SVM results: coefficients(alpha*y), number and indecies of SVs, for nonlinear SVM only
*
* @param: sample indices of the training (sub)set in the total training set
* @param: support vector coefficients: alpha*y
* @param: bool indicators  FOR THE TRAINING SET: is/isn't a support vector
*
*/
template<typename TKernel>
void SGD<TKernel>::GetSV(ArrayList<index_t> &dataset_index, ArrayList<double> &coef, ArrayList<bool> &sv_indicator) {
  n_sv_ = 0;
  if (learner_typeid_ == 0) {// SVM_C
    for (index_t i = 0; i < n_data_; i++) {
      if (fabs(coef_[i]) >= SGD_ALPHA_ZERO) { // support vectors found
	coef.PushBack() = coef_[i];
	sv_indicator[dataset_index[i]] = true;
	n_sv_++;
      }
      else {
	coef.PushBack() = 0;
      }
    }
    printf("Number of support vectors: %d.\n", n_sv_);
  }
  else if (learner_typeid_ == 1) {// SVM_R
    // TODO
    /*
    for (index_t i = 0; i < n_data_; i++) {
      double alpha_diff = -alpha_[i] + alpha_[i+n_data_]; // alpha_i^* - alpha_i
      if (fabs(alpha_diff) >= SGD_ALPHA_ZERO) { // support vectors found
	coef.PushBack() = alpha_diff; 
	sv_indicator[dataset_index[i]] = true;
	n_sv_++;
      }
      else {
	coef.PushBack() = 0;
      }
    }
    */
  }
}

#endif
