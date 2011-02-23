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

// tolerance of sacale_w
const double SGD_SCALE_W_TOLERANCE = 1.0e-9;
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
  const Dataset_sl *dataset_;
  index_t n_data_; /* number of data samples */
  index_t n_features_; /* # of features == # of row - 1, exclude the last row (for labels) */

  Vector coef_; /* alpha*y, to be optimized */
  index_t n_alpha_; /* number of lagrangian multipliers in the dual */
  index_t n_sv_; /* number of support vectors */
  
  index_t i_cache_, j_cache_; /* indices for the most recently cached kernel value */
  double cached_kernel_value_; /* cache */

  ArrayList<int> y_; /* list that stores "labels" */
  ArrayList<NZ_entry> w_;
  double bias_;
  double scale_w_; // the scale for w

  // parameters
  double C_; // for SVM_C
  double epsilon_; // for SVM_R
  bool b_linear_; // whether it's a linear SVM
  double lambda_; // regularization parameter. lambda = 1/(C*n_data)
  index_t n_iter_; // number of iterations
  index_t n_epochs_; // number of epochs
  double accuracy_; // accuracy for stopping creterion
  double eta_; // step length. eta = 1/(lambda*t)
  double t_;

  ArrayList<index_t> old_from_new_; // for generating a random sequence of training data
  //ArrayList<index_t> new_from_old_; // for generating a random sequence of training data

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
      n_epochs_ = (index_t)param_[3];
      n_iter_ = (index_t)param_[4];
      accuracy_ = param_[5];
    }
    else if (learner_typeid == 1) { // SVM_R
    }
  }

  void Train(int learner_typeid, const Dataset_sl &dataset_in);

  Kernel& kernel() {
    return kernel_;
  }

  double Bias() const {
    return bias_;
  }

  void GetW(ArrayList<NZ_entry> &w_out) {
    w_out.InitCopy(w_, w_.size());
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
      return -1.0;
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

    // Do Caching. Store the recently caculated kernel values.
    //i_cache_ = i;
    //j_cache_ = j;

    cached_kernel_value_ = kernel_.Eval((dataset_->x)[i], (dataset_->x)[j]);

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
  
  w_.Init(0); // all coefs of w_ are init as 0

  if (learner_typeid_ == 0) { // SVM_C
    if (b_linear_) { // linear SVM
      coef_.Init(0); // not used, plain init
    }
    else { // nonlinear SVM
      n_alpha_ = n_data_;
      coef_.Init(n_alpha_);
      coef_.SetZero();

      //w_.Init(0); // not used, plain init
    }

    y_.Init(n_data_);
    for (i = 0; i < n_data_; i++) {
      y_[i] = (dataset_->y)[i] > 0 ? 1 : -1;
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
void SGD<TKernel>::Train(int learner_typeid, const Dataset_sl &dataset_in) {
  index_t i, j, epo, ct;
  // Load data
  dataset_ = &dataset_in;
  n_data_ = dataset_->n_points;
  n_features_ = dataset_->n_features;
  
  /* general learner-independent initializations */
  if (n_epochs_ > 0) { // # of epochs provided, use it
    n_iter_ = n_data_;
  }
  else { // # of epochs not provided, use n_iter_ to count iterations
    n_epochs_ = 1; // not exactly one epoch, just use it for one loop
  }
  
  DEBUG_ASSERT(C_ != 0);
  lambda_ = 1.0/(C_*n_data_);
  bias_ = 0.0;

  /* learners initialization */
  LearnersInit_(learner_typeid);
  old_from_new_.Init(n_data_);

  index_t work_idx_old = 0;

  /* Begin SGD iterations */
  if (b_linear_) { // linear SVM, output: w, bias
    double yt, yt_hat, yy_hat;
    double sqrt_n = sqrt(n_data_);
    double eta0 = sqrt_n / max(1.0, LossFunctionGradient_(learner_typeid, -sqrt_n)); // initial step length
    t_ = 1.0 / (eta0 * lambda_);
    scale_w_ = 1.0;

    printf("SGD_linear training begins...\n");
    for (epo = 0; epo<n_epochs_; epo++) {
      /* To mimic the online learning senario, in each epoch, 
	 we randomly permutate the training set, indexed by old_from_new_ */
      for (i=0; i<n_data_; i++) {
	old_from_new_[i] = i; 
      }
      for (i=0; i<n_data_; i++) {
	j = rand() % n_data_;
	swap(old_from_new_[i], old_from_new_[j]);
      }

      ct = 0;
      while (ct <= n_iter_) {
	work_idx_old = old_from_new_[ct % n_data_];

	eta_ = 1.0 / (lambda_ * t_); // update step size
	scale_w_ = scale_w_ - scale_w_ / t_; // update scale of w
	//la::Scale(scale_w, &w_); // Note: moving w's scaling calculation to the testing session is faster

	if (scale_w_ < SGD_SCALE_W_TOLERANCE) {
	  SparseScale(scale_w_, w_);
	  scale_w_ = 1.0;
	  //printf("epo %d: scale_w tolerance reached.\n", epo);
	}
	NZ_entry *xt;
	xt = (dataset_->x)[work_idx_old];
	//print_ex(xt);
	yt = y_[work_idx_old];
	//yt_hat = la::Dot(w_, xt) * scale_w_ + bias_;
	yt_hat = SparseDot(w_, xt) * scale_w_ + bias_;
	yy_hat = yt * yt_hat;

	// update w by Stochastic Gradient Descent: w_{t+1} = w_t - eta_t * grad_t = (1-eta*lambda) * w_t + eta * [yt*xt]^+
	double eta_grad = eta_ * LossFunctionGradient_(learner_typeid, yy_hat) * yt;
	if (eta_grad != 0) {
	  // Note: moving w's scaling calculation w_t*(1-1/t) to the testing session is faster
	  SparseAddExpert(-eta_grad / scale_w_, xt, w_);
	  // update bias
	  //printf("%g-> ", bias_);
	  bias_ -= eta_grad * 0.01;
	  //printf("%g\n", bias_);
	}

	t_ += 1.0;
	ct ++;
	//print_svec(w_);
      }
    } // for epo

    // Calculate objective value; default: no calculation to save time
    int objvalue = fx_param_int(NULL, "objvalue", 0);
    if (objvalue > 0) {
      double v = 0.0, hinge_loss = 0.0, loss_sum= 0.0;
      
      // primal objective value
      for (i=0; i< n_data_; i++) {
	NZ_entry *xt;
	xt = (dataset_->x)[work_idx_old];
	//hinge_loss = 1- y_[i] * (scale_w_ * la::Dot(w_, xt) + bias_);
	hinge_loss = 1- y_[i] * (scale_w_ * SparseDot(w_, xt) + bias_);
	if (hinge_loss > 0) {
	  loss_sum += hinge_loss * C_;
	}
      }
      
      for (j=0; j<w_.size(); j++) {
	v += w_[j].value * w_[j].value;
      }
      v = v * scale_w_ * scale_w_ / 2.0 + loss_sum;
      
      printf("Primal objective value: %lf\n", v);


      // find max(abs(w_i))
      double wi_abs_max = -INFINITY;
      double wi_abs;
      for (i=0; i<w_.size(); i++) {
	wi_abs = fabs(w_[i].value);
	if (wi_abs > wi_abs_max) {
	  wi_abs_max = wi_abs;
	}
      }
      // round small w_i to 0
      index_t w_ct = 0;
      double round_factor = fx_param_double(NULL, "round_factor", 1.0e32);
      double round_thd = wi_abs_max / round_factor;
      for (i=0; i<w_.size(); i++) {
	if ( fabs(w_[i].value) > round_thd ) {
	  w_ct ++;
	}
	else {
	  w_.Remove(i);
	}
      }
      printf("%d out of %d features are non zero\n", w_ct, n_features_);
    }

  }
  else { // nonlinear SVM, output: coefs(i.e. alpha*y), bias
    // it's more expensive to calc the accuracy than linear SVM, so we just use n_iter_ as stop criterion
    double delta;
    double yt, yt_hat, yy_hat;
    double one_minus_eta_lambda;
    index_t n_real_epo, n_data_res;
    double kernel_value;
    
    Vector b_calc_kernel;
    b_calc_kernel.Init(n_data_);
    b_calc_kernel.SetAll(0);

    Vector coef_long;
    n_iter_ = n_iter_ * n_epochs_;
    coef_long.Init(n_iter_);

    // initial step length
    //double sqrt_n = sqrt(n_data_);
    //double eta0 = sqrt_n / max(1.0, LossFunctionGradient_(learner_typeid, -sqrt_n)); // initial step length
    //t_ = 1.0 / (eta0 * lambda_);
    //double eta0 = 1.0 / (2*lambda_);

    /* To mimic the online learning senario, we randomly permutate the training set, indexed by old_from_new_ */
    for (i=0; i<n_data_; i++) {
      old_from_new_[i] = i; 
    }
    for (i=0; i<n_data_; i++) {
      j = rand() % n_data_;
      swap(old_from_new_[i], old_from_new_[j]);
    }

    t_ = 1.0;
    ct = 0;

    printf("SGD_kernel training begins...\n");
    while (ct < n_iter_) {
      work_idx_old = old_from_new_[ct % n_data_];
      
      yt = y_[work_idx_old];
      yt_hat = 0.0;
      
      n_real_epo = index_t(ceil(ct/n_data_));
      n_data_res = ct - n_real_epo * n_data_;
      if (n_real_epo > 0) {
	for (i=0; i<ct; i++) {
	  if (fabs(coef_long[i]) >= SGD_ALPHA_ZERO) {
	    b_calc_kernel[i%n_data_] = 1;
	  }
	}
	for (i=0; i<n_data_res; i++) {
	  if (b_calc_kernel[i]>0) {
	    kernel_value = CalcKernelValue_(old_from_new_[i], work_idx_old);
	    for (j=0; j<n_real_epo+1; j++) {
	      yt_hat += coef_long[j*n_data_+i] * kernel_value;
	    }
	  }
	}
	for (i=n_data_res; i<n_data_; i++) {
	  if (b_calc_kernel[i]>0) {
	    kernel_value = CalcKernelValue_(old_from_new_[i], work_idx_old);
	    for (j=0; j<n_real_epo; j++) {
	      yt_hat += coef_long[j*n_data_+i] * kernel_value;
	    }
	  }
	}
      }
      else { // ct < n_data_
	for (i=0; i<ct; i++) {
	  if (fabs(coef_long[i]) >= SGD_ALPHA_ZERO) {
	    yt_hat += coef_long[i] * CalcKernelValue_(old_from_new_[i], work_idx_old);
	  }
	}
      }
      
      yt_hat += bias_;
      yy_hat = yt * yt_hat;
      
      // update step length
      eta_ = 1.0 / (lambda_ * t_);
      //eta_ = eta0 / sqrt(t_);
      
      if (ct >= 1) {
	bias_ += coef_long[ct-1];
      }
      
      // update old coefs (for i<t)
      one_minus_eta_lambda = 1.0 - eta_ * lambda_;
      //printf("%d: %f\n", ct, one_minus_eta_lambda);
      for (i=0; i<ct; i++) {
	coef_long[i] = coef_long[i] * one_minus_eta_lambda; 
      }
      
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
    for (i=0; i<n_iter_; i++) {
      work_idx_old = old_from_new_[i % n_data_];
      coef_[work_idx_old] = coef_[work_idx_old] + coef_long[i];
    }
  } // else
}


/* Get results for nonlinear SGD: coefficients(alpha*y), number and indecies of SVs
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

// ./regmin --learner_name=svm_c --mode=train_test --train_data=heart_scale --test_data=heart_scale --opt=sgd --kernel=linear --normalize=0 --objvalue=1 --c=0.1 --n_iter=4000

// ./regmin --learner_name=svm_c --mode=train_test --train_data=heart_scale --test_data=heart_scale --opt=sgd --kernel=gaussian --normalize=0 --objvalue=1 --sigma=1 --c=10 --n_iter=5000

// ./regmin --learner_name=svm_c --mode=train_test --train_data=real_sim --test_data=real_sim --opt=sgd --kernel=linear --normalize=0 --objvalue=0 --c=1 --n_iter=140000
