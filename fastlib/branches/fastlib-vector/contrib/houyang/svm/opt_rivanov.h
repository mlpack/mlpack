/**
 * @author Hua Ouyang
 *
 * @file opt_rivanov.h
 *
 * This head file contains functions for performing Robust Ivanov optimization for linear SVM
 *
 *
 * @see svm.h
 */

#ifndef U_SVM_OPT_RIVANOV_H
#define U_SVM_OPT_RIVANOV_H

#include "fastlib/fastlib.h"

// tolerance of sacale_w
//const double SCALE_W_TOLERANCE = 1.0e-9;
// threshold that determines whether an alpha is a SV or not
const double RIVANOV_ALPHA_ZERO = 1.0e-7;


template<typename TKernel>
class RIVANOV {
  FORBID_ACCIDENTAL_COPIES(RIVANOV);

 public:
  typedef TKernel Kernel;

 private:
  int learner_typeid_;

  Kernel kernel_;
  const Dataset *dataset_;
  index_t n_data_; /* number of data samples */
  index_t n_features_; /* # of features == # of row - 1, exclude the last row (for labels) */
  index_t n_features_bias_; /* # of features + 1 , [x, 1], for the bias term */
  Matrix datamatrix_; /* alias for the data matrix */

  Vector coef_; /* alpha*y, to be optimized */
  index_t n_alpha_; /* number of lagrangian multipliers in the dual  */
  index_t n_sv_; /* number of support vectors */
  
  index_t i_cache_, j_cache_; /* indices for the most recently cached kernel value */
  double cached_kernel_value_; /* cache */

  ArrayList<int> y_; /* list that stores "labels" */

  Vector w_; /* the slope of the decision hyperplane, including bias: [w, b] */
  double bias_;
  //double scale_w_; // the scale for w

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

  bool is_constant_step_size_; // whether use constant step size (default) or not

  ArrayList<index_t> old_from_new_; // for generating a random sequence of training data

 public:
  RIVANOV() {}
  ~RIVANOV() {}

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

  void Train(int learner_typeid, const Dataset* dataset_in);

  Kernel& kernel() {
    return kernel_;
  }

  Vector* GetW() {
    return &w_;
  }
  
  /*
  double ScaleW() const {
    return scale_w_;
  }
  */

  //void GetSV(ArrayList<index_t> &dataset_index, ArrayList<double> &coef, ArrayList<bool> &sv_indicator);

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

};


/**
 * Initialization according to different SVM learner types
 *
 * @param: learner type id 
 */
template<typename TKernel>
void RIVANOV<TKernel>::LearnersInit_(int learner_typeid) {
  index_t i;
  learner_typeid_ = learner_typeid;
  
  if (learner_typeid_ == 0) { // SVM_C
    if (b_linear_) { // linear SVM
      w_.Init(n_features_bias_);
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
* Robust Ivanov training for 2-classes
*
* @param: input 2-classes data matrix with labels (1,-1) in the last row
*/
template<typename TKernel>
void RIVANOV<TKernel>::Train(int learner_typeid, const Dataset* dataset_in) {
  index_t i, j, epo, ct;

  index_t total_n_iter;
  double DX, M, x_sq_sup;
  double cons_step, eta_sum;

  /* general learner-independent initializations */
  dataset_ = dataset_in;
  datamatrix_.Alias(dataset_->matrix());
  n_data_ = datamatrix_.n_cols();
  n_features_ = datamatrix_.n_rows() - 1;
  n_features_bias_ = n_features_ + 1;

  if (n_epochs_ > 0) { // # of epochs provided, use it
    n_iter_ = n_data_;
    total_n_iter = n_iter_ * n_epochs_;
  }
  else { // # of epochs not provided, use n_iter_ to count iterations
    n_epochs_ = 1; // not exactly one epoch, just use it for one loop
    total_n_iter = n_iter_;
  }
  
  DEBUG_ASSERT(C_ != 0);
  lambda_ = 1.0/(C_*n_data_);
  bias_ = 0.0;

  /* learners initialization */
  LearnersInit_(learner_typeid);
  old_from_new_.Init(n_data_);

  // determine step sizes
  DX = sqrt( sqrt( C_*n_data_ ) );
  x_sq_sup = 0.0;
  for (i=0; i<n_features_ ; i++) {
    x_sq_sup += math::Sqr(datamatrix_.get(i, 0));
  }
  M = sqrt(x_sq_sup);
  is_constant_step_size_ = fx_param_int(NULL, "constant_step", 1); // default using constant step size
  if (is_constant_step_size_) {
    cons_step = DX / (M * sqrt(total_n_iter));
  }
  else {
    cons_step = DX / M;
  }

  eta_sum = 0.0;
  Vector eta_w_sum;
  eta_w_sum.Init(n_features_bias_);

  index_t work_idx_old = 0;

  /* Begin Robust Ivanov iterations */
  if (b_linear_) { // linear SVM, output: w, bias
    double yt, yt_hat, yy_hat;
    double eta_grad = 0;
    t_ = 1.0;

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

	if (is_constant_step_size_) {
	  eta_ = cons_step; // constant step size
	}
	else {
	  eta_ = cons_step / sqrt(t_);
	}
	eta_sum += eta_;

	Vector xt;
	datamatrix_.MakeColumnVector(work_idx_old, &xt);
	xt[n_features_] = 1.0; // for bias term: x <- [x,1], w <- [w, b]
	yt = y_[work_idx_old];
	yt_hat = la::Dot(w_, xt);
	yy_hat = yt * yt_hat;

	// w_{t+1} = w_t + eta * [yt*xt]^+
	if (yy_hat < 1.0) {
	    eta_grad = eta_ * LossFunctionGradient_(learner_typeid, yy_hat) * yt; // also need *xt, but it's done in next line
	    la::AddExpert(eta_grad, xt, &w_);
	}
	
	// Do projection
	double w_norm_sq = 0.0;
	for (i=0; i<w_.length() ; i++) {
	  w_norm_sq += math::Sqr(w_[i]);
	}
	if (w_norm_sq * lambda_ > 1.0) {
	  //printf("epo:%d, projection\n", epo);
	  la::Scale( sqrt(1.0/ (lambda_*w_norm_sq)), &w_);
	}

	// update eta_w_sum
	la::AddExpert(eta_, w_, &eta_w_sum);
	
	t_ += 1.0;
	ct ++;
      }
    }// for epo
    la::ScaleOverwrite(1.0/eta_sum, eta_w_sum, &w_);

    // Calculate objective value; default: no calculation to save time
    int objvalue = fx_param_int(NULL, "objvalue", 0);
    if (objvalue > 0) {
      double v = 0.0, hinge_loss = 0.0, loss_sum= 0.0;
      
      // primal objective value
      for (i=0; i< n_data_; i++) {
	Vector xt;
	datamatrix_.MakeColumnVector(i, &xt);
	xt[n_features_] = 1.0; // for bias term: x <- [x,1], w <- [w, b]
	hinge_loss = 1- y_[i] * la::Dot(w_, xt);
	if (hinge_loss > 0) {
	  loss_sum += hinge_loss * C_;
	}
      }
      v = loss_sum / C_;
      
      printf("Primal objective value: %lf\n", v);
    }

  }
  else { // nonlinear SVM, output: coefs(i.e. alpha*y), bias
    // TODO
  }
}

#endif
