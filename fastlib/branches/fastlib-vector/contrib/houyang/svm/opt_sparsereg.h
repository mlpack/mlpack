/**
 * @author Hua Ouyang
 *
 * @file opt_sparsereg.h
 *
 * This head file contains functions for performing L1-regularized linear loss optimization
 *
 *
 * @see svm.h
 */

#ifndef U_SVM_OPT_SPARSEREG_H
#define U_SVM_OPT_SPARSEREG_H

#include "fastlib/fastlib.h"

const double SPARSEREG_ZERO = 1.0e-30;

template<typename TKernel>
class SPARSEREG {
  FORBID_ACCIDENTAL_COPIES(SPARSEREG);

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

  index_t n_sv_; /* number of support vectors */
  
  ArrayList<int> y_; /* list that stores "labels" */

  Vector w_; /* the slope of the decision hyperplane, including bias: [w, b] */
  Vector w_p_; /* coefficients for positive w_t+ */
  Vector w_n_; /* coefficients for negative w_t- */

  // parameters
  double C_; // for SVM_C
  double lambda_; // regularization parameter. lambda = 1/(C*n_data)
  index_t n_iter_; // number of iterations
  index_t n_epochs_; // number of epochs
  double accuracy_; // accuracy for stopping creterion
  double eta_; // step length. eta = 1/(lambda*t)
  double t_;

  bool is_constant_step_size_; // whether use constant step size (default) or not

  ArrayList<index_t> old_from_new_; // for generating a random sequence of training data

 public:
  SPARSEREG() {}
  ~SPARSEREG() {}

  /**
   * Initialization for parameters
   */
  void InitPara(int learner_typeid, ArrayList<double> &param_) {
    // init parameters
    if (learner_typeid == 0) { // SVM_C
      C_ = param_[0];
      n_epochs_ = (index_t)param_[2];
      n_iter_ = (index_t)param_[3];
      accuracy_ = param_[4];
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
void SPARSEREG<TKernel>::LearnersInit_(int learner_typeid) {
  index_t i;
  learner_typeid_ = learner_typeid;

  w_.Init(n_features_bias_);
  w_.SetZero();
  w_p_.Init(n_features_bias_);
  w_p_.SetAll(0.00001); // TODO
  //w_p_.SetAll(SPARSEREG_ZERO);
  w_n_.Init(n_features_bias_);
  w_n_.SetAll(0.00001); // TODO
  //w_n_.SetAll(SPARSEREG_ZERO);
  
  y_.Init(n_data_);
  for (i = 0; i < n_data_; i++) {
    y_[i] = datamatrix_.get(datamatrix_.n_rows()-1, i) > 0 ? 1 : -1;
  }
}


/**
* L1-regularization training for 2-classes
*
* @param: input 2-classes data matrix with labels (1,-1) in the last row
*/
template<typename TKernel>
void SPARSEREG<TKernel>::Train(int learner_typeid, const Dataset* dataset_in) {
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

  /* Begin training iterations */
  double yt, yt_hat, yy_hat;
  double eta_yt = 0;
  double exp_eta_yt_xt_i = 1, exp_two_eta_yt_xt_i = 1;
  double exp_lambda = exp(lambda_);
  double exp_minus_lambda = 1.0 / exp_lambda;
  double exp_two_lambda = exp_lambda * exp_lambda;
  double exp_two_minus_lambda = 1.0 / exp_two_lambda;
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
      yt_hat = la::Dot(w_, xt); // TODO: sparse dot product
      yy_hat = yt * yt_hat;

      // Update w_{t+1}
      if (yy_hat < 1.0) {
	eta_yt = eta_ * LossFunctionGradient_(learner_typeid, yy_hat) * yt;
      }
      for (i=0; i<n_features_bias_; i++) {
	if (w_p_[i] > SPARSEREG_ZERO || w_n_[i] > SPARSEREG_ZERO) {
	//if (w_p_[i] > SPARSEREG_ZERO && w_n_[i] > SPARSEREG_ZERO) { // more aggressive
	  if (xt[i] > 0 && yy_hat < 1.0) {
	    exp_eta_yt_xt_i = exp(eta_yt * xt[i]);
	    exp_two_eta_yt_xt_i = exp_eta_yt_xt_i * exp_eta_yt_xt_i;
	  }
	  else {
	    exp_eta_yt_xt_i = 1;
	    exp_two_eta_yt_xt_i = 1;
	  }

	  if ( w_p_[i] * exp_two_eta_yt_xt_i * exp_two_minus_lambda >= w_n_[i] ) {
	    w_p_[i] = w_p_[i] * exp_eta_yt_xt_i * exp_minus_lambda;
	    w_n_[i] = w_n_[i] * exp_lambda / exp_eta_yt_xt_i;
	  }
	  else if ( w_p_[i] * exp_two_eta_yt_xt_i * exp_two_lambda <= w_n_[i] ) {
	    w_p_[i] = w_p_[i] * exp_eta_yt_xt_i * exp_lambda;
	    w_n_[i] = w_n_[i] * exp_minus_lambda / exp_eta_yt_xt_i;
	  }
	  else {
	    w_p_[i] = (w_p_[i] * exp_eta_yt_xt_i * exp_minus_lambda + w_n_[i] * exp_lambda / exp_eta_yt_xt_i) / 2.0;
	    w_n_[i] = w_p_[i];
	  }
	  w_[i] = w_p_[i] - w_n_[i];
	}
	else {
	  w_[i] = 0;
	}
      }
      
      // update eta_w_sum
      la::AddExpert(eta_, w_, &eta_w_sum);
      
      t_ += 1.0;
      ct ++;
    }
  }// for epo
  la::ScaleOverwrite(1.0/eta_sum, eta_w_sum, &w_);

  // find max(abs(w_i))
  double wi_abs_max = -INFINITY;
  double wi_abs;
  for (i=0; i<n_features_bias_; i++) {
    wi_abs = fabs(w_[i]);
    if (wi_abs > wi_abs_max) {
      wi_abs_max = wi_abs;
    }
  }
  // round small w_i to 0
  index_t w_ct = 0;
  double round_factor = fx_param_double(NULL, "round_factor", 1.0e32);
  double round_thd = wi_abs_max / round_factor;
  for (i=0; i<n_features_bias_; i++) {
    if ( fabs(w_[i]) > round_thd ) {
      w_ct ++;
      printf("w_dim:%d, w_value:%lf\n", i, w_[i]);
    }
    else {
      w_[i] = 0;
    }
  }
  printf("%d out of %d features are non zero\n", w_ct, n_features_);
  
  // Calculate objective value; default: no calculation to save time
  int objvalue = fx_param_int(NULL, "objvalue", 0);
  if (objvalue > 0) {
    double v = 0.0, hinge_loss = 0.0, loss_sum= 0.0;
    
    // primal objective value
    for (i=0; i< n_data_; i++) {
      Vector xt;
      datamatrix_.MakeColumnVector(i, &xt);
      xt[n_features_] = 1.0; // for bias term: x <- [x,1], w <- [w, b]
      hinge_loss = 1- y_[i] * la::Dot(w_, xt); // TODO: sparse dot product
      if (hinge_loss > 0) {
	loss_sum += hinge_loss * C_;
      }
    }
    v = loss_sum / C_;
    
    printf("Primal objective value: %lf\n", v);
  }
}

#endif
