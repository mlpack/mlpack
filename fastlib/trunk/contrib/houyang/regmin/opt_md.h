/**
 * @author Hua Ouyang
 *
 * @file opt_md.h
 *
 * This head file contains functions for performing L1-regularized linear loss optimization, using Mirror Descent
 *
 *
 * @see svm.h
 */

#ifndef U_SVM_OPT_MD_H
#define U_SVM_OPT_MD_H

#include "fastlib/fastlib.h"

const double MD_ZERO = 1.0e-30;

template<typename TKernel>
class MD {
  FORBID_ACCIDENTAL_COPIES(MD);

 public:
  typedef TKernel Kernel;

 private:
  int learner_typeid_;

  Kernel kernel_;
  const Dataset_sl *dataset_;
  index_t n_data_; /* number of data samples */
  index_t n_features_; /* # of features == # of row - 1, exclude the last row (for labels) */
  //  index_t n_features_bias_; /* # of features + 1 , [x, 1], for the bias term */

  index_t n_sv_; /* number of support vectors */

  index_t w_nnz_;

  double round_thd_;
  
  ArrayList<int> y_; /* list that stores "labels" */

  ArrayList<NZ_entry> w_; /* the slope of the decision hyperplane, including bias: [w, b] */
  ArrayList<NZ_entry> w_p_; /* coefficients for positive w_t+ */
  ArrayList<NZ_entry> w_n_; /* coefficients for negative w_t- */

  double scale_w_; // the scale for w

  // parameters
  double C_; // \|w\|_1^1 \leq C
  double lambda_; // regularization parameter. lambda = 1/(C*n_data)
  index_t n_iter_; // number of iterations
  index_t n_epochs_; // number of epochs
  double accuracy_; // accuracy for stopping creterion
  double eta_; // step length. eta = 1/(lambda*t)
  double t_;

  //bool is_constant_step_size_; // whether use constant step size (default) or not

  ArrayList<index_t> old_from_new_; // for generating a random sequence of training data

 public:
  MD() {}
  ~MD() {}

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

  void Train(int learner_typeid, Dataset_sl &dataset_in);

  Kernel& kernel() {
    return kernel_;
  }

  void GetW(ArrayList<NZ_entry> &w_out) {
    index_t wp_size = w_p_.size();
    index_t ct_nz = 0;
    w_out.Init(w_nnz_);
    for (index_t i=0; i<wp_size; i++) {
      if (fabs(w_p_[i].value) >= round_thd_ ) {
	w_out[ct_nz].index = w_p_[i].index;
	w_out[ct_nz].value = w_p_[i].value;
	ct_nz ++;
      }
    }
  }
  
  double ScaleW() const {
    return scale_w_;
  }

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
      //return LogisticLossGradient_(yy_hat);
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

  /**
   * Gradient of the Logistic Loss function
   */
  double LogisticLossGradient_(double yy_hat) {
    double tmp = exp(-yy_hat);
    return -tmp/(1+tmp);
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
void MD<TKernel>::LearnersInit_(int learner_typeid) {
  index_t i;
  double C_iv_f = C_/(n_features_);
  double C_iv_2f = C_iv_f/2.0;
  
  learner_typeid_ = learner_typeid;
  
  // init w, w+, w-
  w_.Init(0);
  w_p_.Init(n_features_);
  w_n_.Init(n_features_);
  for (i=0; i<n_features_; i++) {
    w_p_[i].index = i;
    w_p_[i].value = C_iv_2f; // TODO
    w_n_[i].index = i;
    w_n_[i].value = C_iv_2f; // TODO
  }
  
  y_.Init(n_data_);
  for (i = 0; i < n_data_; i++) {
    y_[i] = (dataset_->y)[i] > 0 ? 1 : -1;
  }
}


/**
* L1-regularization training for 2-classes
*
* @param: input 2-classes data matrix with labels (1,-1) in the last row
*/
template<typename TKernel>
void MD<TKernel>::Train(int learner_typeid, Dataset_sl &dataset_in) {
  index_t i, j, epo, ct;

  index_t total_n_iter;
  //double DX, M, x_sq_sup;
  double cons_step, w_sum = 1.0;

  /* general learner-independent initializations */
  dataset_ = &dataset_in;
  n_data_ = dataset_->n_points;
  n_features_ = dataset_->n_features;

  printf("n_data=%d,n_feature=%d\n", n_data_, n_features_);

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
  double x_sq_sp = 0.0;
  NZ_entry *xs;
  xs = (dataset_->x)[0];
  while (xs->index != -1) {
    x_sq_sp += math::Sqr(xs->value);
    ++xs;
  }
  //eta_ = 1/(4.0*x_sq_sp);
  t_ = 4.0 * x_sq_sp;

  cons_step = 1;

  index_t work_idx_old = 0;

  /* Begin training iterations */
  double yt, yt_hat, yy_hat;
  scale_w_ = 1.0;
  
  printf("MD training begins...\n");
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
      //eta_ = 1.0 /(lambda_ * sqrt(t_));
      //eta_ = 1.0 / (lambda_ * t_);
      //eta_ = 1.0 / t_;
      eta_ = 1.0 /sqrt(t_);
      
      NZ_entry *xt;
      xt = (dataset_->x)[work_idx_old];
      yt = y_[work_idx_old];
      yt_hat = SparseDot(w_p_, xt) - SparseDot(w_n_, xt);
      yy_hat = yt * yt_hat;

      double eta_grad_tmp = eta_* yt * LossFunctionGradient_(learner_typeid, yy_hat);
      //printf("epo:%d, ct:%d, eta_grad:%f\n", epo, ct, eta_grad_tmp);
      if (eta_grad_tmp != 0) {
	xt = (dataset_->x)[work_idx_old];
	index_t ct_w = 0;

	while (ct_w < w_p_.size() && xt->index != -1) {
	  if (w_p_[ct_w].index < xt->index) {
	    ++ct_w;
	  }
	  else if (w_p_[ct_w].index == xt->index) {
	    w_p_[ct_w].value = w_p_[ct_w].value * exp(-eta_grad_tmp * xt->value);
	    ++ct_w;
	    ++xt;
	  }
	  else { // w_p_[ct_w].index > xt->index
	    ++xt;
	  }
	}
	ct_w = 0;
	xt = (dataset_->x)[work_idx_old];
	while (ct_w < w_n_.size() && xt->index != -1) {
	  if (w_n_[ct_w].index < xt->index) {
	    ++ct_w;
	  }
	  else if (w_n_[ct_w].index == xt->index) {
	    w_n_[ct_w].value = w_n_[ct_w].value * exp(eta_grad_tmp * xt->value);
	    ++ct_w;
	    ++xt;
	  }
	  else { // w_n_[ct_w].index > xt->index
	    ++xt;
	  }
	}

	// calc sum_i(w_p_i+w_n_i)
	w_sum = 0.0;
	for (i=0; i<w_p_.size(); i++) {
	  w_sum += w_p_[i].value;
	}
	for (i=0; i<w_n_.size(); i++) {
	  w_sum += w_n_[i].value;
	}

	if (w_sum > C_) {
	  // printf("epo:%d,iter:%d, w_sum=%f\n", epo, ct, w_sum);
	  SparseScale(C_/w_sum, w_p_);
	  SparseScale(C_/w_sum, w_n_);
	}

      }

      //printf("epo:%d, ct:%d-- sacle_w:%f\n", epo, ct, scale_w_);

      /*
      for (i=0; i<w_p_.size(); i++)
	printf("w_p[%d]=%f\n", i, w_p_[i].value);
      for (i=0; i<w_n_.size(); i++)
      	printf("w_n[%d]=%f\n", i, w_n_[i].value);
      for (i=0; i<w_.size(); i++)
	printf("w[%d]=%f\n", i, w_[i].value);
      */

      t_ += 1.0;
      ct ++;
    }
  }// for epo

  SparseSubOverwrite(w_n_, w_p_); // w_p <= w_p - w_n

  // rounding w
  index_t wp_size = w_p_.size();
  round_thd_ = fx_param_double(NULL, "thd", 1.0e-5);
  w_nnz_ = 0;
  for (ct=0; ct<wp_size; ct++) {
    if (fabs(w_p_[ct].value) >= round_thd_) { // TODO: thresholding
      w_nnz_++;
    }
  }
  
  printf("%d out of %d features are non zero. NZ rate:%f\n", w_nnz_, n_features_, (double)(w_nnz_)/(double)n_features_);

  /*
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
    //printf("w[%d]=%f\n", i, w_[i].value);
    if ( fabs(w_[i].value) > round_thd ) {
      w_ct ++;
      //printf("w_dim:%d, w_value:%lf\n", i, w_[i]);
    }
    else {
      w_.Remove(i);
    }
  }
  printf("%d out of %d features are non zero\n", w_ct, n_features_);
  */
  
  // Calculate objective value; default: no calculation to save time
  int objvalue = fx_param_int(NULL, "objvalue", 0);
  if (objvalue > 0) {
    double hinge_loss = 0.0, loss_sum= 0.0;
    
    // primal objective value
    for (i=0; i< n_data_; i++) {
      hinge_loss = 1- y_[i] * SparseDot(w_p_, (dataset_->x)[i]);
      if (hinge_loss > 0) {
	loss_sum += hinge_loss;
      }
    }
    
    printf("Primal objective value: %lf\n", loss_sum);
  }

}

#endif


