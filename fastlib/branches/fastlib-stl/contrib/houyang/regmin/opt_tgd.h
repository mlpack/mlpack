/**
 * @author Hua Ouyang
 *
 * @file opt_tgd.h
 *
 * This head file contains functions for performing L1-regularized linear loss optimization, using Truncated Gradient Descent
 *
 *
 * @see svm.h
 */

#ifndef U_SVM_OPT_TGD_H
#define U_SVM_OPT_TGD_H

#include "fastlib/fastlib.h"

const double TGD_ZERO = 1.0e-30;

template<typename TKernel>
class TGD {
  FORBID_ACCIDENTAL_COPIES(TGD);

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

  double scale_w_; // the scale for w

  // parameters
  double C_; // \|w\|_1^1 \leq C
  double g_; // regularization parameter in Langford's paper. g = 1/C
  index_t n_iter_; // number of iterations
  index_t n_epochs_; // number of epochs
  double accuracy_; // accuracy for stopping creterion
  double eta_; // step length. eta = 1/sqrt(t)
  double t_;
  index_t k_; // perform truncation every k iterations

  ArrayList<index_t> old_from_new_; // for generating a random sequence of training data

 public:
  TGD() {}
  ~TGD() {}

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
    index_t w_size = w_.size();
    index_t ct_nz = 0;
    w_out.Init(w_nnz_);
    for (index_t i=0; i<w_size; i++) {
      if (fabs(w_[i].value) >= round_thd_ ) {
	w_out[ct_nz].index = w_[i].index;
	w_out[ct_nz].value = w_[i].value;
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
      //return HingeLossGradient_(yy_hat);
      return LogisticLossGradient_(yy_hat);
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
void TGD<TKernel>::LearnersInit_(int learner_typeid) {
  index_t i;
  
  learner_typeid_ = learner_typeid;
  
  // init w, w+, w-
  w_.Init(n_features_);
  for (i=0; i<n_features_; i++) {
    w_[i].index = i;
    w_[i].value = 0.0; // TODO
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
void TGD<TKernel>::Train(int learner_typeid, Dataset_sl &dataset_in) {
  index_t i, j, epo, ct, ct_k;

  index_t total_n_iter;

  k_ = fx_param_int(NULL, "k", 1);
  g_ = 1.0 / C_;

  /* general learner-independent initializations */
  dataset_ = &dataset_in;
  n_data_ = dataset_->n_points;
  n_features_ = dataset_->n_features;

  if (n_epochs_ > 0) { // # of epochs provided, use it
    n_iter_ = n_data_;
    total_n_iter = n_iter_ * n_epochs_;
  }
  else { // # of epochs not provided, use n_iter_ to count iterations
    n_epochs_ = 1; // not exactly one epoch, just use it for one loop
    total_n_iter = n_iter_;
  }
  
  DEBUG_ASSERT(C_ != 0);

  /* learners initialization */
  LearnersInit_(learner_typeid);
  old_from_new_.Init(n_data_);

  index_t work_idx_old = 0;

  /* Begin training iterations */
  double yt, yt_hat, yy_hat;
  //double sqrt_n = sqrt(n_data_);
  //double eta0 = sqrt_n / max(1.0, LossFunctionGradient_(learner_typeid, -sqrt_n)); // initial step length
  //double eta_grad = 0;
  //t_ = 1.0 / (eta0 * lambda_);
  t_ = 1.0;
  scale_w_ = 1.0;
    
  ct_k = 0;
  printf("TGD training begins...\n");
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
      
      // eta_ = 1.0 / t_;
      eta_ = 1.0 / sqrt(t_);
      
      //scale_w_ = scale_w_ / w_sum;
      
      NZ_entry *xt;
      xt = (dataset_->x)[work_idx_old];
      yt = y_[work_idx_old];
      yt_hat = SparseDot(w_, xt) * scale_w_;
      yy_hat = yt * yt_hat;
      

      double grad_tmp = yt * LossFunctionGradient_(learner_typeid, yy_hat);

      //printf("epo:%d, ct:%d, w_size=%d, y_grad:%f\n", epo, ct, w_.size(), grad_tmp);
      for (i=0; i<n_features_; i++) {
	if (i < xt->index) {
	  if (w_[i].value >= -g_ && w_[i].value <= g_) {
	    w_[i].value = 0;
	  }
	  // else w_[i].value remains unchanged
	}
	else { // i== xt->index
	  double db_tmp = w_[i].value - eta_ * grad_tmp * xt->value;
	  if (db_tmp >= 0.0 && db_tmp <= g_) {
	    if (!(ct_k % k_)) {
	      db_tmp -= eta_ * k_ * g_;
	    }
	    if (db_tmp >=0.0) {
	      w_[i].value = db_tmp;
	    }
	    else {
	      w_[i].value = 0.0;
	    }
	  }
	  else if (db_tmp >= -g_ && db_tmp <= 0.0) { 
	    if (!(ct_k % k_)) {
	      db_tmp += eta_ * k_ * g_;
	    }
	    if (db_tmp <=0.0) {
	      w_[i].value = db_tmp;
	    }
	    else {
	      w_[i].value = 0.0;
	    }
	  }
	  else {
	    w_[i].value = db_tmp;
	  }
	  ++xt;
	}
      }

      /*
      printf("epo:%d, ct:%d\n", epo, ct);
      for (i=0; i<w_.size(); i++)
      	printf("w[%d]=%f\n", i, w_[i].value);
      */
      
      t_ += 1.0;
      ct ++;
      ct_k ++;
    }
  }// for epo

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

  // rounding w
  index_t w_size = w_.size();
  round_thd_ = fx_param_double(NULL, "thd", 1.0e-5);
  w_nnz_ = 0;
  for (ct=0; ct<w_size; ct++) {
    if (fabs(w_[ct].value) >= round_thd_) { // TODO: thresholding
      w_nnz_++;
    }
  }

  printf("%d out of %d features are non zero. NZ rate:%f\n", w_nnz_, n_features_, (double)(w_nnz_)/(double)n_features_);
  
  // Calculate objective value; default: no calculation to save time
  int objvalue = fx_param_int(NULL, "objvalue", 0);
  if (objvalue > 0) {
    double hinge_loss = 0.0, loss_sum= 0.0, w_sum = 0.0;
    
    // primal objective value
    for (i=0; i<n_data_; i++) {
      hinge_loss = 1- y_[i] * SparseDot(w_, (dataset_->x)[i]);
      if (hinge_loss > 0) {
	loss_sum += hinge_loss;
      }
    }
    for (i=0; i<n_features_; i++) {
      w_sum += fabs(w_[i].value);
    }
    double obj_value = loss_sum + g_ * w_sum;
    
    printf("Primal objective value: %lf\n", obj_value);
  }

}

#endif

