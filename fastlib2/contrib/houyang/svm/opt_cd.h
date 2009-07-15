/**
 * @author Hua Ouyang
 *
 * @file opt_cd.h
 *
 * This head file contains functions for performing Coordinate Descent based optimization for linear L1- and L2- SVMs
 *
 * The algorithms in the following papers are implemented:
 *
 * 1. Primal Coordinate Descent for L2-SVM
 * @ARTICLE{Chang_PCD,
 * author = "Kai-Wei Chang, Cho-Jui Hsieh, Chih-Jen Lin",
 * title = "{Coordinate Descent Method for Large-scale L2-loss Linear Support Vector Machines}",
 * booktitle = "{Journal of Machines Learning Research}",
 * year = 2008,
 * }
 *
 * 2. Dual Coordinate Descent for L1- and L2-SVM
 * @ARTICLE{Hsieh_DCD,
 * author = "Cho-Jui Hsieh, Kai-Wei Chang, Chih-Jen Lin",
 * title = "{A Dual Coordinate Descent Method for Large Scale Linear SVM}",
 * booktitle = ICML,
 * year = 2008,
 * }
 *
 * @see svm.h
 */

#ifndef U_SVM_OPT_CD_H
#define U_SVM_OPT_CD_H

#include "fastlib/fastlib.h"

// threshold that determines whether an alpha is a SV or not
const double CD_ALPHA_ZERO = 1.0e-7;


template<typename TKernel>
class CD {
  FORBID_ACCIDENTAL_COPIES(CD);

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
  index_t n_alpha_; /* number of lagrangian multipliers in the dual */
  
  index_t i_cache_, j_cache_; /* indices for the most recently cached kernel value */
  double cached_kernel_value_; /* cache */

  ArrayList<int> y_; /* list that stores "labels" */

  Vector w_; /* the slope of the decision hyperplane y=w^T x+b */
  double bias_;

  // parameters
  double C_; // for SVM_C
  double epsilon_; // for SVM_R

  double lambda_; // regularization parameter. lambda = 1/(C*n_data)
  index_t n_iter_; // number of iterations
  index_t n_epochs_; // number of epochs
  double accuracy_; // accuracy for stopping creterion
  double t_;

  ArrayList<index_t> old_from_new_; // for generating a random sequence of training data
  ArrayList<index_t> new_from_old_; // for generating a random sequence of training data

 public:
  CD() {}
  ~CD() {}

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

  double Bias() const {
    return bias_;
  }

  Vector* GetW() {
    return &w_;
  }


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
void CD<TKernel>::LearnersInit_(int learner_typeid) {
  index_t i;
  learner_typeid_ = learner_typeid;
  
  if (learner_typeid_ == 0) { // SVM_C
    w_.Init(n_features_);
    w_.SetZero();
    
    coef_.Init(0); // not used, plain init

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
* Coordinate descent based SVM training for 2-classes
*
* @param: input 2-classes data matrix with labels (1,-1) in the last row
*/
template<typename TKernel>
void CD<TKernel>::Train(int learner_typeid, const Dataset* dataset_in) {
  index_t i, j, epo, ct;
  
  /* general learner-independent initializations */
  dataset_ = dataset_in;
  datamatrix_.Alias(dataset_->matrix());
  n_data_ = datamatrix_.n_cols();
  n_features_ = datamatrix_.n_rows() - 1;

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

  double sqrt_n = sqrt(n_data_);
  double eta0 = sqrt_n / max(1.0, LossFunctionGradient_(learner_typeid, -sqrt_n)); // initial step length
  double eta_grad = INFINITY;
  t_ = 1.0 / (eta0 * lambda_);

  /* Begin CD iterations */
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
      
      Vector xt;
      datamatrix_.MakeColumnSubvector(work_idx_old, 0, n_features_, &xt);
      double yt = y_[work_idx_old];
      double yt_hat = la::Dot(w_, xt) + bias_;
      double yy_hat = yt * yt_hat;
      if (yy_hat < 1.0) {
	// update w by Stochastic Gradient Descent: w_{t+1} = (1-eta*lambda) * w_t + eta * [yt*xt]^+
	eta_grad = LossFunctionGradient_(learner_typeid, yy_hat) * yt; // also need *xt, but it's done in next line
	la::AddExpert(eta_grad, xt, &w_); // Note: moving w's scaling calculation to the testing session is faster
	// update bias
	bias_ += eta_grad * 0.01;
      }
      t_ += 1.0;
      ct ++;
    }
  }
  
}


#endif
