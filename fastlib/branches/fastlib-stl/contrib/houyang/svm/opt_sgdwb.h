

#ifndef U_SVM_OPT_SGDWB_H
#define U_SVM_OPT_SGDWB_H

#include "fastlib/fastlib.h"

// tolerance of sacale_w
const double SGDWB_SCALE_W_TOLERANCE = 1.0e-9;
// threshold that determines whether an alpha is a SV or not
const double SGDWB_ALPHA_ZERO = 1.0e-7;


template<typename TKernel>
class SGDWB {
  FORBID_ACCIDENTAL_COPIES(SGDWB);

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
  index_t n_alpha_; /* number of lagrangian multipliers in the dual */
  index_t n_sv_; /* number of support vectors */
  
  index_t i_cache_, j_cache_; /* indices for the most recently cached kernel value */
  double cached_kernel_value_; /* cache */

  ArrayList<int> y_; /* list that stores "labels" */

  Vector w_; /* the slope of the decision hyperplane y=w^T x+b */
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

  double rho_;// for soft margin nonlinear SGDWB SVM

 public:
  SGDWB() {}
  ~SGDWB() {}

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
void SGDWB<TKernel>::LearnersInit_(int learner_typeid) {
  index_t i;
  learner_typeid_ = learner_typeid;
  rho_ = fx_param_double(NULL, "rho", 1.0); // specify the soft margin. default value 1.0: hard margin
  
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
* Steepest descent based SGDWB training for 2-classes
*
* @param: input 2-classes data matrix with labels (1,-1) in the last row
*/
template<typename TKernel>
void SGDWB<TKernel>::Train(int learner_typeid, const Dataset* dataset_in) {
  index_t i, j, epo, ct;
  
  /* general learner-independent initializations */
  dataset_ = dataset_in;
  datamatrix_.Alias(dataset_->matrix());
  n_data_ = datamatrix_.n_cols();
  n_features_ = datamatrix_.n_rows() - 1;
  n_features_bias_ = n_features_ + 1;

  if (n_epochs_ > 0) { // # of epochs provided, use it
    n_iter_ = n_data_;
  }
  else { // # of epochs not provided, use n_iter_ to count iterations
    n_epochs_ = 1; // not exactly one epoch, just use it for one loop
  }
  
  DEBUG_ASSERT(C_ != 0);
  lambda_ = 1.0/(C_*n_data_);
  

  /* learners initialization */
  LearnersInit_(learner_typeid);
  old_from_new_.Init(n_data_);

  index_t work_idx_old = 0;

  /* Begin SGDWB iterations */
  if (b_linear_) { // linear SVM, output: w, bias
    double yt, yt_hat, yy_hat;
    double sqrt_n = sqrt(n_data_);
    double eta0 = sqrt_n / max(1.0, LossFunctionGradient_(learner_typeid, -sqrt_n)); // initial step length
    double eta_grad = 0;
    t_ = 1.0 / (eta0 * lambda_);
    scale_w_ = 1.0;

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

	if (scale_w_ < SGDWB_SCALE_W_TOLERANCE) {
	  la::Scale(scale_w_, &w_);
	  scale_w_ = 1.0;
	  //printf("epo %d: scale_w tolerance reached.\n", epo);
	}
	
	Vector xt;
	datamatrix_.MakeColumnVector(work_idx_old, &xt);
	xt[n_features_] = 1.0; // for bias term: x <- [x,1], w <- [w, b]
	yt = y_[work_idx_old];
	yt_hat = la::Dot(w_, xt) * scale_w_;
	yy_hat = yt * yt_hat;

	if (yy_hat < 1.0) {
	  // update w by Stochastic Gradient Descent: w_{t+1} = (1-eta*lambda) * w_t + eta * [yt*xt]^+
	  eta_grad = eta_ * LossFunctionGradient_(learner_typeid, yy_hat) * yt; // also need *xt, but it's done in next line
	  la::AddExpert(eta_grad/scale_w_, xt, &w_); // Note: moving w's scaling calculation w_t*(1-1/t) to the testing session is faster
	}
	t_ += 1.0;
	ct ++;
      }
    } // for epo

    // Calculate objective value; default: no calculation to save time
    int objvalue = fx_param_int(NULL, "objvalue", 0);
    if (objvalue > 0) {
      double v = 0.0, hinge_loss = 0.0, loss_sum= 0.0;
      
      // primal objective value
      for (i=0; i< n_data_; i++) {
	Vector xt;
	datamatrix_.MakeColumnVector(i, &xt);
	xt[n_features_] = 1.0; // for bias term: x <- [x,1], w <- [w, b]
	hinge_loss = 1- y_[i] * scale_w_ * la::Dot(w_, xt);
	if (hinge_loss > 0) {
	  loss_sum += hinge_loss * C_;
	}
      }
      for (j=0; j<n_features_bias_; j++) {
	v += math::Sqr(w_[j]);
      }
      v = v * scale_w_ * scale_w_ / 2.0 + loss_sum;
      
      printf("Primal objective value: %lf\n", v);


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
      for (i=0; i<n_features_; i++) {
	if ( fabs(w_[i]) > round_thd ) {
	  w_ct ++;
	}
	else {
	  w_[i] = 0;
	}
      }
      printf("%d out of %d features are non zero\n", w_ct, n_features_);
    }

  }
  else { // nonlinear SVM, output: coefs(i.e. alpha*y), bias
    // TODO
  } // else
}


/* Get results for nonlinear SGDWB: coefficients(alpha*y), number and indecies of SVs
*
* @param: sample indices of the training (sub)set in the total training set
* @param: support vector coefficients: alpha*y
* @param: bool indicators  FOR THE TRAINING SET: is/isn't a support vector
*
*/
template<typename TKernel>
void SGDWB<TKernel>::GetSV(ArrayList<index_t> &dataset_index, ArrayList<double> &coef, ArrayList<bool> &sv_indicator) {
  n_sv_ = 0;
  if (learner_typeid_ == 0) {// SVM_C
    for (index_t i = 0; i < n_data_; i++) {
      if (fabs(coef_[i]) >= SGDWB_ALPHA_ZERO) { // support vectors found
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
  }
}

#endif
