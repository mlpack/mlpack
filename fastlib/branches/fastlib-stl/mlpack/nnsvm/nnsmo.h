#ifndef U_NNSVM_NNSMO_H
#define U_NNSVM_NNSMO_H

#include <fastlib/fastlib.h>

#include <armadillo>
#include <fastlib/base/arma_compat.h>

/* TODO: I don't actually want these to be public */
/* but sometimes we should provide freedoms for our advanced users */
const double NNSMO_ZERO = 1.0e-8;
const double NNSMO_TOLERANCE = 1.0e-3;

template<typename TKernel>
class NNSMO {

 public:
  typedef TKernel Kernel;

 private:
  arma::mat kernel_cache_sign_;
  Kernel kernel_;
  const Dataset* dataset_;
  index_t n_data_; // number of data samples
  arma::mat matrix_; // alias for the data matrix
  arma::vec alpha_; // the alphas, to be optimized
  arma::vec error_; // the error cache
  double thresh_; // negation of the intercept
  double c_;
  int budget_;
  double sum_alpha_; 

  index_t n_feature_; // number of data features
  double w_square_sum_; // square sum of the weight vector
  arma::vec VTA_; //
  double eps_; // the tolerace of progress on alpha values
  int max_iter_; // the maximum iteration, termination criteria

 public:
  NNSMO() {}
  ~NNSMO() {}

  /**
   * Initializes an NNSMO problem.
   *
   * You must initialize separately the kernel.
   */
  void Init(const Dataset& dataset_in, double c_in, int budget_in, double eps_in, int max_iter_in) {
    c_ = c_in;

    dataset_ = &dataset_in;
    
    n_data_ = matrix_.n_cols;
    budget_ = min(budget_in, n_data_);

    alpha_.zeros(n_data_);
    sum_alpha_ = 0;

    error_.zeros(n_data_);
    for(index_t i = 0; i < n_data_; i++)
      error_[i] -= GetLabelSign_(i);

    thresh_ = 0;

    n_feature_ = matrix_.n_rows - 1;
    VTA_.zeros(n_feature_);
    eps_ = eps_in;
    max_iter_ = max_iter_in;

  }

  void Train();

  const Kernel& kernel() const {
    return kernel_;
  }

  Kernel& kernel() {
    return kernel_;
  }

  double threshold() const {
    return thresh_;
  }

  void GetNNSVM(arma::mat& support_vectors, arma::vec& alpha, arma::vec& w) const;

private:
  index_t TrainIteration_(bool examine_all);

  bool TryChange_(index_t j);

  double CalculateDF_(index_t i, index_t j, double error_j);

  bool TakeStep_(index_t i, index_t j, double error_j);

  double FixAlpha_(double alpha) const {
    if (alpha < NNSMO_ZERO) {
      alpha = 0;
    } else if (alpha > c_ - NNSMO_ZERO) {
      alpha = c_;
    }
    return alpha;
  }

  bool IsBound_(double alpha) const {
    return alpha <= 0 || alpha >= c_;
  }

 // labels: the last row of the data matrix, 0 or 1
  int GetLabelSign_(index_t i) const {
    return (matrix_(matrix_.n_rows - 1, i) != 0) ? 1 : -1;
  }

  void GetVector_(index_t i, arma::vec& v) const {
    v = arma::vec((double*) matrix_.colptr(i), matrix_.n_rows - 1, false, true); // manual ugly constructor
  }

  double Error_(index_t i) const {
    return error_[i];
  }
  
  double Evaluate_(index_t i) const;

  double EvalKernel_(index_t i, index_t j) const {
    return kernel_cache_sign_(i, j) * (GetLabelSign_(i) * GetLabelSign_(j));
  }
  
  void CalcKernels_() {
    kernel_cache_sign_.set_size(n_data_, n_data_);
    fprintf(stderr, "Kernel Start\n");
    for (index_t i = 0; i < n_data_; i++) {
      for (index_t j = 0; j < n_data_; j++) {
        arma::vec v_i;
        GetVector_(i, v_i);
        arma::vec v_j;
        GetVector_(j, v_j);
        double k = kernel_.Eval(v_i, v_j);        
        kernel_cache_sign_(j, i) = k * GetLabelSign_(i) * GetLabelSign_(j);
      }
    }
    fprintf(stderr, "Kernel Stop\n");
  }
};

// return the support vector, the support alpha vector and the weight vector of the trained NNSVM
template<typename TKernel>
void NNSMO<TKernel>::GetNNSVM(arma::mat& support_vectors, arma::vec& support_alpha, arma::vec& w) const {
  index_t n_support = 0;
  index_t i_support = 0;

  for (index_t i = 0; i < n_data_; i++) {
    if (alpha_[i] != 0) {
      n_support++;
    }
  }

  support_vectors.set_size(matrix_.n_rows - 1, n_support);
  support_alpha.set_size(n_support);

  for (index_t i = 0; i < n_data_; i++) {
 	
    if (alpha_[i] != 0) {
      arma::vec source;
      arma::vec dest;
      GetVector_(i, source);
      dest = support_vectors.unsafe_col(i_support);
      dest = source;
      support_alpha[i_support] = alpha_[i] * GetLabelSign_(i);
      i_support++;
    }
  }

  w.set_size(n_feature_);
  for(index_t s = 0; s < n_feature_; s++)
    w[s] = math::ClampNonNegative(VTA_[s]);
}

//NNSMO training for 2-classes
template<typename TKernel>
void NNSMO<TKernel>::Train() {
  bool examine_all = true;
  index_t num_changed = 0;
  int n_iter = 0;
  int counter = 1000;
  if (counter > n_data_)
    counter = n_data_;

  // calculate kernel_cache_sign_: [k_ij* y_i* y_j]
  CalcKernels_();
  while ((num_changed > 0 || examine_all)) {
    VERBOSE_GOT_HERE(0);
    //NNSMO training iterations
	num_changed = TrainIteration_(examine_all);

    if (examine_all) {
      examine_all = false;
    } else if (num_changed == 0) {
      examine_all = true;
    }
    
	//if exceed the maximum number of iterations, finished
    if (++n_iter == max_iter_) {
      fprintf(stderr, "Max iterations Reached! \n");
      break;
    }
    
	//for every max(n_data_, 1000) iterations, show progress
    if (n_iter % counter == 0 ) 
      fprintf(stderr, ".");
  }

  //compute the final objective value
  double obj = sum_alpha_ - w_square_sum_/2;
  fprintf(stderr, "iter=%d, %d, %f, %f, %f, obj=%f \n", n_iter, num_changed, thresh_, sum_alpha_, w_square_sum_, obj);
}

//NNSMO training iteration
template<typename TKernel>
index_t NNSMO<TKernel>::TrainIteration_(bool examine_all) {
  index_t num_changed = 0;

  for (index_t i = 0; i < n_data_; i++) {
    if ((examine_all || !IsBound_(alpha_[i])) && TryChange_(i)) {
      num_changed++;
    }
  }
  return num_changed;
}

// try to find the working set 
//	outer loop: alpha_j, KKT violation
//	inner loop: alpha_i, maximum objective value increase with respective to alpha_i, j
template<typename TKernel>
bool NNSMO<TKernel>::TryChange_(index_t j) {
  double error_j = Error_(j);
  double rj = error_j * GetLabelSign_(j);
  
  VERBOSE_GOT_HERE(0);

  if (!((rj < -NNSMO_TOLERANCE && alpha_[j] < c_)
      || (rj > NNSMO_TOLERANCE && alpha_[j] > 0))) {
    return false; // nothing to change
  }

  // first try the one we suspect to have the largest yield
  index_t i = -1;
  double df_max = 0;
  for (index_t k = 0; k < n_data_; k++) {
    double df_k = CalculateDF_(k, j, error_j);
    if (df_k > df_max) {
      df_max = df_k;
      i = k;
    }
  }
  if (i != -1 && TakeStep_(i, j, error_j)) {
    return true;
  }

  VERBOSE_GOT_HERE(0);

  return false;
}

//compute the increase of objective value with respect to updating of alpha_i, alpha_j
template<typename TKernel>
double NNSMO<TKernel>::CalculateDF_(index_t i, index_t j, double error_j)  {
  //1. check i,j
  if (i == j) {
    VERBOSE_GOT_HERE(0);
    return -1;
  }
  int yi = GetLabelSign_(i);
  int yj = GetLabelSign_(j);
  double alpha_i = alpha_[i];
  double alpha_j = alpha_[j];
  double l;
  double u;
  int s = (yi == yj) ? 1 : -1;
  double error_i = Error_(i);
  double r;

  //2. compute L, H of alpha_j
  if (s < 0) {
    DEBUG_ASSERT(s == -1);
    r = alpha_j - alpha_i; // target values are not equal
  } else {
    r = alpha_j + alpha_i - c_; // target values are equal
  }
  l = math::ClampNonNegative(r);
  u = c_ + math::ClampNonPositive(r);
  
  if (l >= u - NNSMO_ZERO) {
    // TODO: might put in some tolerance
    VERBOSE_MSG(0, "l=%f, u=%f, r=%f, c_=%f, s=%f", l, u, r, c_, s);
    VERBOSE_GOT_HERE(0);
    return -1;
  }

  //3. compute eta using cached kernel values 
  double kii = EvalKernel_(i, i);
  double kij = EvalKernel_(i, j);
  double kjj = EvalKernel_(j, j);
  double eta = +2*kij - kii - kjj;
  VERBOSE_MSG(0, "kij=%f, kii=%f, kjj=%f", kij, kii, kjj);
  
  // calculate alpha_j^{new}
  if (likely(eta < 0)) {
    VERBOSE_MSG(0, "Common case");
    alpha_j = alpha_[j] - yj * (error_i - error_j) / eta;
    alpha_j = FixAlpha_(math::ClampRange(alpha_j, l, u));
  } else {
    VERBOSE_MSG(0, "Uncommon case");
	return -1;
  }
  alpha_j = FixAlpha_(alpha_j);
  double delta_alpha_j = alpha_j - alpha_[j];

  // check if there is progress
  if (fabs(delta_alpha_j) < eps_*(alpha_j + alpha_[j] + eps_)) {
    VERBOSE_GOT_HERE(0);
    return -1;
  }

  //4. compute increase of objective value 		
  arma::vec w(n_feature_);
  for (index_t s = 0; s < n_feature_; s++) {
    double VTdA_s = (matrix_(s, j) - matrix_(s, i)) * yj * delta_alpha_j;
    w[s] = math::ClampNonNegative(VTA_[s] + VTdA_s);
  }
  double delta_f = w_square_sum_ / 2 - dot(w, w) / 2; 
  if(yi != yj)
    delta_f += 2* delta_alpha_j;

  VERBOSE_GOT_HERE(0);
  return delta_f;
}

// update alpha_i, alpha_j, as well as the VTA_, negation of intercept: thresh_ and the error cache: error_
template<typename TKernel>
bool NNSMO<TKernel>::TakeStep_(index_t i, index_t j, double error_j) {
  //1. check i,j
  if (i == j) {
    VERBOSE_GOT_HERE(0);
    return false;
  }

  int yi = GetLabelSign_(i);
  int yj = GetLabelSign_(j);
  double alpha_i = alpha_[i];
  double alpha_j = alpha_[j];
  double l;
  double u;
  int s = (yi == yj) ? 1 : -1;
  double error_i = Error_(i);
  double r;
  
  //2. compute L, H of alpha_j
  if (s < 0) {
    DEBUG_ASSERT(s == -1);
    r = alpha_j - alpha_i; // target values are not equal
  } else {
    r = alpha_j + alpha_i - c_; // target values are equal
  }
  l = math::ClampNonNegative(r);
  u = c_ + math::ClampNonPositive(r);

  if (l >= u - NNSMO_ZERO) {
    // TODO: might put in some tolerance
    VERBOSE_MSG(0, "l=%f, u=%f, r=%f, c_=%f, s=%f", l, u, r, c_, s);
    VERBOSE_GOT_HERE(0);
    return false;
  }
  
  //3. compute eta using cached kernel values 
  double kii = EvalKernel_(i, i);
  double kij = EvalKernel_(i, j);
  double kjj = EvalKernel_(j, j);
  double eta = +2*kij - kii - kjj;
  VERBOSE_MSG(0, "kij=%f, kii=%f, kjj=%f", kij, kii, kjj);
  
  // calculate alpha_j^{new}
  if (likely(eta < 0)) {
    VERBOSE_MSG(0,"Common case");
    alpha_j = alpha_[j] - yj * (error_i - error_j) / eta;
    alpha_j = FixAlpha_(math::ClampRange(alpha_j, l, u));
  } else {
    VERBOSE_MSG(0, "Uncommon case");
	return false;
  }
  alpha_j = FixAlpha_(alpha_j);
  double delta_alpha_j = alpha_j - alpha_[j];

  // check if there is progress
  if (fabs(delta_alpha_j) < eps_*(alpha_j + alpha_[j] + eps_)) {
    VERBOSE_GOT_HERE(0);
    return false;
  }

  // calculate alpha_i^new
  alpha_i = alpha_i - (s)*(delta_alpha_j);
  if (alpha_i < NNSMO_ZERO) {
    alpha_j += s * alpha_i;
    alpha_i = 0;
  } else if (alpha_i > c_ - NNSMO_ZERO) {
    double t = alpha_i - c_;
    alpha_j += s * t;
    alpha_i = c_;
  }  
  double delta_alpha_i = alpha_i - alpha_[i];
  delta_alpha_j = alpha_j - alpha_[j];
  
  //4. update VTA_, w_square_sum_
  arma::vec w(n_feature_);
  for (index_t s = 0; s < n_feature_; s++) {
    double VTdA_s = matrix_(s, i) * yi * delta_alpha_i + matrix_(s, j) * yj * delta_alpha_j;
    VTA_[s] += VTdA_s;
    w[s] = math::ClampNonNegative(VTA_[s]);
  }
  w_square_sum_ = dot(w, w);

  // update alpha_, sum_alpha_
  alpha_[i] = alpha_i;
  alpha_[j] = alpha_j;
  sum_alpha_ += delta_alpha_i + delta_alpha_j;

  // update error cache and threshold
  double thresh_sum = 0;
  index_t nb_count = 0;
  for (index_t k = 0; k < n_data_ ; k++) {
    arma::vec x_k;
    GetVector_(k, x_k);
    error_[k] = dot(w, x_k) - GetLabelSign_(k);
    if(!IsBound_(alpha_[k])) {
      thresh_sum += error_[k];
      nb_count++;
    }	  
  }
  if(nb_count > 0)
    thresh_ = thresh_sum/nb_count;

  // update error cache using the new threshold
  for (index_t k = 0; k < n_data_ ; k++ )
    error_[k] -= thresh_;

  VERBOSE_GOT_HERE(0);
  return true;
}

#endif 
