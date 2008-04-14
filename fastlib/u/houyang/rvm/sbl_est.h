/**
 * @author Hua Ouyang
 *
 * @file sbl_est.h
 *
 * This head file contains functions for performing Sparse Bayesian Learning for Relevance Vector Machine
 *
 * The algorithms in the following papers are implemented (the notations follow Bishop_PRML):
 *
 * @ARTICLE{Tipping_RVM_JMLR,
 * author = "M. E. Tipping",
 * title = "{Sparse Bayesian Learning and the Relevance Vector machine}"
 * journal = "{Journal of Machine Learning Research}",
 * year = 2001
 * }
 *
 * @INPROCEEDINGS{Tipping_RVM_NIPS,
 * author = "M. E. Tipping",
 * title = "{The Relevance Vector machine}",
 * booktitle = NIPS,
 * number = 12,
 * year = 2000
 * }
 *
 * @BOOK{ Bishop_PRML,
 * author = "C.M. Bishop",
 * title = "{Pattern Recognition and Machine Learning}",
 * publisher = "Springer",
 * year = 2006
 * }
 *
 * @see rvm.h
 */

#ifndef U_RVM_SBL_EST_H
#define U_RVM_SBL_EST_H

#include "fastlib/fastlib.h"

#include "math/statistics.h"

/* TODO: I don't actually want these to be public */
// Prune basis function when its alpha is greater than this
const double ALPHA_MAX = 1.0e12;
// Iteration number during training where we switch to 'analytic pruning'
const index_t PRUNE_POINT = 50;
// Terminate estimation when no log-alpha value changes by more than this
const double MIN_DELTA_LOGALPHA = 1.0e-3;

template<typename TKernel>
class SBL_EST {
  FORBID_ACCIDENTAL_COPIES(SBL_EST);

 public:
  typedef TKernel Kernel;

 private:
  Kernel kernel_; /* kernel */
  const Dataset *dataset_; /* input data */
  index_t n_data_; /* number of data samples */
  Matrix matrix_; /* alias for the data matrix */
  Matrix PHI_;  /* kernel matrix with bias term in the first column */
  //index_t n_rv_; /* number of relevance vectors */
  //Vector error_;

 public:
  SBL_EST() {}
  ~SBL_EST() {}

  void Train(int learner_typeid, const Dataset* dataset_in, Vector &alpha_v, double &beta, index_t max_iter, ArrayList<index_t> &rv_index, ArrayList<double> &weights);

  Kernel& kernel() {
    return kernel_;
  }

  //index_t num_rv() const {
  //  return n_rv_;
  //}

 private:
  /*
    double Error_(index_t i) const {
    double val;
    if (!IsBound_(alpha_[i])) {
      val = error_[i];
      VERBOSE_MSG(0, "error values %f and %f", error_[i], Evaluate_(i) - GetLabelSign_(i));
    } else {
      val = CalculateError_(i);
    }
    return val;
    }

  double CalculateError_(index_t i) const {
    return Evaluate_(i) - GetLabelSign_(i);
    }
  
    double Evaluate_(index_t i) const; 

  double EvalKernel_(index_t i, index_t j) const {
    return kernel_cache_sign_.get(i, j) * (GetLabelSign_(i) * GetLabelSign_(j));
    }
  */
  
  /**
   * Calculate kernel values, dim(PHI_)==n_data_ x (1+n_data_)
   */

  void GetVector_(index_t i, Vector *v) const {
    matrix_.MakeColumnSubvector(i, 0, matrix_.n_rows()-1, v);
  }
  
  void CalcKernels_() {
    PHI_.Init(n_data_, n_data_+1); // the first column is for b (bias)
    fprintf(stderr, "Kernel Start\n");
    for (index_t i = 0; i < n_data_; i++) {
      for (index_t j = 0; j < n_data_; j++) {
        Vector v_i;
        GetVector_(i, &v_i);
        Vector v_j;
        GetVector_(j, &v_j);

        PHI_.set(j, i+1, kernel_.Eval(v_i, v_j));
      }
    }
    fprintf(stderr, "Kernel Stop\n");
  }
};

/**
* RVM training, for both regression and classification
*
* @param: input 2-classes data matrix with labels (1,-1) in the last row
*/
template<typename TKernel>
void SBL_EST<TKernel>::Train(int learner_typeid, const Dataset* dataset_in, Vector &alpha_v, double &beta, int max_iter, ArrayList<index_t> &rv_index, ArrayList<double> &weights) {
  dataset_ = dataset_in;
  matrix_.Alias(dataset_->matrix());
  n_data_ = matrix_.n_cols();
  index_t i,j;

  /* weights corresponding to non zero alpha_v, dim(w_nz) == ct_non_zero x 1*/
  Matrix w_nz;

  // Obtain kernel matrix PHI=[b K], b is the bias term
  CalcKernels_();
  // first all-one column is for bias
  for (i=0; i<max_iter; i++)
    PHI_.set(i, 1, 1);

  Matrix train_values; // the vector that stores the values for data points, dim: n_data_ x 1
  train_values.Init(n_data_, 1);
  for(i=0; i<n_data_; i++)
    train_values.set(i, 1, dataset_in->get(dataset_in->n_features()-1, i) );
  Matrix PHI_t; // dim(PHI_t) == n_data_+1 x 1
  la::MulTransAInit(PHI_, train_values, &PHI_t); // PHI_t = PHI_' * train_values
  
  bool LastIter = false;

  index_t ct_non_zero = 0;
  /* Training Iterations */
  for (index_t c=0; c<max_iter; c++) {
    /* 1. Prune large values of alpha, get the shrinked set */
    for(i=0; i<n_data_+1; i++) { // dim(alpha_v) == n_data_+1 x 1
      if (alpha_v[i] < ALPHA_MAX) {
	ct_non_zero ++;
      }
    }
    Vector alpha_nz; // dim(alpha_nz) == ct_non_zero x 1
    Vector alpha_nz_idx; // dim(alpha_nz_idx) == n_data_
    Matrix PHI_nz; // dim(PHI_nz) == n_data_ x ct_non_zero
    Matrix PHI_t_nz; // dim(PHI_t_nz) == ct_non_zero x 1
    alpha_nz.Init(ct_non_zero);
    alpha_nz_idx.Init(n_data_);
    alpha_nz_idx.SetAll(0);
    PHI_nz.Init(n_data_, ct_non_zero);
    PHI_t_nz.Init(ct_non_zero, 1);
    ct_non_zero = 0;
    Vector source, dest;
    for(i=0; i<n_data_+1; i++) {
      if (alpha_v[i] < ALPHA_MAX) {
	alpha_nz[ct_non_zero] = alpha_v[i];
	alpha_nz_idx[i] = 1;
	PHI_.MakeColumnVector(i, &source);
	PHI_nz.MakeColumnVector(ct_non_zero, &dest);
	dest.CopyValues(source);

	ct_non_zero ++;
      }
    }

    /* 2. Compute marginal likelihood (the objective function to be maximized) */
    Matrix U; // dim(U) == ct_non_zero x ct_non_zero    
    double ED = 0.0;
    double betaED, logBeta;

    if (learner_typeid == 1) { // RVM Regression
      Matrix Hessian; // dim(Hessian) == ct_non_zero x ct_non_zero
      Hessian.InitDiagonal(alpha_nz); // Hessian = diag(alpha_nz)
      
      Matrix temp_mat; // dim(temp_mat) == ct_non_zero x ct_non_zero
      la::MulTransAInit(PHI_nz, PHI_nz, &temp_mat); // PHI_nz'*PHI_nz
      la::AddExpert(beta, temp_mat, &Hessian); // Hessian = (PHI_nz'*PHI_nz)*beta + diag(alpha_nz);
      temp_mat.Destruct();
      
      la::CholeskyInit(Hessian, &U); // Hessian = U'*U
      la::Inverse(&U); // U = U^-1, OUTPUT
      Hessian.Destruct();
      
      la::MulTransBInit(U, U, &w_nz); // U^-1 * (U^-1)'
      la::MulOverwrite(w_nz, PHI_t_nz, &w_nz); // U^-1 * (U^-1)' * PHI_t_nz
      la::Scale(beta, &w_nz); // w_nz = U^-1 * (U^-1)' * PHI_t_nz * beta, OUTPUT
      U.Destruct();
      PHI_t_nz.Destruct();
      
      Matrix temp_PHInz_mult_wnz;
      la::MulInit(PHI_nz, w_nz, &temp_PHInz_mult_wnz);
      PHI_nz.Destruct();
      
      for (i=0; i<n_data_; i++)
	ED += math::Sqr( train_values.get(i,1) - temp_PHInz_mult_wnz.get(i,1) );
      temp_PHInz_mult_wnz.Destruct();
      betaED = beta * ED; // OUTPUT, betaED = beta * sum((t-PHI_nz*w(nonZero)).^2);;
      logBeta = n_data_ * log(beta);
    }
    else if (learner_typeid == 0) { // RVM Classification
      // TODO
      logBeta = 0;
    }

    double logdetH;
    double dbtemp;
    Vector diagSig; // dim(diagSig) == ct_non_zero x 1    
    logdetH = 0.0;
    diagSig.Init(ct_non_zero);
    for (i=0; i<ct_non_zero; i++) {
      logdetH += log(U.get(i,i));
      dbtemp = 0.0;
      for (j=0; j<ct_non_zero; j++) {
	dbtemp += math::Sqr(U.get(i,j));
      }
      diagSig[i] = dbtemp; // diagSig = sum(Ui.^2,2);
    }
    logdetH = (-2) * logdetH; // logdetH = -2*sum(log(diag(Ui)));

    Vector gamma; // dim(gamma) == ct_non_zero x 1
    gamma.Init(ct_non_zero);
    for (i=0; i<ct_non_zero; i++)
      gamma[i] = - alpha_nz[i] * diagSig[i];
    Vector ones;
    ones.Init(ct_non_zero);
    ones.SetAll(1);
    la::AddTo(ones, &gamma); // gamma = 1 - alpha_nz.*diagSig;
    ones.Destruct();

    // marginal	= -0.5* [(w(nonZero).^2)'*alpha_nz - sum(log(alpha_nz)) + logdetH - logBeta + betaED ];
    Vector w_nz_sq;
    w_nz_sq.Init(ct_non_zero);
    for (i=0; i<ct_non_zero; i++)
      w_nz_sq[i] = math::Sqr(w_nz.get(i,1));
    double marginal = la::Dot(w_nz_sq, alpha_nz);
    w_nz_sq.Destruct();
    for (i=0; i<ct_non_zero; i++)
      marginal -= log(alpha_nz[i]);
    marginal = -0.5 * ( marginal + logdetH - logBeta + betaED);

    /* 3. Iterative Re-estimation for parameters (alpha and beta) and Termination check */
    if (LastIter == false) {
      Vector logAlpha_nz;
      logAlpha_nz.Init(ct_non_zero);
      for (i=0; i<ct_non_zero; i++)
	logAlpha_nz[i] = log(alpha_nz[i]);
      
      // update alpha
      if (c < PRUNE_POINT) {
	// MacKay-style update given in original NIPS paper
	for (i=0; i<ct_non_zero; i++) {
	  alpha_nz[i] = gamma[i] / math::Sqr(w_nz.get(i,1));
	}
      }
      else {
	// Hybrid update based on NIPS theory paper and AISTATS submission
	for (i=0; i<ct_non_zero; i++) {
	  alpha_nz[i] = gamma[i] / ( math::Sqr(w_nz.get(i,1))/gamma[i] - diagSig[i] );
	  if (alpha_nz[i] < 0)
	    alpha_nz[i] = INFINITY;
	}
      }
      for(i=0; i<n_data_+1; i++) {
	if (alpha_nz_idx[i] != 0)
	  //dbtemp = alpha_nz[i];
	  //alpha_v[i] = dbtemp;
	  alpha_v[i] = alpha_nz[i];
      }
      alpha_nz_idx.Destruct();     

      index_t ct = 0;
      for (i=0; i<ct_non_zero; i++) {
	if (alpha_nz[i] != 0)
	  ct ++;
      }
      Vector DAlpha;
      DAlpha.Init(ct);
      ct = 0;
      for (i=0; i<ct_non_zero; i++) {
	if (alpha_nz[i] != 0) {
	  DAlpha[ct] = fabs( logAlpha_nz[i] - log(alpha_nz[i]) );
	  ct ++;
	}
      }
      alpha_nz.Destruct();
      double maxDAlpha = - INFINITY;
      for (i=0; i<ct; i++) {
	if (DAlpha[i] > maxDAlpha)
	  maxDAlpha = DAlpha[i];
      }
      DAlpha.Destruct();

      // Terminate if the largest alpha change is judged too small
      if (maxDAlpha < MIN_DELTA_LOGALPHA)
	LastIter = true;
      // update beta for regression
      if (learner_typeid == 1) {
	double sum_gamma = 0.0;
	for (i=1; i<ct_non_zero; i++)
	  sum_gamma += gamma[i];
	beta = (n_data_ - sum_gamma) / ED;
      }
    }
    else {
      // The last iteration due to termination, leave outer loop
      break;
    }
  } /* Training Iterations end */

  /* Get RVM results:weights, indecies of RVs */
  double alpha_v_temp;
  for (i=0; i<n_data_; i++) {
    alpha_v_temp = alpha_v[i];
    if (alpha_v_temp != 0)
      *rv_index.AddBack() = i;      
  }
  for (i=0; i<ct_non_zero; i++)
    *weights.AddBack() = w_nz.get(i,1);
}


#endif
