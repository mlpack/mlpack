#ifndef NAIVE_LPR_H
#define NAIVE_LPR_H

#include "fastlib/fastlib_int.h"

template<typename TKernel>
class NaiveLpr {

  FORBID_ACCIDENTAL_COPIES(NaiveLpr);

 private:

  struct datanode *module_;

  /** query dataset */
  Matrix qset_;

  /** reference dataset */
  Matrix rset_;

  /** reference target */
  Vector rset_targets_;

  /** kernel */
  TKernel kernel_;

  /** numerator vector X^T W(q) Y for each query point */
  ArrayList<Vector> numerator_;
  
  /** denominator matrix X^T W(q) X for each query point */
  ArrayList<Matrix> denominator_;

  /** computed regression values */
  Vector regression_values_;

  /** local polynomial approximation order */
  int lpr_order_;

  /** total number of coefficients for the local polynomial */
  int total_num_coeffs_;

  void PseudoInverse(const Matrix &A, Matrix *A_inv) {
    Vector ro_s;
    Matrix ro_U, ro_VT;

    // compute the SVD of A
    la::SVDInit(A, &ro_s, &ro_U, &ro_VT);
    
    // take the transpose of V^T and U
    Matrix ro_VT_trans;
    Matrix ro_U_trans;
    la::TransposeInit(ro_VT, &ro_VT_trans);
    la::TransposeInit(ro_U, &ro_U_trans);
    Matrix ro_s_inv;
    ro_s_inv.Init(ro_VT_trans.n_cols(), ro_U_trans.n_rows());
    ro_s_inv.SetZero();

    // initialize the diagonal by the inverse of ro_s
    for(index_t i = 0; i < ro_s.length(); i++) {
      ro_s_inv.set(i, i, 1.0 / ro_s[i]);
    }
    Matrix intermediate;
    la::MulInit(ro_s_inv, ro_U_trans, &intermediate);
    la::MulInit(ro_VT_trans, intermediate, A_inv);
  }

 public:
  
  NaiveLpr() {}

  ~NaiveLpr() {}

  void Compute() {

    // temporary variables for multiindex looping
    ArrayList<int> heads;
    Vector weighted_values;

    printf("\nStarting naive LPR...\n");
    fx_timer_start(NULL, "naive_lpr_compute");

    // initialization of temporary variables for computation...
    heads.Init(qset_.n_rows() + 1);
    weighted_values.Init(total_num_coeffs_);    

    // compute unnormalized sum for the numerator vector and the denominator
    // matrix
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      
      const double *q_col = qset_.GetColumnPtr(q);
      for(index_t r = 0; r < rset_.n_cols(); r++) {
	const double *r_col = rset_.GetColumnPtr(r);
	const double r_target = rset_targets_[r];
	double dsqd = la::DistanceSqEuclidean(qset_.n_rows(), q_col, r_col);
	double kernel_value = kernel_.EvalUnnormOnSq(dsqd);

	// multiindex looping
	for(index_t i = 0; i < qset_.n_rows(); i++) {
	  heads[i] = 0;
	}
	heads[qset_.n_rows()] = INT_MAX;

	weighted_values[0] = 1.0;
	for(index_t k = 1, t = 1, tail = 1; k <= lpr_order_; k++, tail = t) {
	  for(index_t i = 0; i < qset_.n_rows(); i++) {
	    int head = (int) heads[i];
	    heads[i] = t;
	    for(index_t j = head; j < tail; j++, t++) {

	      // compute numerator vector position t based on position j
	      weighted_values[t] = weighted_values[j] * r_col[i];
	    }
	  }
	}

	// tally up the sum here
	for(index_t i = 0; i < total_num_coeffs_; i++) {
	  numerator_[q][i] = numerator_[q][i] + r_target *
	    weighted_values[i] * kernel_value;

	  for(index_t j = 0; j < total_num_coeffs_; j++) {
	    denominator_[q].set(i, j, denominator_[q].get(i, j) + 
				weighted_values[i] * weighted_values[j] *
				kernel_value);
	  }
	}
      } // end of looping over each reference point
    } // end of looping over each query point

    // now iterate over all query points and compute regression estimate
    for(index_t q = 0; q < qset_.n_cols(); q++) {

      const double *q_col = qset_.GetColumnPtr(q);
      Matrix denominator_inv_q;
      Vector beta_q;

      // now invert the denominator matrix for each query point and multiply
      // by the numerator vector
      PseudoInverse(denominator_[q], &denominator_inv_q);      
      la::MulInit(denominator_inv_q, numerator_[q], &beta_q);

      // compute the vector [1, x_1, \cdots, x_D, second order, ...]
      weighted_values[0] = 1.0;
      for(index_t k = 1, t = 1, tail = 1; k <= lpr_order_; k++, tail = t) {
	for(index_t i = 0; i < qset_.n_rows(); i++) {
	  int head = (int) heads[i];
	  heads[i] = t;
	  for(index_t j = head; j < tail; j++, t++) {

	    // compute numerator vector position t based on position j
	    weighted_values[t] = weighted_values[j] * q_col[i];
	  }
	}
      }

      // compute the dot product between the multiindex vector for the query
      // point by the beta_q
      regression_values_[q] = la::Dot(weighted_values, beta_q);
    }
    
    fx_timer_stop(NULL, "naive_lpr_compute");
    printf("\nNaive LPR completed...\n");
  }

  void Init(Matrix &queries, Matrix &references, Matrix &reference_targets,
	    struct datanode *module_in) {

    // Set the module to the incoming.
    module_ = module_in;

    // Copy the datasets.
    qset_.Copy(queries);
    rset_.Copy(references);

    // read the reference weights
    
    rset_targets_.Copy(reference_targets.GetColumnPtr(0),
		       reference_targets.n_cols());

    // get bandwidth
    kernel_.Init(fx_param_double_req(module_, "bandwidth"));

    // compute total number of coefficients
    lpr_order_ = fx_param_int(module_in, "lpr_order", 1);
    total_num_coeffs_ = (int) math::BinomialCoefficient(lpr_order_ + 
							qset_.n_rows(),
							qset_.n_rows());
    
    // allocate temporary storages for storing the numerator vectors and
    // the denominator matrices
    numerator_.Init(qset_.n_cols());
    denominator_.Init(qset_.n_cols());

    for(index_t i = 0; i < qset_.n_cols(); i++) {
      numerator_[i].Init(total_num_coeffs_);
      numerator_[i].SetZero();
      denominator_[i].Init(total_num_coeffs_, total_num_coeffs_);
      denominator_[i].SetZero();
    }
    
    // allocate density storage
    regression_values_.Init(qset_.n_cols());
    regression_values_.SetZero();
  }

  void PrintDebug() {

    FILE *stream = stdout;
    const char *fname = NULL;

    if((fname = fx_param_str(module_, "naive_lpr_output", NULL)) != NULL) {
      stream = fopen(fname, "w+");
    }
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      fprintf(stream, "%g\n", regression_values_[q]);
    }
    
    if(stream != stdout) {
      fclose(stream);
    }    
  }

  void ComputeMaximumRelativeError(const Vector &regression_estimate) {
    
    double max_rel_err = 0;
    for(index_t q = 0; q < regression_values_.length(); q++) {
      double rel_err = fabs(regression_estimate[q] - regression_values_[q]) / 
	regression_values_[q];
      
      if(rel_err > max_rel_err) {
	max_rel_err = rel_err;
      }
    }
    
    fx_format_result(NULL, "maxium_relative_error_for_fast_LPR", "%g", 
		     max_rel_err);
  }

};

#endif
