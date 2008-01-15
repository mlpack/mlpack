#ifndef NAIVE_KDE_H
#define NAIVE_KDE_H

template<typename TKernel>
class NaiveKde {
  
  FORBID_ACCIDENTAL_COPIES(NaiveKde);

 private:
  
  /** query dataset */
  Matrix qset_;
  
  /** reference dataset */
  Matrix rset_;
  
  /** kernel */
  TKernel kernel_;

  /** computed densities */
  Vector densities_;
  
 public:

  NaiveKde() {
  }

  ~NaiveKde() {
  }

  void Compute() {

    printf("\nStarting naive KDE...\n");
    fx_timer_start(NULL, "naive_kde_compute");

    // compute unnormalized sum
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      
      const double *q_col = qset_.GetColumnPtr(q);
      for(index_t r = 0; r < rset_.n_cols(); r++) {
	const double *r_col = rset_.GetColumnPtr(r);
	double dsqd = la::DistanceSqEuclidean(qset_.n_rows(), q_col, r_col);
	
	densities_[q] += kernel_.EvalUnnormOnSq(dsqd);
      }
    }
    
    // then normalize it
    double norm_const = kernel_.CalcNormConstant(qset_.n_rows()) * 
      rset_.n_cols();
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      densities_[q] /= norm_const;
    }
    fx_timer_stop(NULL, "naive_kde_compute");
    printf("\nNaive KDE completed...\n");
  }

  void Init() {
    densities_.SetZero();
  }

  void Init(Matrix &qset, Matrix &rset) {

    // get datasets
    qset_.Alias(qset);
    rset_.Alias(rset);

    // get bandwidth
    kernel_.Init(fx_param_double_req(NULL, "bandwidth"));
    
    // allocate density storage
    densities_.Init(qset.n_cols());
    densities_.SetZero();
  }

  void PrintDebug() {

    FILE *stream = stdout;
    const char *fname = NULL;

    if(fx_param_exists(NULL, "naive_kde_output")) {
      fname = fx_param_str(NULL, "naive_kde_output", NULL);
      stream = fopen(fname, "w+");
    }
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      fprintf(stream, "%g\n", densities_[q]);
    }
    
    if(stream != stdout) {
      fclose(stream);
    }    
  }
  
  void ComputeMaximumRelativeError(const Vector &density_estimates) {
    
    double max_rel_err = 0;
    for(index_t q = 0; q < densities_.length(); q++) {
      double rel_err = fabs(density_estimates[q] - densities_[q]) / 
	densities_[q];

      if(rel_err > max_rel_err) {
	max_rel_err = rel_err;
      }
    }
    
    fx_format_result(NULL, "maxium_relative_error_for_fast_KDE", "%g", 
		     max_rel_err);
  }

};

#endif
