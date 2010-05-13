#ifndef DIAG_GAUSSIAN_H
#define DIAG_GAUSSIAN_H

class DiagGaussian {

 private:
  double norm_constant_;
  double log_norm_constant_;
  double min_variance_;

  OBJECT_TRAVERSAL_ONLY(DiagGaussian) {
    OT_PTR(mu_);
    OT_PTR(sigma_);
  }

 public:

  int n_dims_;

  Vector* mu_;
  Vector* sigma_;

  void Init(int n_dims_in) {
    Init(n_dims_in, 0);
  }
  
  void Init(int n_dims_in, double min_variance_in) {
    Init(n_dims_in, min_variance_in, 1);
  }

  void Init(int n_dims_in, double min_variance_in, int n_components_in) {
    n_dims_ = n_dims_in;
    min_variance_ = min_variance_in;

    mu_ = new Vector();
    mu_ -> Init(n_dims_);

    sigma_ = new Vector();
    sigma_ -> Init(n_dims_);
  }
  
  void Init(const Vector &mu_in,
	    const Vector &sigma_in) {
    mu_ = new Vector();
    mu_ -> Copy(mu_in);
    
    sigma_ = new Vector();
    sigma_ -> Copy(sigma_in);

    n_dims_ = mu_ -> length();
  }
  
  void CopyValues(const DiagGaussian &other) {
    mu_ -> CopyValues(other.mu());
    sigma_ -> CopyValues(other.sigma());
  }
  
  int n_dims() const {
    return n_dims_;
  }

  const Vector mu () const {
    return *mu_;
  }

  const Vector sigma() const {
    return *sigma_;
  }
  
  void SetMu(const Vector &mu_in) {
    mu_ -> CopyValues(mu_in);
  }

  void SetSigma(const Vector &sigma_in) {
    sigma_ -> CopyValues(sigma_in);
  }
  
  void RandomlyInitialize() {
    sigma_ -> SetZero();
    for(int i = 0; i < n_dims_; i++) {
      (*mu_)[i] = drand48();
      
      // spherical covariance
      (*sigma_)[i] = 1;
    }
    ComputeNormConstant();
  }

  void ComputeNormConstant() {
    double prod = 1;
    double sum = 0;
    for(int i = 0; i < n_dims_; i++) {
      prod *= (*sigma_)[i];
      sum += log((*sigma_)[i]);
    }
    norm_constant_ = 1 / (pow(2.0 * M_PI, n_dims_ / ((double)2)) * sqrt(prod));
    log_norm_constant_ = -0.5 * (((double)n_dims_) * log(2.0 * M_PI) + sum);
  }

  template<typename T>
    double PkthComponent(const GenVector<T> &x, int component_num) {
    return Pdf(x);
  }

  template<typename T>
    double LogPkthComponent(const GenVector<T> &x, int component_num) {
    return LogPdf(x);
  }
  
  template<typename T>
    double Pdf(const GenVector<T> &x) {
    double sum = 0;
    for(int i = 0; i < n_dims_; i++) {
      double diff = x[i] - (*mu_)[i];
      sum += (diff * diff / (*sigma_)[i]);
    }
    //sigma_ -> PrintDebug("sigma_");
    //printf("sum = %f, log_norm_constant = %f\n", sum, log_norm_constant_);
    //return exp(-0.5 * sum) * norm_constant_;
    //return exp(-0.5 * sum + log(norm_constant_));
    return exp(-0.5 * sum + log_norm_constant_);
  }
  
  template<typename T>
    double LogPdf(const GenVector<T> &x) {
    double sum = 0;
    for(int i = 0; i < n_dims_; i++) {
      double diff = x[i] - (*mu_)[i];
      sum += (diff * diff / (*sigma_)[i]);
    }
    //sigma_ -> PrintDebug("sigma_");
    //printf("sum = %f, log_norm_constant = %f\n", sum, log_norm_constant_);
    //return exp(-0.5 * sum) * norm_constant_;
    //return exp(-0.5 * sum + log(norm_constant_));
    return -0.5 * sum + log_norm_constant_;
  }


  void SetZero() {
    mu_ -> SetZero();
    sigma_ -> SetZero();
  }
  
  template<typename T>
    void Accumulate(double weight, const GenVector<T> &example,
		    int component_num) {
    la::AddExpert(weight, example, mu_);
    
    Vector result;
    result.Init(n_dims_); // inefficient, don't want to have to init this every time we call Accumulate
    for(int i = 0; i < n_dims_; i++) {
      result[i] = example[i] * example[i];
    }
    la::AddExpert(weight, result, sigma_);
  }

  void Normalize(double one_over_normalization_factor) {
    double normalization_factor = 
      ((double)1) / one_over_normalization_factor;
    //printf("normalization_factor = %e\n", normalization_factor);
    la::Scale(normalization_factor, mu_);
    la::Scale(normalization_factor, sigma_);
    for(int i = 0; i < n_dims_; i++) {
      (*sigma_)[i] -= (*mu_)[i] * (*mu_)[i];
      
      if((*sigma_)[i] < min_variance_) {
	printf("was %f\n", (*sigma_)[i]);
	(*sigma_)[i] = min_variance_;
      }
      else {
	//printf("is %f\n", (*sigma_)[i]);
      }
      
    }
    ComputeNormConstant();
  }

  void Normalize(double one_over_normalization_factor,
		 const DiagGaussian &alternate_distribution) {
    /*
      if(isinf((*mu_)[0])) {
      printf("before - mu is infinite\n");
      }
   
      printf("mu = [%e %e]\n", (*mu_)[0], (*mu_)[1]);
      (*mu_).PrintDebug("mu_");
      printf("one_over_normalization_factor = %e\n",
      one_over_normalization_factor);
    */
    if(one_over_normalization_factor > 1e-100) {
      double normalization_factor = 
	((double)1) / one_over_normalization_factor;
      //printf("normalization_factor = %e\n", normalization_factor);
      la::Scale(normalization_factor, mu_);
      la::Scale(normalization_factor, sigma_);
      for(int i = 0; i < n_dims_; i++) {
	(*sigma_)[i] -= (*mu_)[i] * (*mu_)[i];
	
	if((*sigma_)[i] < min_variance_) {
	  (*sigma_)[i] = min_variance_;
	}
	
      }
      ComputeNormConstant();
    }
    else {
      //printf("alt: %e\n", (*(alternate_distribution.sigma_))[0]);
      mu_ -> CopyValues(*(alternate_distribution.mu_));
      sigma_ -> CopyValues(*(alternate_distribution.sigma_));
      ComputeNormConstant(); // should not be necessary
      //printf("alternate\n");
    }
    /*
      if(isinf((*mu_)[0])) {
      printf("after - mu is infinite\n");
      exit(1);
      }
      else if((*sigma_)[0] < min_variance_) {
      printf("after: (*sigma_)[0] < min_variance_: %e\n", (*sigma_)[0]);
      exit(1);
      }
    */
  }

  void PrintDebug(const char *name = "", FILE *stream = stderr) const {
    fprintf(stream, name);
    mu_ -> PrintDebug("mu");
    sigma_ -> PrintDebug("sigma");
/*     for(int i = 0; i < n_dims_; i++) { */
/*       printf("sigma[%d] = %e\n", i, (*sigma_)[i]); */
/*     } */

  }
    
  
  ~DiagGaussian() {
    Destruct();
  }

  void Destruct() {
    delete mu_;
    delete sigma_;
  }


  /*  
  // Note: This is the multivariate Gaussian case.
  // TODO: mixture of Gaussian case (nearly trivial)
  static double Compute(Gaussian x,
  Gaussian y,
  double lambda,
  int n_T) {
  double val = 1;

  int n = x.n_dims();    
 
  Matrix sigma_x, sigma_y;

  sigma_x.Copy(x.sigma());
  sigma_y.Copy(y.sigma());


  // 1) compute norm constants by which result should eventually be divided
  double x_norm_constant =
  pow(2 * M_PI, ((double)n) / 2.)
  * sqrt(fabs(la::Determinant(sigma_x)));
  double y_norm_constant =
  pow(2 * M_PI, ((double)n) / 2.)
  * sqrt(fabs(la::Determinant(sigma_y)));

  // 2) compute the first, non-exponentiated term
    
  // take care of the 1/2 scalars in the exponent
  la::Scale(2, &sigma_x);
  la::Scale(2, &sigma_y);

  Matrix sigma_x_inv;
  la::InverseInit(sigma_x, &sigma_x_inv);

  Matrix sigma_y_inv;
  la::InverseInit(sigma_y, &sigma_y_inv);

  Matrix product;
  la::MulInit(sigma_x_inv, sigma_y_inv, &product);

  Matrix sum;
  la::AddInit(sigma_x_inv, sigma_y_inv, &sum);
    
  la::Scale(lambda, &sum);

  la::AddTo(product, &sum);

  double first_term = pow(M_PI, n) / sqrt(fabs(la::Determinant(sum)));


  // 3) compute the second, exponentiated term - correct

  Matrix lambda_I;
  lambda_I.Init(n, n);
  lambda_I.SetZero();
  for(int i = 0; i < n; i++) {
  lambda_I.set(i, i, lambda);
  }
  Matrix sigma_x_inv_plus_lambda_I;
  la::AddInit(sigma_x_inv, lambda_I, &sigma_x_inv_plus_lambda_I);
  //Matrix A;
  //HMM_Distance::MatrixSqrtSymmetric(sigma_x_inv_plus_lambda_I, &A);
  Matrix B;
  la::InverseInit(sigma_x_inv_plus_lambda_I, &B);

  Matrix temp_mat;
  Matrix alpha;
  la::MulInit(sigma_x_inv, B, &temp_mat);
  la::MulInit(temp_mat, sigma_x_inv, &alpha);
  la::Scale(-1, &alpha);
  la::AddTo(sigma_y_inv, &alpha);
  la::AddTo(sigma_x_inv, &alpha);

  Vector beta;
  la::MulOverwrite(sigma_x_inv, B, &temp_mat);
  la::MulInit(temp_mat, x.mu(), &beta);
  la::Scale(lambda, &beta);
  Vector temp_vec;
  la::MulInit(sigma_y_inv, y.mu(), &temp_vec);
  la::AddTo(temp_vec, &beta);
  la::Scale(-2, &beta);

  double delta;
  la::MulOverwrite(x.mu(), B, &temp_vec);
  delta = -lambda * lambda * la::Dot(temp_vec, x.mu());
  la::MulOverwrite(y.mu(), sigma_y_inv, &temp_vec);
  delta += la::Dot(temp_vec, y.mu());
  delta += lambda * la::Dot(x.mu(), x.mu());

  Matrix alpha_inv;
  la::InverseInit(alpha, &alpha_inv);
  la::MulOverwrite(beta, alpha_inv, &temp_vec);
  double result = (la::Dot(temp_vec, beta) / 4) - delta;


  double second_term = exp(result);

  val =
  first_term * second_term
  / (x_norm_constant * y_norm_constant);
    
  return val;
  }
  */
};
    
#endif /* DIAG_GAUSSIAN_H */
