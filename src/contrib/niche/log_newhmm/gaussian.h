#ifndef GAUSSIAN_H
#define GAUSSIAN_H

class Gaussian {
 public:

  int n_dims_;

  Vector* mu_;
  Matrix* sigma_;

  void Init(int n_dims_in) {
    n_dims_ = n_dims_in;

    mu_ = new Vector();
    mu_ -> Init(n_dims_);

    sigma_ = new Matrix();
    sigma_ -> Init(n_dims_, n_dims_);
  }
  
  void Init(const Vector &mu_in,
	    const Matrix &sigma_in) {
    mu_ = new Vector();
    mu_ -> Copy(mu_in);
    
    sigma_ = new Matrix();
    sigma_ -> Copy(sigma_in);

    n_dims_ = mu_ -> length();
  }
  
  void CopyValues(const Gaussian &other) {
    mu_ -> CopyValues(other.mu());
    sigma_ -> CopyValues(other.sigma());
  }
  
  int n_dims() const {
    return n_dims_;
  }

  const Vector mu () const {
    return *mu_;
  }

  const Matrix sigma() const {
    return *sigma_;
  }
  
  void SetMu(const Vector &mu_in) {
    mu_ -> CopyValues(mu_in);
  }

  void SetSigma(const Matrix &sigma_in) {
    sigma_ -> CopyValues(sigma_in);
  }
  
  void RandomlyInitialize() {
    sigma_ -> SetZero();
    for(int i = 0; i < n_dims_; i++) {
      (*mu_)[i] = drand48();
      
      // spherical covariance
      (*sigma_).set(i, i, 1);
    }
  }
  /*
  void SetZero() {
    mu_ -> SetZero();
    sigma_ -> SetZero();
  }
  
  void Accumulate(double weight, const Vector &example,
		  int component_num) {
    la::AddExpert(weight, example, mu_);
    
    Vector result;
    result.Init(n_dims_); // inefficient, don't want to have to init this every time we call Accumulate
    for(int i = 0; i < n_dims_; i++) {
      result[i] = example[i] * example[i];
    }
    la::AddExpert(weight * weight, result, sigma_);
  }

  void Normalize(double normalization_factor) {
    la::Scale(normalization_factor, mu_);
    la::Scale(normalization_factor * normalization_factor, sigma_);
    for(int i = 0; i < n_dims_; i++) {
      sigma_[i] -= mu_[i] * mu_[i];
    }
  }
  */
  void PrintDebug(const char *name = "", FILE *stream = stderr) const {
    fprintf(stream, name);
    mu_ -> PrintDebug("mu");
    sigma_ -> PrintDebug("sigma");
  }
    
  
  ~Gaussian() {
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
    
#endif /* GAUSSIAN_H */
