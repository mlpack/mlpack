#ifndef LDS_H
#define LDS_H

#include "fastlib/fastlib.h"
#include "gaussian.h"
#include "mmk.h"


class LDS {
 private:

  int n_dims_latent_;
  int n_dims_obs_;

  Vector mu_0_;
  Matrix Sigma_0_;

  Matrix A_;
  Matrix C_;
  Matrix Q_;
  Matrix R_;
  
 public:

  void Init(int n_dims_latent_in, int n_dims_obs_in, bool load) {
    n_dims_latent_ = n_dims_latent_in;
    n_dims_obs_ = n_dims_obs_in;

    if(load) {
      Matrix mu_0_Matrix;
      data::Load("synth/mu_0.dat", &mu_0_Matrix);
      mu_0_.Init(mu_0_Matrix.n_rows());
      mu_0_.CopyValues(mu_0_Matrix.ptr());

      data::Load("synth/Sigma_0.dat", &Sigma_0_);
      data::Load("synth/A.dat", &A_);
      data::Load("synth/C.dat", &C_);
      data::Load("synth/Q.dat", &Q_);
      data::Load("synth/R.dat", &R_);
      
      
      if((mu_0_.length() != n_dims_latent_)
	 || (Sigma_0_.n_rows() != n_dims_latent_)
	 || (Sigma_0_.n_cols() != n_dims_latent_)
	 || (A_.n_rows() != n_dims_latent_)
	 || (A_.n_cols() != n_dims_latent_)
	 || (C_.n_rows() != n_dims_obs_)
	 || (C_.n_cols() != n_dims_latent_)
	 || (Q_.n_rows() != n_dims_latent_)
	 || (Q_.n_cols() != n_dims_latent_)
	 || (R_.n_cols() != n_dims_obs_)
	 || (R_.n_cols() != n_dims_obs_)) {
	printf("error, dimension mismatch somewhere\n");
	exit(1);
      }
    }
    else {
      mu_0_.Init(n_dims_latent_);
      Sigma_0_.Init(n_dims_latent_, n_dims_latent_);
      
      A_.Init(n_dims_latent_, n_dims_latent_);
      C_.Init(n_dims_obs_, n_dims_latent_);
      Q_.Init(n_dims_latent_, n_dims_latent_);
      R_.Init(n_dims_obs_, n_dims_obs_);
    }
  }

  int n_dims_latent() {
    return n_dims_latent_;
  }

  int n_dims_obs() {
    return n_dims_obs_;
  }

  const Matrix& A () const {
    return A_;
  }

  const Matrix& C () const {
    return C_;
  }

  const Matrix& Q () const {
    return Q_;
  }

  const Matrix& R () const {
    return R_;
  }

  const Vector& mu_0 () const {
    return mu_0_;
  }

  const Matrix& Sigma_0 () const {
    return Sigma_0_;
  }

  static double Compute(LDS lds_1, LDS lds_2, double lambda, int n_T) {
    // recursion for computing the covariance matrices and means of the gaussians for the MMK
    Vector* mu_1_q_t = new Vector();
    mu_1_q_t -> Copy(lds_1.mu_0());
    Vector* mu_1_q_t_plus_1 = new Vector();
    mu_1_q_t_plus_1 -> Init(mu_1_q_t -> length());

    Vector* mu_2_q_t = new Vector();
    mu_2_q_t -> Copy(lds_2.mu_0());
    Vector* mu_2_q_t_plus_1 = new Vector();
    mu_2_q_t_plus_1 -> Init(mu_2_q_t -> length());


    Vector* temp_mu;

    
    Matrix Sigma_1_q_t;
    Sigma_1_q_t.Copy(lds_1.Sigma_0());

    Matrix Sigma_2_q_t;
    Sigma_2_q_t.Copy(lds_2.Sigma_0());


    Matrix temp_latent_1;
    temp_latent_1.Init(lds_1.n_dims_latent(), lds_1.n_dims_latent());
    
    Matrix temp_latent_2;
    temp_latent_2.Init(lds_2.n_dims_latent(), lds_2.n_dims_latent());


    Vector mu_1_x_t;

    Vector mu_2_x_t;


    Matrix Sigma_1_x_t;

    Matrix Sigma_2_x_t;


    Matrix temp_obs_1;

    Matrix temp_obs_2;


    Matrix A_1, A_1_transpose;
    A_1.Copy(lds_1.A());
    la::TransposeInit(A_1, &A_1_transpose);

    Matrix A_2, A_2_transpose;
    A_2.Copy(lds_2.A());
    la::TransposeInit(A_2, &A_2_transpose);


    Matrix C_1, C_1_transpose;
    C_1.Copy(lds_1.C());
    la::TransposeInit(C_1, &C_1_transpose);

    Matrix C_2, C_2_transpose;
    C_2.Copy(lds_2.C());
    la::TransposeInit(C_2, &C_2_transpose);


    Matrix Q_1;
    Q_1.Copy(lds_1.Q());

    Matrix Q_2;
    Q_2.Copy(lds_2.Q());


    Matrix R_1;
    R_1.Copy(lds_1.R());

    Matrix R_2;
    R_2.Copy(lds_2.R());



    MeanMapKernel<Gaussian> mmk_gaussian;
    mmk_gaussian.Init(lambda);


    /* first one */

    // compute observable disxtributions

    la::MulInit(C_1, *mu_1_q_t, &mu_1_x_t);
    la::MulInit(C_1, Sigma_1_q_t, &temp_obs_1);
    la::MulInit(temp_obs_1, C_1_transpose, &Sigma_1_x_t);
    la::AddTo(R_1, &Sigma_1_x_t);
    
    la::MulInit(C_2, *mu_2_q_t, &mu_2_x_t);
    la::MulInit(C_2, Sigma_2_q_t, &temp_obs_2);
    la::MulInit(temp_obs_2, C_2_transpose, &Sigma_2_x_t);
    la::AddTo(R_2, &Sigma_2_x_t);
    
    Gaussian gaussian_1;
    gaussian_1.Init(&mu_1_x_t, &Sigma_1_x_t);
    
    Gaussian gaussian_2;
    gaussian_2.Init(&mu_2_x_t, &Sigma_2_x_t);

    double val = mmk_gaussian.Compute(gaussian_1, gaussian_2);
    
    /* end first one */
    

    for(int t = 0; t < n_T; t++) {
      // update latent variables for next time step
      la::MulOverwrite(A_1, *mu_1_q_t, mu_1_q_t_plus_1);
      la::MulOverwrite(A_1, Sigma_1_q_t, &temp_latent_1);
      la::MulOverwrite(temp_latent_1, A_1_transpose, &Sigma_1_q_t);
      la::AddTo(Q_1, &Sigma_1_q_t);

      la::MulOverwrite(A_2, *mu_2_q_t, mu_2_q_t_plus_1);
      la::MulOverwrite(A_2, Sigma_2_q_t, &temp_latent_2);
      la::MulOverwrite(temp_latent_2, A_2_transpose, &Sigma_2_q_t);
      la::AddTo(Q_2, &Sigma_2_q_t);
      
      temp_mu = mu_1_q_t;
      mu_1_q_t = mu_1_q_t_plus_1;
      mu_1_q_t_plus_1 = temp_mu;
      
      temp_mu = mu_2_q_t;
      mu_2_q_t = mu_2_q_t_plus_1;
      mu_2_q_t_plus_1 = temp_mu;
            

      // compute observable distributions
      la::MulOverwrite(C_1, *mu_1_q_t, &mu_1_x_t);
      la::MulOverwrite(C_1, Sigma_1_q_t, &temp_obs_1);
      la::MulOverwrite(temp_obs_1, C_1_transpose, &Sigma_1_x_t);
      la::AddTo(R_1, &Sigma_1_x_t);

      la::MulOverwrite(C_2, *mu_2_q_t, &mu_2_x_t);
      la::MulOverwrite(C_2, Sigma_2_q_t, &temp_obs_2);
      la::MulOverwrite(temp_obs_2, C_2_transpose, &Sigma_2_x_t);
      la::AddTo(R_2, &Sigma_2_x_t);

      gaussian_1.SetMu(&mu_1_x_t);
      gaussian_1.SetSigma(&Sigma_1_x_t);

      gaussian_2.SetMu(&mu_2_x_t);
      gaussian_2.SetSigma(&Sigma_2_x_t);

      val *= mmk_gaussian.Compute(gaussian_1, gaussian_2);
    }

    delete mu_1_q_t;
    delete mu_1_q_t_plus_1;
    delete mu_2_q_t;
    delete mu_2_q_t_plus_1;
    
    return val;
  }


};

#endif /* LDS_H */
