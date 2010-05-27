#ifndef LDS_H
#define LDS_H

#include "fastlib/fastlib.h"


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

  void Init(int n_dims_latent_in, int n_dims_obs_in, bool load, char* suffix) {
    n_dims_latent_ = n_dims_latent_in;
    n_dims_obs_ = n_dims_obs_in;

    if(load) {

      char *filename;
      filename = (char*) malloc(100 * sizeof(char));

      Matrix mu_0_Matrix;
      sprintf(filename, "mu_0_%s.dat", suffix);
      data::Load(filename, &mu_0_Matrix);
      mu_0_.Init(mu_0_Matrix.n_rows());
      mu_0_.CopyValues(mu_0_Matrix.ptr());

      sprintf(filename, "Sigma_0_%s.dat", suffix);
      data::Load(filename, &Sigma_0_);
      sprintf(filename, "A_%s.dat", suffix);
      data::Load("synth/A.dat", &A_);
      sprintf(filename, "C_%s.dat", suffix);
      data::Load("synth/C.dat", &C_);
      sprintf(filename, "Q_%s.dat", suffix);
      data::Load("synth/Q.dat", &Q_);
      sprintf(filename, "R_%s.dat", suffix);
      data::Load("synth/R.dat", &R_);
      
      free(filename);
      
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


};

#endif /* LDS_H */
