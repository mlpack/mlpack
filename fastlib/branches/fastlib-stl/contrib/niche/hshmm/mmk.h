#ifndef MMK_H
#define MMK_H

#include "multinomial.h"
#include "gaussian.h"
#include "hmm.h"
#include "lds.h"

class MeanMapKernel {

 private:
  double lambda_;
  int n_T_;
  
 public:
  void Init(double lambda_in) {
    Init(lambda_in, 1);
  }
  
  void Init(double lambda_in, int n_T_in) {
    lambda_ = lambda_in;
    n_T_ = n_T_in;
  }

  void SetLambda(double lambda_in) {
    lambda_ = lambda_in;
  }

  
  double Compute(const Multinomial &x, const Multinomial &y) {

    int n_dims = x.n_dims();

    double sum = 0;
    for(int i = 0; i < n_dims; i++) {
      for(int j = 0; j < n_dims; j++) {
	if(i == j) {
	  sum += (x.p()[i] * y.p()[j]);
	}
	else {
	  sum += (x.p()[i] * y.p()[j] * exp(-lambda_));
	}
      }
    }

    return sum;
  }


  // Note: This is the multivariate Gaussian case.
  // TODO: mixture of Gaussian case (nearly trivial)
  double Compute(const Gaussian &x, const Gaussian &y) {
    double val = -1;

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
    
    la::Scale(lambda_, &sum);

    la::AddTo(product, &sum);

    double first_term = pow(M_PI, n) / sqrt(fabs(la::Determinant(sum)));


    // 3) compute the second, exponentiated term - correct

    Matrix lambda_I;
    lambda_I.Init(n, n);
    lambda_I.SetZero();
    for(int i = 0; i < n; i++) {
      lambda_I.set(i, i, lambda_);
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
    la::Scale(lambda_, &beta);
    Vector temp_vec;
    la::MulInit(sigma_y_inv, y.mu(), &temp_vec);
    la::AddTo(temp_vec, &beta);
    la::Scale(-2, &beta);

    double delta;
    la::MulOverwrite(x.mu(), B, &temp_vec);
    delta = -lambda_ * lambda_ * la::Dot(temp_vec, x.mu());
    la::MulOverwrite(y.mu(), sigma_y_inv, &temp_vec);
    delta += la::Dot(temp_vec, y.mu());
    delta += lambda_ * la::Dot(x.mu(), x.mu());

    Matrix alpha_inv;
    la::InverseInit(alpha, &alpha_inv);
    la::MulOverwrite(beta, alpha_inv, &temp_vec);
    double result = (la::Dot(temp_vec, beta) / 4) - delta;


    double second_term = exp(result);

    val =
      first_term * second_term
      / (x_norm_constant * y_norm_constant);
    //printf("val = %f\n", val);
    return val;
  }




  template <typename TDistribution>
  double Compute(const HMM<TDistribution> &hmm_a,
		 const HMM<TDistribution> &hmm_b) const {
    MeanMapKernel mmk;
    mmk.Init(lambda_);

    bool slow_computation = false;

    if(slow_computation == true) {
    
      int n1 = hmm_a.n_states();
      int n2 = hmm_b.n_states();
    


      Matrix phi;
      phi.Init(n1, n2);
      for(int i1 = 0; i1 < n1; i1++) {
	for(int i2 = 0; i2 < n2; i2++) {
	  phi.set(i1, i2,
		  mmk.Compute(hmm_a.state_distributions()[i1],
			      hmm_b.state_distributions()[i2]));
	
	}
      }
    
      Matrix* psi_last;
      Matrix* psi_current;
      Matrix* psi_temp;
      psi_last    = new Matrix(n1, n2);
      psi_current = new Matrix(n1, n2);
    
      for(int i1 = 0; i1 < n1; i1++) {
	for(int i2 = 0; i2 < n2; i2++) {
	  psi_current -> set(i1, i2,
			     hmm_a.p_initial()[i1] * hmm_b.p_initial()[i2]);
	  //printf("psi_current[i1,i2] = %f\n",
	  //       psi_current -> get(i1, i2));
	}
      }
    
      for(int t = 1; t <= n_T_; t++) {
	psi_temp = psi_current;
	psi_current = psi_last;
	psi_last = psi_temp;
      
	for(int j1 = 0; j1 < n1; j1++) {
	  for(int j2 = 0; j2 < n2; j2++) {
	    double sum = 0;
	    for(int i1 = 0; i1 < n1; i1++) {
	      for(int i2 = 0; i2 < n2; i2++) {
		sum +=
		  hmm_a.p_transition().get(i1, j1) * // switched from .get(j1, i1)
		  hmm_b.p_transition().get(i2, j2) * // switched from .get(j2, i2)
		  phi.get(i1, i2) *
		  psi_last -> get(i1, i2);
	      }
	    }
	    psi_current -> set(j1, j2, sum);
	    //printf("psi_current[i1,i2] = %f\n",
	    //	 psi_current -> get(j1, j2));

	  }
	}
      }

      double sum = 0;
      for(int i1 = 0; i1 < n1; i1++) {
	for(int i2 = 0; i2 < n2; i2++) {
	  sum +=
	    phi.get(i1, i2) *
	    psi_current -> get(i1, i2);
	  //printf("psi_current[i1,i2] = %f\n",
	  //     psi_current -> get(i1, i2));
	}
      }

      delete psi_last;
      delete psi_current;
      return sum;
    
	
      /*
      // let Q and Q` be the set of states from hmm_a and hmm_b respectively

      for each q in Q, q` in Q`
      psi(q,q`) = E_{x,x`}[k(x,x`)]
      end

      for each q_0,q`_0
      phi(q_0,q`_0) = p(q_0) * p(q_0`)
      end

      for t = 1:T
      for each q_t,q`_t
      phi(q_t,q`_t) =
      \sum_{q_{t-1},q`_{t-1}}
      p(q_t | q_{t-1}) * p(q`_t | q`_{t-1})
      * phi(q_{t-1}, q`_{t-1})
      * psi(q_{t-1},q`_{t-1})
      end
      end

      dist =
      \sum_{q_T,q`_T}
      phi(q_T,q`_T) psi(q_T,q`_T)
      */
    }
    else {
      int n1 = hmm_a.n_states();
      int n2 = hmm_b.n_states();

      Matrix psi;
      psi.Init(n2, n1);
      for(int i1 = 0; i1 < n1; i1++) {
	for(int i2 = 0; i2 < n2; i2++) {
	  psi.set(i2, i1,
		  mmk.Compute(hmm_a.state_distributions()[i1],
			      hmm_b.state_distributions()[i2]));
	}
      }

      Matrix phi;
      phi.Init(n2, n1);
      double unif1 = ((double)1) / ((double)n1);
      double unif2 = ((double)2) / ((double)n2);
      for(int i1 = 0; i1 < n1; i1++) {
	for(int i2 = 0; i2 < n2; i2++) {
	  phi.set(i2, i1,
		  unif1 * unif2);//hmm_a.p_initial()[i1] * hmm_b.p_initial()[i2]);
	}
      }

      for(int i1 = 0; i1 < n1; i1++) {
	for(int i2 = 0; i2 < n2; i2++) {
	  phi.set(i2, i1, phi.get(i2, i1) * psi.get(i2, i1));
	}
      }

      Matrix transition2_transpose;
      la::TransposeInit(hmm_b.p_transition(), &transition2_transpose);

      Matrix temp1;
      temp1.Init(n2, n1);

      // main iteration
      for(int t = 1; t <= n_T_; t++) {
	la::MulOverwrite(transition2_transpose, phi, &temp1);
	la::MulOverwrite(temp1, hmm_a.p_transition(), &phi);
	for(int i1 = 0; i1 < n1; i1++) {
	  for(int i2 = 0; i2 < n2; i2++) {
	    phi.set(i2, i1, phi.get(i2, i1) * psi.get(i2, i1));
	  }
	}
      }

      double sum = 0;
      for(int i1 = 0; i1 < n1; i1++) {
	for(int i2 = 0; i2 < n2; i2++) {
	  sum += phi.get(i2, i1);
	}
      }
      return sum;
    }
  }


double Compute(LDS lds_1, LDS lds_2) {
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


    
    MeanMapKernel mmk;
    mmk.Init(lambda_);


    // first one //

    // compute observable distributions

    la::MulInit(C_1, *mu_1_q_t, &mu_1_x_t);
    la::MulInit(C_1, Sigma_1_q_t, &temp_obs_1);
    la::MulInit(temp_obs_1, C_1_transpose, &Sigma_1_x_t);
    la::AddTo(R_1, &Sigma_1_x_t);
    
    la::MulInit(C_2, *mu_2_q_t, &mu_2_x_t);
    la::MulInit(C_2, Sigma_2_q_t, &temp_obs_2);
    la::MulInit(temp_obs_2, C_2_transpose, &Sigma_2_x_t);
    la::AddTo(R_2, &Sigma_2_x_t);
    
    Gaussian gaussian_1;
    gaussian_1.Init(mu_1_x_t, Sigma_1_x_t);
    
    Gaussian gaussian_2;
    gaussian_2.Init(mu_2_x_t, Sigma_2_x_t);

    double val = mmk.Compute(gaussian_1, gaussian_2);
    
    // end first one //
    

    for(int t = 0; t < n_T_; t++) {
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

      gaussian_1.SetMu(mu_1_x_t);
      gaussian_1.SetSigma(Sigma_1_x_t);

      gaussian_2.SetMu(mu_2_x_t);
      gaussian_2.SetSigma(Sigma_2_x_t);

      val *= mmk.Compute(gaussian_1, gaussian_2);
    }

    delete mu_1_q_t;
    delete mu_1_q_t_plus_1;
    delete mu_2_q_t;
    delete mu_2_q_t_plus_1;
    
    return val;
  }


};




#endif /* MMK_H */

