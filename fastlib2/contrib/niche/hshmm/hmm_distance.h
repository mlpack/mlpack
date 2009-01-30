#ifndef HMM_DISTANCE_H
#define HMM_DISTANCE_H

#include "hmm.h"

class HMM_Distance {

 public:

  static double Compute(HMM hmm_a, HMM hmm_b, int num_times) {
    //double dist = 0;
    
    int n1 = hmm_a.n_states();
    int n2 = hmm_b.n_states();

    Matrix phi;
    phi.Init(n1, n2);
    for(int i1 = 0; i1 < n1; i1++) {
      for(int i2 = 0; i2 < n2; i2++) {
	phi.set(i1, i2,
		ObservableKernel(hmm_a.state_distributions()[i1],
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
			   hmm_a.p_initial()[0] * hmm_b.p_initial()[0]);
      }
    }
    
    for(int t = 1; t <= num_times; t++) {
      psi_temp = psi_current;
      psi_current = psi_last;
      psi_last = psi_temp;
      
      for(int j1 = 0; j1 < n1; j1++) {
	for(int j2 = 0; j2 < n2; j2++) {
	  double sum = 0;
	  for(int i1 = 0; i1 < n1; i1++) {
	    for(int i2 = 0; i2 < n2; i2++) {
	      sum +=
		hmm_a.p_transition().get(j1, i1) *
		hmm_b.p_transition().get(j2, i2) *
		phi.get(i1, i2) *
		psi_last -> get(i1, i2);
	    }
	  }
	  psi_current -> set(j1, j2, sum);
	}
      }
    }

    double sum = 0;
    for(int i1 = 0; i1 < n1; i1++) {
      for(int i2 = 0; i2 < n2; i2++) {
	sum +=
	  phi.get(i1, i2) *
	  psi_current -> get(i1, i2);
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

  
  // Note: This is the multivariate Gaussian case.
  // TODO: mixture of Gaussian case (nearly trivial)
  static double ObservableKernel(Distribution x,
				 Distribution y) {
    double val = 1;

    int n = x.n_dims();    
    double lambda = 1; //NOTE: CHANGE THIS LATER

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
    sigma_x_inv.PrintDebug("sigma_x_inv");
    sigma_y_inv.PrintDebug("sigma_y_inv");
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


    printf("result = %f\n", result);

    alpha.PrintDebug("alpha");
    beta.PrintDebug("beta");
    printf("delta = %f\n", delta);

    /*

    // 3) compute the second, exponentiated term - wrong

    Matrix eye;
    eye.Init(n, n);
    eye.SetZero();
    for(int i = 0; i < n; i++) {
      eye.set(i, i, 1);
    }
    
    Matrix lambda_sigma_x;
    la::ScaleInit(lambda, sigma_x, &lambda_sigma_x);

    Matrix alpha;
    la::AddInit(lambda_sigma_x, eye, &alpha);
    la::Inverse(&alpha); // alpha = inv(lambda * sigma_x + I)
    
    Matrix A;
    la::AddInit(sigma_x, sigma_y, &A);
    la::Scale(lambda, &A);
    la::AddTo(eye, &A);
    la::Inverse(&A); // A = inv(lambda * (sigma_x + sigma_y) + I)


    Matrix beta;
    la::AddInit(sigma_x, sigma_y, &beta);
    la::Inverse(&beta); // beta = inv(sigma_x + sigma_y)

    Vector temp_vec1, temp_vec2;

    la::MulInit(x.mu(), alpha, &temp_vec1);
    la::MulInit(temp_vec1, sigma_y, &temp_vec2);
    la::MulOverwrite(temp_vec2, beta, &temp_vec1);
    la::MulOverwrite(temp_vec1, sigma_x, &temp_vec2);
    la::MulOverwrite(temp_vec2, alpha, &temp_vec1);
    double result = lambda * lambda * la::Dot(temp_vec1, x.mu());

    la::MulOverwrite(x.mu(), alpha, &temp_vec1);
    la::MulOverwrite(temp_vec1, sigma_x, &temp_vec2);
    la::MulOverwrite(temp_vec2, beta, &temp_vec1);
    result += lambda * la::Dot(temp_vec1, y.mu());

    la::MulOverwrite(x.mu(), alpha, &temp_vec1);
    la::MulOverwrite(temp_vec1, sigma_y, &temp_vec2);
    la::MulOverwrite(temp_vec2, A, &temp_vec1);
    result -= lambda * lambda * la::Dot(temp_vec1, x.mu());

    la::MulOverwrite(x.mu(), alpha, &temp_vec1);
    la::MulOverwrite(temp_vec1, sigma_y, &temp_vec2);
    la::MulOverwrite(temp_vec2, A, &temp_vec1);
    la::MulOverwrite(temp_vec1, sigma_y_inv, &temp_vec2);
    result -= lambda * la::Dot(temp_vec2, y.mu());

    la::MulOverwrite(x.mu(), alpha, &temp_vec1);
    la::MulOverwrite(temp_vec1, sigma_y, &temp_vec2);
    la::MulOverwrite(temp_vec2, A, &temp_vec1);
    la::MulOverwrite(temp_vec1, sigma_x, &temp_vec2);
    la::MulOverwrite(temp_vec2, sigma_y_inv, &temp_vec1);
    result -= lambda * lambda * la::Dot(temp_vec1, y.mu());

     
    la::MulOverwrite(y.mu(), beta, &temp_vec1);
    la::MulOverwrite(temp_vec1, sigma_x, &temp_vec2);
    la::MulOverwrite(temp_vec2, alpha, &temp_vec1);
    result += lambda * la::Dot(temp_vec1, x.mu());
    
    la::MulOverwrite(y.mu(), beta, &temp_vec1);
    la::MulOverwrite(temp_vec1, sigma_x, &temp_vec2);
    la::MulOverwrite(temp_vec2, sigma_y_inv, &temp_vec1);
    result += la::Dot(temp_vec1, y.mu());

    la::MulOverwrite(y.mu(), A, &temp_vec1);
    result -= lambda * la::Dot(temp_vec1, x.mu());

    la::MulOverwrite(y.mu(), A, &temp_vec1);
    la::MulOverwrite(temp_vec1, sigma_y_inv, &temp_vec2);
    result -= la::Dot(temp_vec2, y.mu());

    la::MulOverwrite(y.mu(), A, &temp_vec1);
    la::MulOverwrite(temp_vec1, sigma_x, &temp_vec2);
    la::MulOverwrite(temp_vec2, sigma_y_inv, &temp_vec1);
    result -= lambda * la::Dot(temp_vec1, y.mu());
    
    la::MulOverwrite(x.mu(), alpha, &temp_vec1);
    result += lambda * lambda * la::Dot(temp_vec1, x.mu());
    la::MulOverwrite(y.mu(), sigma_y_inv, &temp_vec1);
    result -= la::Dot(temp_vec1, y.mu());
    result -= lambda * la::Dot(x.mu(), x.mu());
    */

    double second_term = exp(result);

    val =
      first_term * second_term
      / (x_norm_constant * y_norm_constant);
    
    return val;
  }


  static void MatrixSqrtSymmetric(Matrix A, Matrix *sqrt_A) {
    Vector D_vector;
    Matrix D, E;
    la::EigenvectorsInit(A, &D_vector, &E);

    int n = D_vector.length();

    D.Init(n, n);
    D.SetZero();
    for(int i = 0; i < n; i++) {
      D.set(i, i, sqrt(D_vector[i]));
    }

    Matrix E_D;
    la::MulInit(E, D, &E_D);
    la::MulTransBInit(E_D, E, sqrt_A);
  }

  static void MatrixPowSymmetric(Matrix A, Matrix *pow_A, double exponent) {
    Vector D_vector;
    Matrix D, E;
    la::EigenvectorsInit(A, &D_vector, &E);

    int n = D_vector.length();

    D.Init(n, n);
    D.SetZero();
    for(int i = 0; i < n; i++) {
      D.set(i, i, pow(D_vector[i], exponent));
    }

    Matrix E_D;
    la::MulInit(E, D, &E_D);
    la::MulTransBInit(E_D, E, pow_A);
  }


};

#endif /* HMM_DISTANCE */
