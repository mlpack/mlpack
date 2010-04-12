#ifndef HMM_DISTANCE_H
#define HMM_DISTANCE_H

#include "hmm.h"
#include "mmk.h"
#include "gaussian.h"

class HMM_Distance {

 public:

  static double Compute(HMM hmm_a, HMM hmm_b, int num_times) {
    MeanMapKernel<Gaussian> mmk;
    mmk.Init(1);
    
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

#endif /* HMM_DISTANCE_H */
