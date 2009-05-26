#ifndef INSIDE_GENERATIVE_MMK_IMPL_H
#error "This is not a public header file!"
#endif

double GenerativeMMK(double lambda,
		     const Multinomial &x,
		     const Multinomial &y) {
  int n_dims = x.n_dims();
  
  double exp_neg_lambda = exp(-lambda);
  
  double sum = 0;
  for(int i = 0; i < n_dims; i++) {
    for(int j = 0; j < n_dims; j++) {
      if(i == j) {
	sum += (x.p()[i] * y.p()[j]);
      }
      else {
	sum += (x.p()[i] * y.p()[j] * exp_neg_lambda);
      }
    }
  }
  
  return sum;
}

double GenerativeMMK(double lambda,
		     const IsotropicGaussian &x,
		     const IsotropicGaussian &y) {
  double h = 1 + lambda * (x.sigma() + y.sigma());
  int n_dims = x.n_dims();
  
  return 
    pow(h, -((double)n_dims) / ((double)2))
    * exp(-lambda * la::DistanceSqEuclidean(x.mu(), y.mu()) / h);
}

template <typename TDistribution>
  double GenerativeMMK(double lambda, int n_T,
		       const HMM<TDistribution> &hmm_a,
		       const HMM<TDistribution> &hmm_b) {
  
  int n1 = hmm_a.n_states();
  int n2 = hmm_b.n_states();
  
  Matrix psi;
  psi.Init(n2, n1);
  for(int i1 = 0; i1 < n1; i1++) {
    for(int i2 = 0; i2 < n2; i2++) {
      psi.set(i2, i1,
	      GenerativeMMK(lambda,
			    hmm_a.state_distributions[i1],
			    hmm_b.state_distributions[i2]));
    }
  }

  Matrix phi;
  phi.Init(n2, n1);
  //double unif1 = ((double)1) / ((double)n1);
  //double unif2 = ((double)2) / ((double)n2);
  for(int i1 = 0; i1 < n1; i1++) {
    for(int i2 = 0; i2 < n2; i2++) {
      phi.set(i2, i1,
	      hmm_a.p_initial[i1] * hmm_b.p_initial[i2]);//unif1 * unif2);
    }
  }

  for(int i1 = 0; i1 < n1; i1++) {
    for(int i2 = 0; i2 < n2; i2++) {
      phi.set(i2, i1, phi.get(i2, i1) * psi.get(i2, i1));
    }
  }

  Matrix transition2_transpose;
  la::TransposeInit(hmm_b.p_transition, &transition2_transpose);

  Matrix temp1;
  temp1.Init(n2, n1);

  // main iteration
  for(int t = 1; t <= n_T; t++) {
    la::MulOverwrite(transition2_transpose, phi, &temp1);
    la::MulOverwrite(temp1, hmm_a.p_transition, &phi);
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

template <typename TDistribution>
void GenerativeMMKBatch(double lambda, int n_T,
			const ArrayList<HMM<TDistribution> > &hmms,
			Matrix *p_kernel_matrix) {
  Matrix &kernel_matrix = *p_kernel_matrix;

  int n_hmms = hmms.size();
  
  kernel_matrix.Init(n_hmms, n_hmms);
  for(int i = 0; i < n_hmms; i++) {
    printf("%f%%\n", ((double)(i + 1)) / ((double)n_hmms));
    for(int j = i; j < n_hmms; j++) {
      double gmmk = 
	GenerativeMMK(lambda, n_T, hmms[i], hmms[j]);

      kernel_matrix.set(j, i, gmmk);
      if(i != j) {
	kernel_matrix.set(i, j, gmmk);
      }
    }
  }
}
