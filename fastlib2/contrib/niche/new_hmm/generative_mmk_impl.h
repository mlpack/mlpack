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

double KDEGenerativeMMK(double lambda,
			const Matrix &samples1,
			const Matrix &samples2,
			double bandwidth1,
			double bandwidth2) {
  
  int n_dims = samples1.n_rows();
  int n_samples1 = samples1.n_cols();
  int n_samples2 = samples2.n_cols();

  double h = ((double)1) + ((double)2) * lambda * (bandwidth1 + bandwidth2);
  double normalization_factor = pow(h, -((double)n_dims) / ((double)2));
  double neg_lambda_over_h = -lambda / h;

  double sum = 0;
  for(int i = 0; i < n_samples1; i++) {
    Vector x;
    samples1.MakeColumnVector(i, &x);

    for(int j = 0; j < n_samples2; j++) {
      Vector y;
      samples2.MakeColumnVector(j, &y);

      sum += exp(neg_lambda_over_h * la::DistanceSqEuclidean(x, y));
    }
  }
  
  return normalization_factor * sum / ((double)(n_samples1 * n_samples2));
}

// scales the data permanently!
void KDEGenerativeMMKBatch(double lambda,
			   const ArrayList<Matrix> &samplings,
			   Matrix *p_kernel_matrix) {
  Matrix &kernel_matrix = *p_kernel_matrix;

  int n_samplings = samplings.size();
  
  // we make the user do this call so we don't have to change samplings here
  //ScaleSamplingsToCube(&samplings);

  Vector optimal_bandwidths;
  optimal_bandwidths.Init(n_samplings);
  for(int k = 0; k < n_samplings; k++) {
    // Query and reference datasets, reference weight dataset.
    const Matrix &references = samplings[k];
    Matrix reference_weights;
    Matrix queries;
    references.PrintDebug("references");

    // data::Load inits a matrix with the contents of a .csv or .arff.
    queries.Alias(references);
    queries.PrintDebug("queries");
  
    // initialize to uniform weights.
    reference_weights.Init(1, queries.n_cols());
    reference_weights.SetAll(1);

    printf("sampling %d\n", k);
    optimal_bandwidths[k] =
      BandwidthLSCV::OptimizeReturn<GaussianKernelAux>(references, 
						       reference_weights);
  }

  optimal_bandwidths.PrintDebug("optimal_bandwidths");

  kernel_matrix.Init(n_samplings, n_samplings);
  for(int i = 0; i < n_samplings; i++) {
    printf("%f%%\n", ((double)(i + 1)) / ((double)n_samplings));
    for(int j = i; j < n_samplings; j++) {
      double gmmk = 
	KDEGenerativeMMK(lambda,
			 samplings[i], samplings[j],
			 optimal_bandwidths[i], optimal_bandwidths[j]);
      
      kernel_matrix.set(j, i, gmmk);
      if(i != j) {
	kernel_matrix.set(i, j, gmmk);
      }
    }
  }
}

void ScaleSamplingsToCube(ArrayList<Matrix> *p_samplings) {
  ArrayList<Matrix> &samplings = *p_samplings;

  int n_samplings = samplings.size();
  int n_dims = samplings[0].n_rows();

  DHrectBound<2> total_bound;
  total_bound.Init(n_dims);

  for(int k = 0; k < n_samplings; k++) {
    const Matrix &sampling = samplings[k];
    int n_points = sampling.n_cols();
    for(int i = 0; i < n_points; i++) {
      Vector point;
      sampling.MakeColumnVector(i, &point);
      total_bound |= point;
    }
  }

  Vector mins;
  mins.Init(n_dims);
  
  Vector ranges;
  ranges.Init(n_dims);
  
  for(int i = 0; i < n_dims; i++) {
    mins[i] = total_bound.get(i).lo;
    ranges[i] = total_bound.get(i).hi - mins[i];
  }
  
  for(int k = 0; k < n_samplings; k++) {
    Matrix &sampling = samplings[k];
    int n_points = sampling.n_cols();
    for(int i = 0; i < n_points; i++) {
      for(int j = 0; j < n_dims; j++) {
	sampling.set(j, i,
		    (sampling.get(j, i) - mins[j]) / ranges[j]);
      }
    }
  }
}
