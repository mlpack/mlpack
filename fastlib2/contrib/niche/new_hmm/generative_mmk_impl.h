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
    * exp(-0.5 * lambda * la::DistanceSqEuclidean(x.mu(), y.mu()) / h);
}

double GenerativeMMK(double lambda,
		     const DiagGaussian &x,
		     const DiagGaussian &y) {

  int n_dims = x.n_dims();    

  Vector normalization_factors;
  normalization_factors.Init(n_dims);
  for(int i = 0; i < n_dims; i++) {
    normalization_factors[i] = 1 / (1 + lambda * (x.sigma()[i] + y.sigma()[i]));
  }

  double global_normalization_factor = 1;
  for(int i = 0; i < n_dims; i++) {
    global_normalization_factor *= normalization_factors[i];
  }
  global_normalization_factor = sqrt(global_normalization_factor);

  double sum = 0;
  for(int i = 0; i < n_dims; i++) {
    double mu_diff = x.mu()[i] - y.mu()[i];
    sum += (mu_diff * mu_diff) * normalization_factors[i];
  }
  sum *= -lambda / 2;
  
  return global_normalization_factor * exp(sum * -lambda / 2);
}


// TODO: create log probability equivalent of this function
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

double GenerativeMMK(double lambda, double rho, int n_T,
		     const HMM<Multinomial> &hmm_a,
		     const HMM<Multinomial> &hmm_b) {
  
  int n1 = hmm_a.n_states();
  int n2 = hmm_b.n_states();
  
  int n_dims = hmm_a.n_dims();
  
  Multinomial* state_distributions1;
  state_distributions1 =
    (Multinomial*) malloc(n1 * sizeof(Multinomial));
  for(int i = 0; i < n1; i++) {
    state_distributions1[i].Init(hmm_a.state_distributions[i].p());
    for(int j = 0; j < n_dims; j++) {
      (*(state_distributions1[i].p_))[j] =
	pow((*(state_distributions1[i].p_))[j], rho);
    }
  }

  Multinomial* state_distributions2;
  state_distributions2 =
    (Multinomial*) malloc(n2 * sizeof(Multinomial));
  for(int i = 0; i < n2; i++) {
    state_distributions2[i].Init(hmm_b.state_distributions[i].p());
    for(int j = 0; j < n_dims; j++) {
      (*(state_distributions2[i].p_))[j] =
	pow((*(state_distributions2[i].p_))[j], rho);
    }
  }


  
  Matrix psi;
  psi.Init(n2, n1);
  for(int i1 = 0; i1 < n1; i1++) {
    for(int i2 = 0; i2 < n2; i2++) {
      psi.set(i2, i1,
	      GenerativeMMK(lambda,
			    state_distributions1[i1],
			    state_distributions2[i2]));
    }
  }

  Matrix phi;
  phi.Init(n2, n1);
  //double unif1 = ((double)1) / ((double)n1);
  //double unif2 = ((double)2) / ((double)n2);
  for(int i1 = 0; i1 < n1; i1++) {
    for(int i2 = 0; i2 < n2; i2++) {
      phi.set(i2, i1,
	      pow(hmm_a.p_initial[i1] * hmm_b.p_initial[i2], rho));
      //unif1 * unif2);
    }
  }

  for(int i1 = 0; i1 < n1; i1++) {
    for(int i2 = 0; i2 < n2; i2++) {
      phi.set(i2, i1, phi.get(i2, i1) * psi.get(i2, i1));
    }
  }


  Matrix transition1;
  transition1.Copy(hmm_a.p_transition);
  for(int j = 0; j < n1; j++) {
    for(int i = 0; i < n1; i++) {
      transition1.set(i, j,
		      pow(transition1.get(i, j), rho));
    }
  }

  Matrix transition2_transpose;
  la::TransposeInit(hmm_b.p_transition, &transition2_transpose);
  for(int i = 0; i < n2; i++) {
    for(int j = 0; j < n2; j++) {
      transition2_transpose.set(j, i,
				pow(transition2_transpose.get(j, i), rho));
    }
  }

  
  Matrix temp1;
  temp1.Init(n2, n1);

  // main iteration
  for(int t = 1; t <= n_T; t++) {
    la::MulOverwrite(transition2_transpose, phi, &temp1);
    la::MulOverwrite(temp1, transition1, &phi);
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
    printf("%f%%\n", 100.0 * ((double)(i + 1)) / ((double)n_hmms));
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

void GenerativeMMKBatch(double lambda, double rho, int n_T,
			const ArrayList<HMM<Multinomial> > &hmms,
			Matrix *p_kernel_matrix) {
  Matrix &kernel_matrix = *p_kernel_matrix;

  int n_hmms = hmms.size();
  
  kernel_matrix.Init(n_hmms, n_hmms);
  for(int i = 0; i < n_hmms; i++) {
    printf("%f%%\n", 100.0 * ((double)(i + 1)) / ((double)n_hmms));
    for(int j = i; j < n_hmms; j++) {
      double gmmk = 
	GenerativeMMK(lambda, rho, n_T, hmms[i], hmms[j]);

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

  double h = 1.0 + (lambda * (bandwidth1 + bandwidth2));
  double normalization_factor = pow(h, -((double)n_dims) / 2.0);
  double neg_half_lambda_over_h = -0.5 * lambda / h;

  double sum = 0;
  for(int i = 0; i < n_samples1; i++) {
    Vector x;
    samples1.MakeColumnVector(i, &x);

    for(int j = 0; j < n_samples2; j++) {
      Vector y;
      samples2.MakeColumnVector(j, &y);

      double val = exp(neg_half_lambda_over_h * la::DistanceSqEuclidean(x, y));
      /*
	if((val > 0) && (i != j)) {
	printf("hit!\n");
	x.PrintDebug("x");
	y.PrintDebug("y");
	printf("dist_sq = %3e\n", la::DistanceSqEuclidean(x, y));
	printf("neg_half_lambda_over_h = %3e\n", neg_half_lambda_over_h);
	}
      */
      //x.PrintDebug("x");
      //y.PrintDebug("y");
      //if(i == j) {
      //printf("%3e\n", val);
      //}
      sum += val;
    }
  }
  //normalization_factor = 1.0;
  return normalization_factor * sum / ((double)(n_samples1 * n_samples2));
}

// be sure to pre-scale data appropriately (we suggest ScaleSamplingsToCube())
void KDEGenerativeMMKBatch(double lambda,
			   const ArrayList<Matrix> &samplings,
			   Matrix *p_kernel_matrix) {
  Matrix &kernel_matrix = *p_kernel_matrix;

  int n_samplings = samplings.size();
  
  Vector optimal_bandwidths;
  optimal_bandwidths.Init(n_samplings);
  for(int k = 0; k < n_samplings; k++) {


    /// kill duplicate points
    Matrix references;
    Matrix reference_weights;
    KillDuplicatePoints(samplings[k], &references, &reference_weights);


    // Query and reference datasets, reference weight dataset.
    //const Matrix references;
    //Matrix reference_weights;
    Matrix queries;
    references.PrintDebug("references");

    // data::Load inits a matrix with the contents of a .csv or .arff.
    queries.Alias(references);
    queries.PrintDebug("queries");
  
    // initialize to uniform weights.
    //reference_weights.Init(1, queries.n_cols());
    //reference_weights.SetAll(1);

    printf("sampling %d\n", k);
    optimal_bandwidths[k] =
      BandwidthLSCV::OptimizeReturn<GaussianKernelAux>(references, 
						       reference_weights);
  }

  PrintDebug("optimal_bandwidths", optimal_bandwidths, "%3e");

  kernel_matrix.Init(n_samplings, n_samplings);
  for(int i = 0; i < n_samplings; i++) {
    printf("%f%%\n", 100.0 * ((double)(i + 1)) / ((double)n_samplings));
    for(int j = i; j < n_samplings; j++) {
      //printf("i = %d, j = %d\n", i, j);
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
