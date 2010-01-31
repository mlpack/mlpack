#ifndef INSIDE_FISHER_KERNEL_IMPL_H
#error "This is not a public header file!"
#endif


template <typename TDistribution, typename T>
double FisherKernel(const HMM<TDistribution> &hmm,
		    const GenMatrix<T> &sequence1,
		    const GenMatrix<T> &sequence2) {

  int n_states = hmm.n_states();
  int n_dims = hmm.n_dims();

  Matrix p_x_given_q_1; // Rabiner's b
  ArrayList<Matrix> p_qq_t_1; // Rabiner's xi = P(q_t, q_{t+1} | X)
  Matrix p_qt_1; // Rabiner's gamma = P(q_t | X)
  double neg_likelihood_1;
  hmm.ExpectationStepNoLearning(sequence1,
				&p_x_given_q_1,
				&p_qq_t_1,
				&p_qt_1,
				&neg_likelihood_1);

  Matrix p_x_given_q_2; // Rabiner's b
  ArrayList<Matrix> p_qq_t_2; // Rabiner's xi = P(q_t, q_{t+1} | X)
  Matrix p_qt_2; // Rabiner's gamma = P(q_t | X)
  double neg_likelihood_2;
  hmm.ExpectationStepNoLearning(sequence2,
				&p_x_given_q_2,
				&p_qq_t_2,
				&p_qt_2,
				&neg_likelihood_2);

  Matrix p_qq_1;
  Matrix p_qq_2;
  ComputePqq(p_qq_t_1, &p_qq_1);
  ComputePqq(p_qq_t_2, &p_qq_2);

  Vector p_q_1;
  Vector p_q_2;
  ComputePq(p_qt_1, &p_q_1);
  ComputePq(p_qt_2, &p_q_2);

  Matrix p_qx_1;
  Matrix p_qx_2;

  ComputePqx(sequence1, n_dims, p_qt_1, &p_qx_1);
  ComputePqx(sequence2, n_dims, p_qt_2, &p_qx_2);

  // copy p_transition and transpose its copy for loop efficiency
  Matrix p_transition;
  p_transition.Copy(hmm.p_transition);
  la::TransposeSquare(&p_transition);

  // start accumulating
  double sum = 0;

  // initial components
  for(int i = 0; i < n_states; i++) {
    if(hmm.p_initial[i] > 1e-100) {
      double weight = 1 / hmm.p_initial[i];
      sum += 
	(weight * p_qt_1.get(0, i) - p_q_1[i])
	*
	(weight * p_qt_2.get(0, i) - p_q_2[i]);
    }
  }

  // transition components
  for(int i = 0; i < n_states; i++) {
    for(int j = 0; j < n_states; j++) {
      if(p_transition.get(j, i) > 1e-100) {
	double weight = 1 / p_transition.get(j, i);
	sum += 
	  (weight * p_qq_1.get(j, i) - p_q_1[j])
	  *
	  (weight * p_qq_2.get(j, i) - p_q_2[j]);
      }
    }
  }

  // observation components
  for(int i = 0; i < n_states; i++) {
    for(int a = 0; a < n_dims; a++) {
      if(hmm.state_distributions[i].p()[a] > 1e-100) {
	double weight = 1 / hmm.state_distributions[i].p()[a];
	double subsum1 = p_qx_1.get(a, i);
/* 	for(int t = 0; t < sequence1_length; t++) { */
/* 	  subsum1 += p_qt_1.get(t, i) * (sequence1.get(0, t) == a); */
/* 	} */
	subsum1 = weight * subsum1 - p_q_1[i];

	double subsum2 = p_qx_2.get(a, i);
/* 	for(int t = 0; t < sequence2_length; t++) { */
/* 	  subsum2 += p_qt_2.get(t, i) * (sequence2.get(0, t) == a); */
/* 	} */
	subsum2 = weight * subsum2 - p_q_2[i];

	sum += subsum1 * subsum2;
      }
    }
  }

  return sum;
}

template <typename TDistribution>
double FisherKernel(const HMM<TDistribution> &hmm,
		  const Matrix &p_qq_1, const Matrix &p_qq_2,
		  const Vector &p_q_1, const Vector &p_q_2,
		  const Matrix &p_qx_1, const Matrix &p_qx_2,
		  const Vector &p_q0_1, const Vector &p_q0_2) {

  int n_states = hmm.n_states();
  int n_dims = hmm.n_dims();

  // copy p_transition and transpose its copy for loop efficiency
  Matrix p_transition;
  p_transition.Copy(hmm.p_transition);
  la::TransposeSquare(&p_transition);
  
  // start accumulating
  double sum = 0;

  // initial components
  for(int i = 0; i < n_states; i++) {
    if(hmm.p_initial[i] > 1e-100) {
      double weight = 1 / hmm.p_initial[i];
      sum += 
	(weight * p_q0_1[i] - p_q_1[i])
	*
	(weight * p_q0_2[i] - p_q_2[i]);
    }
  }

  // transition components
  for(int i = 0; i < n_states; i++) {
    for(int j = 0; j < n_states; j++) {
      if(p_transition.get(j, i) > 1e-100) {
	double weight = 1 / p_transition.get(j, i);
	sum += 
	  (weight * p_qq_1.get(j, i) - p_q_1[j])
	  *
	  (weight * p_qq_2.get(j, i) - p_q_2[j]);
      }
    }
  }

  // observation components
  for(int i = 0; i < n_states; i++) {
    for(int a = 0; a < n_dims; a++) {
      if(hmm.state_distributions[i].p()[a] > 1e-100) {
	double weight = 1 / hmm.state_distributions[i].p()[a];
	double subsum1 = p_qx_1.get(a, i);
	subsum1 = weight * subsum1 - p_q_1[i];

	double subsum2 = p_qx_2.get(a, i);
	subsum2 = weight * subsum2 - p_q_2[i];

	sum += subsum1 * subsum2;
      }
    }
  }

  return sum;
}

template <typename T>
void FisherKernelBatch(const HMM<Multinomial> &hmm,
		       const ArrayList<GenMatrix<T> > &sequences,
		       Matrix* p_kernel_matrix) {
  Matrix &kernel_matrix = *p_kernel_matrix;


  int n_states = hmm.n_states();
  int n_dims = hmm.n_dims();
  
  // we don't care about these so we reuse them
  Matrix p_x_given_q;
  p_x_given_q.Init(0, 0);
  ArrayList<Matrix> p_qq_t;
  p_qq_t.Init(0);
  Matrix p_qt;
  p_qt.Init(0, 0);
  double neg_likelihood;

  ArrayList<Matrix> p_qq_arraylist;
  ArrayList<Vector> p_q_arraylist;
  ArrayList<Matrix> p_qx_arraylist;
  ArrayList<Vector> p_q0_arraylist;


  int n_sequences = sequences.size();

  p_qq_arraylist.Init(n_sequences);
  p_q_arraylist.Init(n_sequences);
  p_qx_arraylist.Init(n_sequences);
  p_q0_arraylist.Init(n_sequences);

  for(int i = 0; i < n_sequences; i++) {
    p_x_given_q.Destruct();
    p_qq_t.Renew();
    p_qt.Destruct();

    hmm.ExpectationStepNoLearning(sequences[i],
				  &p_x_given_q,
				  &p_qq_t,
				  &p_qt,
				  &neg_likelihood);

    ComputePqq(p_qq_t, &(p_qq_arraylist[i]));
    ComputePq(p_qt, &(p_q_arraylist[i]));
    ComputePqx(sequences[i], n_dims, p_qt, &(p_qx_arraylist[i]));
    p_q0_arraylist[i].Init(n_states);
    for(int j = 0; j < n_states; j++) {
      p_q0_arraylist[i][j] = p_qt.get(0, j);
    }
  }
  
  kernel_matrix.Init(n_sequences, n_sequences);
  for(int i = 0; i < n_sequences; i++) {
    printf("%f%%\n", 100.0 * ((double)(i + 1)) / ((double)n_sequences));
    for(int j = i; j < n_sequences; j++) {
      double fk =
	FisherKernel(hmm,
		     p_qq_arraylist[i], p_qq_arraylist[j],
		     p_q_arraylist[i], p_q_arraylist[j],
		     p_qx_arraylist[i], p_qx_arraylist[j],
		     p_q0_arraylist[i], p_q0_arraylist[j]);
      
      kernel_matrix.set(j, i, fk);
      if(i != j) {
	kernel_matrix.set(i, j, fk);
      }
    }
  }
}

template <typename T>
void FisherKernelBatch(double lambda,
		       const HMM<Multinomial> &hmm,
		       const ArrayList<GenMatrix<T> > &sequences,
		       Matrix* p_kernel_matrix) {
  if(lambda < 0) {
    FisherKernelBatch(hmm, sequences, p_kernel_matrix);
    return;
  }

  Matrix &kernel_matrix = *p_kernel_matrix;


  int n_states = hmm.n_states();
  int n_dims = hmm.n_dims();
  
  // we don't care about these so we reuse them
  Matrix p_x_given_q;
  p_x_given_q.Init(0, 0);
  ArrayList<Matrix> p_qq_t;
  p_qq_t.Init(0);
  Matrix p_qt;
  p_qt.Init(0, 0);
  double neg_likelihood;

  Matrix p_qq;
  p_qq.Init(0, 0);
  Vector p_q;
  p_q.Init(0);
  Matrix p_qx;
  p_qx.Init(0, 0);
  Vector p_q0;
  p_q0.Init(0);


  int n_sequences = sequences.size();

  ArrayList<Vector> u_arraylist;
  u_arraylist.Init(n_sequences);
  int n_score_components =
    n_states + (n_states * n_states) + (n_states * n_dims);
  for(int k = 0; k < n_sequences; k++) {
    u_arraylist[k].Init(n_score_components);
  }

  // copy p_transition and transpose its copy for loop efficiency
  Matrix p_transition;
  p_transition.Copy(hmm.p_transition);
  la::TransposeSquare(&p_transition);


  for(int k = 0; k < n_sequences; k++) {
    p_x_given_q.Destruct();
    p_qq_t.Renew();
    p_qt.Destruct();
    
    p_qq.Destruct();
    p_q.Destruct();
    p_qx.Destruct();
    p_q0.Destruct();

    hmm.ExpectationStepNoLearning(sequences[k],
				  &p_x_given_q,
				  &p_qq_t,
				  &p_qt,
				  &neg_likelihood);

    ComputePqq(p_qq_t, &p_qq);
    ComputePq(p_qt, &p_q);
    ComputePqx(sequences[k], n_dims, p_qt, &p_qx);
    p_q0.Init(n_states);
    for(int j = 0; j < n_states; j++) {
      p_q0[j] = p_qt.get(0, j);
    }
    
    
    ////////////////////////////
    
    // construct Fisher score vectors
    int u_index = 0;
    
    // initial components
    for(int i = 0; i < n_states; i++) {
      if(hmm.p_initial[i] > 1e-100) {
	double weight = 1 / hmm.p_initial[i];
	u_arraylist[k][u_index] = weight * p_q0[k] - p_q[k];
      }
      else {
	u_arraylist[k][u_index] = 0;
      }
      u_index++;
    }
    
    // transition components
    for(int i = 0; i < n_states; i++) {
      for(int j = 0; j < n_states; j++) {
	if(p_transition.get(j, i) > 1e-100) {
	  double weight = 1 / p_transition.get(j, i);
	  u_arraylist[k][u_index] = weight * p_qq.get(j, i) - p_q[j];
	}
	else {
	  u_arraylist[k][u_index] = 0;
	}
	u_index++;
      }
    }

    // observation components
    for(int i = 0; i < n_states; i++) {
      for(int a = 0; a < n_dims; a++) {
	if(hmm.state_distributions[i].p()[a] > 1e-100) {
	  double weight = 1 / hmm.state_distributions[i].p()[a];
	  u_arraylist[k][u_index] = weight * p_qx.get(a, i) - p_q[i];
	}
	else {
	  u_arraylist[k][u_index] = 0;
	}
	u_index++;
      }
    }

    ///////////////////////////
  }
  
  kernel_matrix.Init(n_sequences, n_sequences);
  for(int i = 0; i < n_sequences; i++) {
    printf("%f%%\n", 100.0 * ((double)(i + 1)) / ((double)n_sequences));
    for(int j = i; j < n_sequences; j++) {
      double fk =
	exp(-lambda * la::DistanceSqEuclidean(u_arraylist[i], u_arraylist[j]));
      kernel_matrix.set(j, i, fk);
      if(i != j) {
	kernel_matrix.set(i, j, fk);
      }
    }
  }
}





// todo: testing
template <typename T>
void FisherKernelBatch(double lambda,
		       const HMM<DiagGaussian> &hmm,
		       const ArrayList<GenMatrix<T> > &sequences,
		       Matrix* p_kernel_matrix) {

  Matrix &kernel_matrix = *p_kernel_matrix;


  int n_states = hmm.n_states();
  int n_dims = hmm.n_dims();
  
  // we don't care about these so we reuse them
  Matrix p_x_given_q;
  p_x_given_q.Init(0, 0);
  ArrayList<Matrix> p_qq_t;
  p_qq_t.Init(0);
  Matrix p_qt;
  p_qt.Init(0, 0);
  double neg_likelihood;

  Matrix p_qq;
  p_qq.Init(0, 0);
  Vector p_q;
  p_q.Init(0);
  //Matrix p_qx;
  //p_qx.Init(0, 0);
  Vector p_q0;
  p_q0.Init(0);


  int n_sequences = sequences.size();

  ArrayList<Vector> u_arraylist;
  u_arraylist.Init(n_sequences);
  int n_score_components =
    n_states + (n_states * n_states) + (2 * n_states * n_dims);
  for(int k = 0; k < n_sequences; k++) {
    u_arraylist[k].Init(n_score_components);
  }

  /*
  // copy p_transition and transpose its copy for loop efficiency
  Matrix p_transition;
  p_transition.Copy(hmm.p_transition);
  la::TransposeSquare(&p_transition);
  */


  Vector sum_mu; // reused in loop below
  sum_mu.Init(n_dims);
  
  Vector sum_sigma; // reused in loop below
  sum_sigma.Init(n_dims);

  Vector p_initial;
  p_initial.Init(n_states);

  Matrix p_transition;
  p_transition.Init(n_states, n_states);

  for(int i = 0; i < n_states; i++) {
    p_initial[i] = exp(hmm.p_initial[i]);
    for(int j = 0; j < n_states; j++) {
      // transpose for loop efficiency
      p_transition.set(j, i, exp(hmm.p_transition.get(i, j)));
    }
  }



  for(int k = 0; k < n_sequences; k++) {
    p_x_given_q.Destruct();
    p_qq_t.Renew();
    p_qt.Destruct();
    
    p_qq.Destruct();
    p_q.Destruct();
    //p_qx.Destruct();
    p_q0.Destruct();

    hmm.ExpectationStepNoLearning(sequences[k],
				  &p_x_given_q,
				  &p_qq_t,
				  &p_qt,
				  &neg_likelihood);

    int seq_length = sequences[k].n_cols();

    // go from log space back to normal space
    for(int i = 0; i < n_states; i++) {
      for(int t = 0; t < seq_length; t++) {
	p_qt.set(t, i, exp(p_qt.get(t, i)));
      }
    }

    // go from log space back to normal space
    for(int i = 0; i < n_states; i++) {
      for(int t = 0; t < seq_length; t++) {
	for(int j = 0; j < n_states; j++) {
	  p_qq_t[i].set(j, t, exp(p_qq_t[i].get(j, t)));
	}
      }
    }

    ComputePqq(p_qq_t, &p_qq);
    ComputePq(p_qt, &p_q);
    //ComputePqx(sequences[k], n_dims, p_qt, &p_qx);
    p_q0.Init(n_states);
    for(int j = 0; j < n_states; j++) {
      p_q0[j] = p_qt.get(0, j);
    }
    
    
    ////////////////////////////
    
    // construct Fisher score vectors
    int u_index = 0;


    
    // initial components
    for(int i = 0; i < n_states; i++) {
      if(p_initial[i] > 1e-100) {
	double weight = 1 / p_initial[i];
	u_arraylist[k][u_index] = weight * p_q0[k] - p_q[k];
      }
      else {
	u_arraylist[k][u_index] = 0;
      }
      u_index++;
    }
    
    // transition components
    for(int i = 0; i < n_states; i++) {
      for(int j = 0; j < n_states; j++) {
	if(p_transition.get(j, i) > 1e-100) {
	  double weight = 1 / p_transition.get(j, i);
	  u_arraylist[k][u_index] = weight * p_qq.get(j, i) - p_q[j];
	}
	else {
	  u_arraylist[k][u_index] = 0;
	}
	u_index++;
      }
    }

    // observation components - mean and diagonal covariance
    for(int i = 0; i < n_states; i++) {
      sum_mu.SetZero();
      sum_sigma.SetZero();
      double sum_gamma = 0;
      for(int t = 0; t < seq_length; t++) {
	sum_gamma += p_qt.get(t, i);
	for(int d = 0; d < n_dims; d++) {
	  double difference =
	    sequences[k].get(d, t) - hmm.state_distributions[i].mu()[d];
	  sum_mu[d] +=
	    p_qt.get(t, i) * difference;
	  sum_sigma[d] +=
	    p_qt.get(t, i) * difference * difference;
	}
      }
      for(int d = 0; d < n_dims; d++) {
	double sig = hmm.state_distributions[i].sigma()[d];
	u_arraylist[k][u_index] =
	  sum_mu[d] / sig;
	u_index++;
	u_arraylist[k][u_index] =
	  (sum_sigma[d] / pow(sig, 1.5)) + (sum_gamma / sqrt(sig));
	u_index++;
      }
    }

    ///////////////////////////
  }

  double max_norm = -1;
  if(lambda > 0) { // Gaussianized Fisher kernel
    for(int i = 0; i < n_sequences; i++) {
      double cur_norm = la::LengthEuclidean(u_arraylist[i]);
      if(cur_norm > max_norm) {
	max_norm = cur_norm;
      }
    }
  }
 
  
  kernel_matrix.Init(n_sequences, n_sequences);
  for(int i = 0; i < n_sequences; i++) {
    printf("%f%%\n", 100.0 * ((double)(i + 1)) / ((double)n_sequences));
    for(int j = i; j < n_sequences; j++) {
      double fk;
      if(lambda < 0) { // standard Fisher kernel
	fk = la::Dot(u_arraylist[i], u_arraylist[j]);
      }
      else { // Gaussianized Fisher kernel
	fk = exp(-lambda * la::DistanceSqEuclidean(u_arraylist[i], u_arraylist[j]) / max_norm);
      }
      kernel_matrix.set(j, i, fk);
      if(i != j) {
	kernel_matrix.set(i, j, fk);
      }
    }
  }

  
  if(lambda < 0) { // standard Fisher kernel
    // scale the kernel matrix when using the standard plain Fisher kernel
    double max_val = -1;
    for(int i = 0; i < n_sequences; i++) {
      if(unlikely(kernel_matrix.get(i, i) > max_val)) {
	max_val = kernel_matrix.get(i, i);
      }
    }
    la::Scale(1 / (((double)n_sequences) * max_val), &kernel_matrix);
  }
   

}
