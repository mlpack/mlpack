#ifndef INSIDE_LATENT_MMK_IMPL_H
#error "This is not a public header file!"
#endif

  
template <typename TDistribution, typename T>
  double LatentMMK(double lambda,
		   const HMM<TDistribution> &hmm,
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

  Matrix p_qx_1;
  Matrix p_qx_2;
  
  ComputePqx(sequence1, n_dims, p_qt_1, &p_qx_1);
  ComputePqx(sequence2, n_dims, p_qt_2, &p_qx_2);

  //p_qq_1.PrintDebug("p_qq_1");
  //p_qq_2.PrintDebug("p_qq_2");

  // everything is ready for call to other function
  return HMMLatentMMK(lambda, n_states, n_dims,
		      sequence1, sequence2,
		      p_qx_1, p_qx_2,
		      p_qq_1, p_qq_2);
}

template <typename T>
double HMMLatentMMK(double lambda, int n_states, int n_dims,
		    const GenMatrix<T> &sequence1,
		    const GenMatrix<T> &sequence2,
		    const Matrix &p_qx_1, const Matrix &p_qx_2,
		    const Matrix &p_qq_1, const Matrix &p_qq_2) {
  double exp_neg_lambda = exp(-lambda);
  
  double p_qt_component = 
    HMMLatentMMKComponentQX(exp_neg_lambda, n_states, n_dims,
			    sequence1, sequence2,
			    p_qx_1, p_qx_2);

  double p_qq_t_component =
    HMMLatentMMKComponentQQ(exp_neg_lambda, n_states,
			    p_qq_1, p_qq_2,
			    sequence1.n_cols(), sequence2.n_cols());

  //printf("p_qt_component = %f\np_qq_t_component = %f\n",
  // p_qt_component, p_qq_t_component);
  
  return p_qt_component + p_qq_t_component;
}		      


template <typename TDistribution, typename T>
  void LatentMMKBatch(double lambda,
		      const HMM<TDistribution> &hmm,
		      const ArrayList<GenMatrix<T> > &sequences,
		      Matrix *p_kernel_matrix) {
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

  //ArrayList<Matrix> p_qt_arraylist;
  ArrayList<Matrix> p_qq_arraylist;
  ArrayList<Matrix> p_qx_arraylist;

  int n_sequences = sequences.size();
  
  //p_qt_arraylist.Init(n_sequences);
  p_qq_arraylist.Init(n_sequences);
  p_qx_arraylist.Init(n_sequences);
  
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
    ComputePqx(sequences[i], n_dims, p_qt, &(p_qx_arraylist[i]));
  }

  kernel_matrix.Init(n_sequences, n_sequences);
  for(int i = 0; i < n_sequences; i++) {
    printf("%f%%\n", 100.0 * ((double)(i + 1)) / ((double)n_sequences));
    for(int j = i; j < n_sequences; j++) {
      double lmmk = 
	HMMLatentMMK(lambda, n_states, n_dims,
		     sequences[i], sequences[j],
		     p_qx_arraylist[i], p_qx_arraylist[j],
		     p_qq_arraylist[i], p_qq_arraylist[j]);

      kernel_matrix.set(j, i, lmmk);
      if(i != j) {
	kernel_matrix.set(i, j, lmmk);
      }
    }
  }
}

double HMMLatentMMKComponentQX(double exp_neg_lambda,
			       int n_states,
			       int n_dims,
			       const GenMatrix<int> &sequence1,
			       const GenMatrix<int> &sequence2,
			       const Matrix &p_qx_1,
			       const Matrix &p_qx_2) {
  int sequence1_length = sequence1.n_cols();
  int sequence2_length = sequence2.n_cols();

  // p_qt's are (sequence index) by state
  //   as in p_qt.get(t, state)

  double sum = 0;
  for(int i = 0; i < n_states; i++) {
    for(int a = 0; a < n_dims; a++) {
      double sum_p_qt_given_xs_equals_a = p_qx_1.get(a, i);
      
      // 1
      double sum1 = p_qx_2.get(a, i);
      
      // 2
      double sum2 = 0;
      for(int j = 0; j < n_states; j++) {
	if(j == i) {
	  continue;
	}
	sum2 += p_qx_2.get(a, j);
      }
      sum2 *= exp_neg_lambda;
      
      // 3
      double sum3 = 0;
      for(int b = 0; b < n_dims; b++) {
	if(b == a) {
	  continue;
	}
	
	double subsum3_1 = p_qx_2.get(b, i);
	
	double subsum3_2 = 0;
	for(int j = 0; j < n_states; j++) {
	  if(j == i) {
	    continue;
	  }
	  subsum3_2 += p_qx_2.get(b, j);
	}
	subsum3_2 *= exp_neg_lambda;
	sum3 += subsum3_1 + subsum3_2;
      }
      sum3 *= exp_neg_lambda;
	
      sum += sum_p_qt_given_xs_equals_a * (sum1 + sum2 + sum3);
    }
  }

  return sum / ((double)(sequence1_length * sequence2_length));
}

double HMMLatentMMKComponentQQ(double exp_neg_lambda, int n_states,
			       const Matrix &p_qq_1,
			       const Matrix &p_qq_2,
			       int sequence1_length,
			       int sequence2_length) {
  double sum = 0;
  
  // p_qq_t's are (to state) by (sequence index) by (from state)
  //   as in p_qq_t[from_state].get(to_state, t)
  
  for(int i = 0; i < n_states; i++) {
    for(int j = 0; j < n_states; j++) {
      double sum1 = p_qq_2.get(j, i);
      
      double sum2 = 0;
      for(int jp = 0; jp < n_states; jp++) {
	if(jp == j) {
	  continue;
	}
	sum2 += p_qq_2.get(jp, i);
      }
      sum2 *= exp_neg_lambda;
      
      double sum3 = 0;
      for(int ip = 0; ip < n_states; ip++) {
	if(ip == i) {
	  continue;
	}
	double subsum3_1 = p_qq_2.get(j, ip);
	
	double subsum3_2 = 0;
	for(int jp = 0; jp < n_states; jp++) {
	  if(jp == j) {
	    continue;
	  }
	  subsum3_2 += p_qq_2.get(jp, ip);
	}
	subsum3_2 *= exp_neg_lambda;
	
	sum3 += subsum3_1 + subsum3_2;
      }
      sum3 *= exp_neg_lambda;
      
      sum += p_qq_1.get(j, i) * (sum1 + sum2 + sum3);
    }
  }
  
  return sum / ((double)((sequence1_length - 1) * (sequence2_length - 1)));
}
