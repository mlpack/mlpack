#ifndef INSIDE_EMPIRICAL_MMK_IMPL_H
#error "This is not a public header file!"
#endif


void GetCounts(int order,
	       int n_symbols,
	       const GenMatrix<int> &sequence,
	       GenVector<int>* p_counts) {
  GenVector<int> &counts = *p_counts;
  
  int clique_size = order + 1;
  
  counts.Init((int)pow(n_symbols, clique_size));
  counts.SetZero();
  
  int n_cliques = sequence.n_cols() - order;
  
  int powers[clique_size];
  for(int r = 0; r < clique_size; r++) {
    powers[r] = (int) pow(n_symbols, r);    
  }
  
  for(int t = 0; t < n_cliques; t++) {
    int index = 0;
    for(int i = 0; i < clique_size; i++) {
      index += sequence.get(0, t + i) * powers[i];
    }
    counts[index]++;
  }
}

double MarkovEmpiricalMMK(double lambda,
			  int order,
			  int n_symbols,
			  const GenMatrix<int> &sequence1,
			  const GenMatrix<int> &sequence2) {

  GenVector<int> counts1;
  GetCounts(order, n_symbols, sequence1, &counts1);

  GenVector<int> counts2;
  GetCounts(order, n_symbols, sequence2, &counts2);

  int n_strings = counts1.length();

  int clique_size = order + 1;
  int powers[clique_size + 1];
  double exp_neg_lambdas[clique_size];
  for(int r = 0; r < clique_size; r++) {
    powers[r] = (int) pow(n_symbols, r);    
    exp_neg_lambdas[r] = exp(-((double)r) * lambda);
  }
  powers[clique_size] = (int) pow(n_symbols, clique_size);

  int encoding1[clique_size];
  int encoding2[clique_size];



  double sum = 0;
  for(int i = 0; i < n_strings; i++) {
    for(int r = 0; r < clique_size; r++) {
      encoding1[r] = (i % powers[r + 1]) / powers[r];
    }

    for(int j = 0; j < n_strings; j++) {
      for(int r = 0; r < clique_size; r++) {
	encoding2[r] = (j % powers[r + 1]) / powers[r];
      }
      
      int n_different = 0;
      for(int r = 0; r < clique_size; r++) {
	if(encoding1[r] != encoding2[r]) {
	  n_different++;
	}
      }

      double val = 
	((double)(counts1[i] * counts2[j]))
	* exp_neg_lambdas[n_different];
      
      sum += val;
    }
  }

  int n_cliques1 = sequence1.n_cols() - order;
  int n_cliques2 = sequence2.n_cols() - order;

  return sum / ((double)(n_cliques1 * n_cliques2));
}

double MarkovEmpiricalMMK(double lambda,
			  int order,
			  const GenMatrix<double> &sequence1,
			  const GenMatrix<double> &sequence2) {

  int seq1_length = sequence1.n_cols();
  int seq2_length = sequence2.n_cols();

  int n_dims = sequence1.n_rows();

  int n_features_per_clique = n_dims * order;

  double sum = 0;
  for(int i = 0; i < seq1_length - order + 1; i++) {
    for(int j = 0; j < seq2_length - order + 1; j++) {
      // pick i->(i+order-1) columns of sequence1
      // pick j->(j+order-1) columns of sequence2
      //compute squared distance between unrolled vectors (equivalently, subtract submatrices and compute frobenius norm)
      double dist_sq = la::DistanceSqEuclidean(n_features_per_clique, 
					       sequence1.ptr() + (i * n_dims),
					       sequence2.ptr() + (j * n_dims));

      // accumulate exp(-lambda * (distance squared)
      sum += exp(-lambda * dist_sq);
    }
  }

  sum /= (double)((seq1_length - order + 1) * (seq2_length - order + 1));

  return sum;
}

void MarkovEmpiricalMMKBatch(double lambda,
			     int order,
			     int n_symbols,
			     const ArrayList<GenMatrix<int> > &sequences,
			     Matrix *p_kernel_matrix) {
  Matrix &kernel_matrix = *p_kernel_matrix;

  int n_sequences = sequences.size();

  kernel_matrix.Init(n_sequences, n_sequences);
  for(int i = 0; i < n_sequences; i++) {
    printf("%f%%\n", 100.0 * ((double)(i + 1)) / ((double)n_sequences));
    for(int j = i; j < n_sequences; j++) {
      double emmk = 
	MarkovEmpiricalMMK(lambda, order, n_symbols,
			   sequences[i], sequences[j]);
      kernel_matrix.set(j, i, emmk);
      if(i != j) {
	kernel_matrix.set(i, j, emmk);
      }
    }
  }
}

void MarkovEmpiricalMMKBatch(double lambda,
			     int order,
			     const ArrayList<GenMatrix<double> > &sequences,
			     Matrix *p_kernel_matrix) {
  Matrix &kernel_matrix = *p_kernel_matrix;

  int n_sequences = sequences.size();

  kernel_matrix.Init(n_sequences, n_sequences);
  for(int i = 0; i < n_sequences; i++) {
    printf("%f%%\n", 100.0 * ((double)(i + 1)) / ((double)n_sequences));
    for(int j = i; j < n_sequences; j++) {
      double emmk = 
	MarkovEmpiricalMMK(lambda, order,
			   sequences[i], sequences[j]);
      kernel_matrix.set(j, i, emmk);
      if(i != j) {
	kernel_matrix.set(i, j, emmk);
      }
    }
  }
}
