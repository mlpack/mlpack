#ifndef INSIDE_EMPIRICAL_MMK_IMPL_H
#error "This is not a public header file!"
#endif


void GetCounts(int order,
	       int n_symbols,
	       const GenMatrix<int> &sequence,
	       GenVector<int>* p_counts) {
  GenVector<int> &counts = *p_counts;
  
  int clique_size = order + 1;
  
  counts.Init(pow(n_symbols, clique_size) - 1.0);
  counts.SetZero();
  
  int n_cliques = sequence.n_cols() - order;
  
  int powers[clique_size];
  for(int i = 0; i < clique_size; i++) {
    powers[i] = (int) pow(n_symbols, i);    
  }
  
  for(int t = 0; t < n_cliques; t++) {
    int index = 0;
    for(int i = 0; i < n_symbols; i++) {
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

}

void MarkovEmpiricalMMKBatch(double lambda,
			     int order,
			     int n_symbols,
			     const ArrayList<GenMatrix<int> > &sequences,
			     Matrix *p_kernel_matrix) {



}
