#ifndef EMPIRICAL_MMK_H
#define EMPIRICAL_MMK_H

#define INSIDE_EMPIRICAL_MMK_IMPL_H

void GetCounts(int order,
	       int n_symbols,
	       const GenMatrix<int> &sequence,
	       GenVector<int>* p_counts);

double MarkovEmpiricalMMK(double lambda,
			  int order,
			  int n_symbols,
			  const GenMatrix<int> &sequence1,
			  const GenMatrix<int> &sequence2);

double MarkovEmpiricalMMK(double lambda,
			  int order,
			  const GenMatrix<double> &sequence1,
			  const GenMatrix<double> &sequence2);

void MarkovEmpiricalMMKBatch(double lambda,
			     int order,
			     int n_symbols,
			     const ArrayList<GenMatrix<int> > &sequences,
			     Matrix *p_kernel_matrix);

void MarkovEmpiricalMMKBatch(double lambda,
			     int order,
			     const ArrayList<GenMatrix<double> > &sequences,
			     Matrix *p_kernel_matrix);




#include "empirical_mmk_impl.h"
#undef INSIDE_EMPIRICAL_MMK_IMPL_H

#endif /* EMPIRICAL_MMK_H */
