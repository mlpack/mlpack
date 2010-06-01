#ifndef EMPIRICAL_MMK_H
#define EMPIRICAL_MMK_H

#define INSIDE_EMPIRICAL_MMK_IMPL_H

double MarkovEmpiricalMMK(double lambda,
			  int order,
			  const GenMatrix<int> &sequence1,
			  const GenMatrix<int> &sequence2);

void MarkovEmpiricalMMKBatch(double lambda,
			     int order,
			     const ArrayList<GenMatrix<int> > &sequences,
			     Matrix *p_kernel_matrix);


#include "empirical_mmk_impl.h"
#undef INSIDE_EMPIRICAL_MMK_IMPL_H

#endif /* EMPIRICAL_MMK_H */
