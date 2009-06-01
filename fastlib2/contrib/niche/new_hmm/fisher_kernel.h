#ifndef FISHER_KERNEL_H
#define FISHER_KERNEL_H

#include "hmm.h"
#include "latent_mmk.h"

#define INSIDE_FISHER_KERNEL_IMPL_H

template <typename TDistribution, typename T>
double FisherKernel(const HMM<TDistribution> &hmm,
		    const GenMatrix<T> &sequence1,
		    const GenMatrix<T> &sequence2);

template <typename TDistribution, typename T>
void FisherKernelBatch(const HMM<TDistribution> &hmm,
			 const ArrayList<GenMatrix<T> > &sequences,
			 Matrix* p_kernel_matrix);


void ComputePq(const Matrix &p_qt,
	       Vector* p_p_q);



		    





#include "fisher_kernel_impl.h"
#undef INSIDE_FISHER_KERNEL_IMPL_H

#endif /* FISHER_KERNEL_IMPL_H */

