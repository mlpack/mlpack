#ifndef FISHER_KERNEL_H
#define FISHER_KERNEL_H

#include "loghmm.h"
#include "hmm_kernel_utils.h"

#define INSIDE_FISHER_KERNEL_IMPL_H


template <typename TDistribution, typename T>
double FisherKernel(const HMM<TDistribution> &hmm,
		    const GenMatrix<T> &sequence1,
		    const GenMatrix<T> &sequence2);

template <typename TDistribution>
double FisherKernel(const HMM<TDistribution> &hmm,
		    const Matrix &p_qq_1, const Matrix &p_qq_2,
		    const Vector &p_q_1, const Vector &p_q_2,
		    const Matrix &p_qx_1, const Matrix &p_qx_2,
		    const Vector &p_q0_1, const Vector &p_q0_2);

template <typename T>
void FisherKernelBatch(const HMM<Multinomial> &hmm,
		       const ArrayList<GenMatrix<T> > &sequences,
		       Matrix* p_kernel_matrix);

template <typename T>
void FisherKernelBatch(double lambda,
		       const HMM<Multinomial> &hmm,
		       const ArrayList<GenMatrix<T> > &sequences,
		       Matrix* p_kernel_matrix);

template <typename T>
void FisherKernelBatch(double lambda,
		       const HMM<DiagGaussian> &hmm,
		       const ArrayList<GenMatrix<T> > &sequences,
		       Matrix* p_kernel_matrix);



#include "fisher_kernel_impl.h"
#undef INSIDE_FISHER_KERNEL_IMPL_H

#endif /* FISHER_KERNEL_IMPL_H */

