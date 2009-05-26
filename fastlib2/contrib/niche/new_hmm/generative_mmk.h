#ifndef GENERATIVE_MMK_H
#define GENERATIVE_MMK_H

#include "multinomial.h"
#include "isotropic_gaussian.h"
#include "hmm.h"

#define INSIDE_GENERATIVE_MMK_IMPL_H

double GenerativeMMK(double lambda,
		     const Multinomial &x,
		     const Multinomial &y);

double GenerativeMMK(double lambda,
		     const IsotropicGaussian &x,
		     const IsotropicGaussian &y);

template <typename TDistribution>
double GenerativeMMK(double lambda, int n_T,
		     const HMM<TDistribution> &hmm_a,
		     const HMM<TDistribution> &hmm_b);

template <typename TDistribution>
void GenerativeMMKBatch(double lambda, int n_T,
			const ArrayList<HMM<TDistribution> > &hmms,
			Matrix *p_kernel_matrix);


#include "generative_mmk_impl.h"
#undef INSIDE_GENERATIVE_MMK_IMPL_H

#endif /* GENERATIVE_MMK_H */
