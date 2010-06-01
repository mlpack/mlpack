#ifndef LATENT_MMK_H
#define LATENT_MMK_H

#include "hmm.h"
#include "hmm_kernel_utils.h"

#define INSIDE_LATENT_MMK_IMPL_H


template <typename TDistribution, typename T>
  double LatentMMK(double lambda,
		      const HMM<TDistribution> &hmm,
		      const GenMatrix<T> &sequence1,
		      const GenMatrix<T> &sequence2);

template <typename T>
  double HMMLatentMMK(double lambda, int n_states, int n_dims,
		      const GenMatrix<T> &sequence1,
		      const GenMatrix<T> &sequence2,
		      const Matrix &p_qt_1, const Matrix &p_qt_2,
		      const Matrix &p_qq_1, const Matrix &p_qq_2);

template <typename TDistribution, typename T>
  void LatentMMKBatch(double lambda,
		      const HMM<TDistribution> &hmm,
		      const ArrayList<GenMatrix<T> > &sequences,
		      Matrix *p_kernel_matrix);

double HMMLatentMMKComponentQX(double exp_neg_lambda,
			       int n_states,
			       int n_dims,
			       const GenMatrix<int> &sequence1,
			       const GenMatrix<int> &sequence2,
			       const Matrix &p_qt_1,
			       const Matrix &p_qt_2);

double HMMLatentMMKComponentQQ(double exp_neg_lambda, int n_states,
			       const Matrix &p_qq_1,
			       const Matrix &p_qq_2,
			       int sequence1_length,
			       int sequence2_length);


#include "latent_mmk_impl.h"
#undef INSIDE_LATENT_MMK_IMPL_H

#endif /* LATENT_MMK_H */
