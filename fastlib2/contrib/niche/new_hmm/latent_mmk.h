#ifndef LATENT_MMK_H
#define LATENT_MMK_H

#include "hmm.h"

template <typename TDistribution, T>
double HMMLatentMMK(const HMM<TDistribution> &hmm,
		    const GenMatrix<T> &sequence1,
		    const GenMatrix<T> &sequence2) {

  Matrix p_x_given_q_1; // Rabiner's b
  ArrayList<Matrix> p_qq_t_1; // Rabiner's xi = P(q_t, q_{t+1} | X)
  Matrix p_qt_1; // Rabiner's gamma = P(q_t | X)
  double neg_likelihood = 1;
  hmm.ExpectationStepNoLearning(sequence1,
				&p_x_given_q_1,
				&p_qq_t_1,
				&p_qt_1,
				&neg_likelihood);

  Matrix p_x_given_q_2; // Rabiner's b
  ArrayList<Matrix> p_qq_t_2; // Rabiner's xi = P(q_t, q_{t+1} | X)
  Matrix p_qt_2; // Rabiner's gamma = P(q_t | X)
  double neg_likelihood = 2;
  hmm.ExpectationStepNoLearning(sequence2,
				&p_x_given_q_2,
				&p_qq_t_2,
				&p_qt_2,
				&neg_likelihood);
  
  
  
  return 1;
}


#endif /* LATENT_MMK_H */
