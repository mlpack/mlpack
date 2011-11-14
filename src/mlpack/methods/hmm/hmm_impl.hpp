/**
 * @file hmm_impl.hpp
 * @author Ryan Curtin
 * @author Tran Quoc Long
 *
 * Implementation of HMM class.
 */
#ifndef __MLPACK_METHODS_HMM_HMM_IMPL_HPP
#define __MLPACK_METHODS_HMM_HMM_IMPL_HPP

// Just in case...
#include "hmm.hpp"

namespace mlpack {
namespace hmm {

/**
 * Create the Hidden Markov Model with the given number of hidden states and the
 * given number of emission states.
 */
template<typename Distribution>
HMM<Distribution>::HMM(const size_t states, const size_t emissions) :
    transition(arma::ones<arma::mat>(states, states) / (double) states),
    emission(arma::ones<arma::mat>(emissions, states) / (double) emissions)
{ /* nothing to do */ }

/**
 * Create the Hidden Markov Model with the given transition matrix and the given
 * emission probability matrix.
 */
template<typename Distribution>
HMM<Distribution>::HMM(const arma::mat& transition, const arma::mat& emission) :
    transition(transition),
    emission(emission)
{ /* nothing to do */ }

/**
 * Train the model using the Baum-Welch algorith, with only the given unlabeled
 * observations.
 */
template<typename Distribution>
void HMM<Distribution>::Train(const std::vector<arma::vec>& dataSeq)
{
  // We should allow a guess at the transition and emission matrices.
  double loglik = 0;
  double oldLoglik = 0;

  // Maximum iterations?
  size_t iterations = 1000;

  // This should be the Baum-Welch algorithm (EM for HMM estimation). This
  // follows the procedure outlined in Elliot, Aggoun, and Moore's book "Hidden
  // Markov Models: Estimation and Control", pp. 36-40.
  for (size_t iter = 0; iter < iterations; iter++)
  {
    // Clear new transition and emission matrices.
    arma::mat newTransition(transition.n_rows, transition.n_cols);
    arma::mat newEmission(emission.n_rows, emission.n_cols);
    newTransition.zeros();
    newEmission.zeros();

    // Reset log likelihood.
    loglik = 0;

    // Loop over each sequence.
    for (size_t seq = 0; seq < dataSeq.size(); seq++)
    {
      arma::mat stateProb;
      arma::mat forward;
      arma::mat backward;
      arma::vec scales;

      // Add the log-likelihood of this sequence.  This is the E-step.
      loglik += Estimate(dataSeq[seq], stateProb, forward, backward, scales);

      // Now re-estimate the parameters.  This is the M-step.
      //   T_ij = sum_d ((1 / P(seq[d])) sum_t (f(i, t) T_ij E_i(seq[d][t]) b(i,
      //           t + 1)))
      //   E_ij = sum_d ((1 / P(seq[d])) sum_{t | seq[d][t] = j} f(i, t) b(i, t)
      // We store the new estimates in a different matrix.
      for (size_t t = 0; t < dataSeq[seq].n_elem; t++)
      {
        for (size_t j = 0; j < transition.n_cols; j++)
        {
          if (t < dataSeq[seq].n_elem - 1)
          {
            // Estimate of T_ij (probability of transition from state j to state
            // i).  We postpone multiplication of the old T_ij until later.
            for (size_t i = 0; i < transition.n_rows; i++)
              newTransition(i, j) += forward(j, t) * backward(i, t + 1) *
                  emission((size_t) dataSeq[seq][t + 1], i) / scales[t + 1];
          }

          // Estimate of E_ij (probability of emission i while in state j).
          newEmission((size_t) dataSeq[seq][t], j) += stateProb(j, t);
        }
      }
    }

    // Assign the new matrices.  We use %= (element-wise multiplication) because
    // every element of the new transition matrix must still be multiplied by
    // the old elements (this is the multiplication we earlier postponed).
    transition %= newTransition;
    emission = newEmission;

    // Now we normalize the transition matrices.
    for (size_t i = 0; i < transition.n_cols; i++)
      transition.col(i) /= accu(transition.col(i));

    for (size_t i = 0; i < emission.n_cols; i++)
      emission.col(i) /= accu(emission.col(i));

    Log::Debug << "Iteration " << iter << ": log-likelihood " << loglik
        << std::endl;

    if (fabs(oldLoglik - loglik) < 1e-5)
    {
      Log::Debug << "Converged after " << iter << " iterations." << std::endl;
      break;
    }

    oldLoglik = loglik;
  }
}

/**
 * Train the model using the given labeled observations; the transition and
 * emission matrices are directly estimated.
 */
template<typename Distribution>
void HMM<Distribution>::Train(const std::vector<arma::vec>& dataSeq,
                              const std::vector<arma::Col<size_t> >& stateSeq)
{
  // Simple error checking.
  if (dataSeq.size() != stateSeq.size())
    Log::Fatal << "HMM::Train(): number of data sequences not equal to number "
        "of state sequences." << std::endl;

  transition.zeros();
  emission.zeros();

  // Estimate the transition and emission matrices directly from the
  // observations.
  for (size_t seq = 0; seq < dataSeq.size(); seq++)
  {
    // Simple error checking.
    if (dataSeq[seq].n_elem != stateSeq[seq].n_elem)
      Log::Fatal << "HMM::Train(): number of observations in sequence " << seq
          << " not equal to number of states" << std::endl;

    // Loop over each observation in the sequence.  For estimation of the
    // transition matrix, we must ignore the last observation.
    for (size_t t = 0; t < dataSeq[seq].n_elem - 1; t++)
    {
      transition(stateSeq[seq][t + 1], stateSeq[seq][t])++;
      emission(dataSeq[seq][t], stateSeq[seq][t])++;
    }

    // Last observation.
    emission(dataSeq[seq][dataSeq[seq].n_elem - 1],
        stateSeq[seq][stateSeq[seq].n_elem - 1])++;
  }

  // Normalize transition matrix and emission matrix..
  for (size_t col = 0; col < transition.n_cols; col++)
  {
    // If the transition probability sum is greater than 0 in this column, the
    // emission probability sum will also be greater than 0.  We want to avoid
    // division by 0.
    double sum = accu(transition.col(col));
    if (sum > 0)
    {
      transition.col(col) /= sum;
      emission.col(col) /= accu(emission.col(col));
    }
  }
}

/**
 * Estimate the probabilities of each hidden state at each time step for each
 * given data observation.
 */
template<typename Distribution>
double HMM<Distribution>::Estimate(const arma::vec& data_seq,
                                   arma::mat& state_prob_mat,
                                   arma::mat& forward_prob_mat,
                                   arma::mat& backward_prob_mat,
                                   arma::vec& scale_vec) const
{
  // First run the forward-backward algorithm.
  Forward(data_seq, scale_vec, forward_prob_mat);
  Backward(data_seq, scale_vec, backward_prob_mat);

  // Now assemble the state probability matrix based on the forward and backward
  // probabilities.
  state_prob_mat = forward_prob_mat % backward_prob_mat;

  // Finally assemble the log-likelihood and return it.
  return accu(log(scale_vec));
}

/**
 * Generate a random data sequence of a given length.  The data sequence is
 * stored in the dataSequence parameter, and the state sequence is stored in
 * the stateSequence parameter.
 */
template<typename Distribution>
void HMM<Distribution>::Generate(const size_t length,
                                 arma::vec& dataSequence,
                                 arma::Col<size_t>& stateSequence,
                                 const size_t startState) const
{
  // Set vectors to the right size.
  stateSequence.set_size(length);
  dataSequence.set_size(length);

  // Set start state (default is 0).
  stateSequence[0] = startState;

  // Choose first emission state.
  double randValue = (double) rand() / (double) RAND_MAX;

  // We just have to find where our random value sits in the probability
  // distribution of emissions for our starting state.
  double probSum = 0;
  for (size_t em = 0; em < emission.n_rows; em++)
  {
    probSum += emission(em, startState);
    if (randValue <= probSum)
    {
      dataSequence[0] = em;
      break;
    }
  }

  // Now choose the states and emissions for the rest of the sequence.
  for (size_t t = 1; t < length; t++)
  {
    // First choose the hidden state.
    randValue = (double) rand() / (double) RAND_MAX;

    // Now find where our random value sits in the probability distribution of
    // state changes.
    probSum = 0;
    for (size_t st = 0; st < transition.n_rows; st++)
    {
      probSum += transition(st, stateSequence[t - 1]);
      if (randValue <= probSum)
      {
        stateSequence[t] = st;
        break;
      }
    }

    // Now choose the emission.
    randValue = (double) rand() / (double) RAND_MAX;

    // Now find where our random value sits in the probability distribution of
    // emissions for the state we just chose.
    probSum = 0;
    for (size_t em = 0; em < emission.n_rows; em++)
    {
      probSum += emission(em, stateSequence[t]);
      if (randValue <= probSum)
      {
        dataSequence[t] = em;
        break;
      }
    }
  }
}

/**
 * Compute the most probable hidden state sequence for the given observation
 * using the Viterbi algorithm. Returns the log-likelihood of the most likely
 * sequence.
 */
template<typename Distribution>
double HMM<Distribution>::Predict(const arma::vec& dataSeq,
                                  arma::Col<size_t>& stateSeq) const
{
  // This is an implementation of the Viterbi algorithm for finding the most
  // probable sequence of states to produce the observed data sequence.  We
  // don't use log-likelihoods to save that little bit of time, but we'll
  // calculate the log-likelihood at the end of it all.
  stateSeq.set_size(dataSeq.n_elem);
  arma::mat stateProb(transition.n_rows, dataSeq.n_elem);

  // The calculation of the first state is slightly different; the probability
  // of the first state being state j is the maximum probability that the state
  // came to be j from another state.  We can do that in one line.
  stateProb.col(0) = transition.col(0) %
      trans(emission.row((size_t) dataSeq[0]));

  // Store the best first state.
  arma::u32 index;
  stateProb.unsafe_col(0).max(index);
  stateSeq[0] = index;

  for (size_t t = 1; t < dataSeq.n_elem; t++)
  {
    // Assemble the state probability for this element.
    // Given that we are in state j, we state with the highest probability of
    // being the previous state.
    for (size_t j = 0; j < transition.n_rows; j++)
    {
      arma::vec prob = stateProb.col(t - 1) % trans(transition.row(j));
      stateProb(j, t) = prob.max() * emission((size_t) dataSeq[t], j);
    }

    // Store the best state.
    stateProb.unsafe_col(t).max(index);
    stateSeq[t] = index;
  }

  return log(stateProb(stateSeq[dataSeq.n_elem - 1], dataSeq.n_elem - 1));
}

/**
 * Compute the log-likelihood of the given data sequence.
 */
template<typename Distribution>
double HMM<Distribution>::LogLikelihood(const arma::vec& dataSeq) const
{
  arma::mat forward;
  arma::vec scales;

  Forward(dataSeq, scales, forward);

  // The log-likelihood is the log of the scales for each time step.
  return accu(log(scales));
}

/**
 * The Forward procedure (part of the Forward-Backward algorithm).
 */
template<typename Distribution>
void HMM<Distribution>::Forward(const arma::vec& dataSeq,
                                arma::vec& scales,
                                arma::mat& forwardProbabilities) const
{
  // Our goal is to calculate the forward probabilities:
  //  P(X_k | o_{1:k}) for all possible states X_k, for each time point k.
  forwardProbabilities.zeros(transition.n_rows, dataSeq.n_elem);
  scales.zeros(dataSeq.n_elem);

  // Starting state (at t = -1) is assumed to be state 0.  This is what MATLAB
  // does in their hmmdecode() function, so we will emulate that behavior.
  size_t obs = (size_t) dataSeq[0];
  forwardProbabilities.col(0) = transition.col(0) % trans(emission.row(obs));
  // Then normalize the column.
  scales[0] = accu(forwardProbabilities.col(0));
  forwardProbabilities.col(0) /= scales[0];

  // Now compute the probabilities for each successive observation.
  for (size_t t = 1; t < dataSeq.n_elem; t++)
  {
    obs = (size_t) dataSeq[t];

    for (size_t j = 0; j < transition.n_rows; j++)
    {
      // The forward probability of state j at time t is the sum over all states
      // of the probability of the previous state transitioning to the current
      // state and emitting the given observation.
      forwardProbabilities(j, t) = accu(forwardProbabilities.col(t - 1) %
          trans(transition.row(j))) * emission(obs, j);
    }

    // Normalize probability.
    scales[t] = accu(forwardProbabilities.col(t));
    forwardProbabilities.col(t) /= scales[t];
  }
}

template<typename Distribution>
void HMM<Distribution>::Backward(const arma::vec& dataSeq,
                                 const arma::vec& scales,
                                 arma::mat& backwardProbabilities) const
{
  // Our goal is to calculate the backward probabilities:
  //  P(X_k | o_{k + 1:T}) for all possible states X_k, for each time point k.
  backwardProbabilities.zeros(transition.n_rows, dataSeq.n_elem);

  // The last element probability is 1.
  backwardProbabilities.col(dataSeq.n_elem - 1).fill(1);

  // Now step backwards through all other observations.
  for (size_t t = dataSeq.n_elem - 2; t + 1 > 0; t--)
  {
    // This will eventually need to depend on the distribution.
    size_t obs = (size_t) dataSeq[t + 1];

    for (size_t j = 0; j < transition.n_rows; j++)
    {
      // The backward probability of state j at time t is the sum over all state
      // of the probability of the next state having been a transition from the
      // current state multiplied by the probability of each of those states
      // emitting the given observation.
      backwardProbabilities(j, t) += accu(transition.col(j) %
          backwardProbabilities.col(t + 1) % trans(emission.row(obs)));

      // Normalize by the weights from the forward algorithm.
      backwardProbabilities(j, t) /= scales[t + 1];
    }
  }
}

}; // namespace hmm
}; // namespace mlpack

#endif
