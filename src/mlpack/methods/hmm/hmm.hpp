/**
 * @file hmm.hpp
 * @author Ryan Curtin
 * @author Tran Quoc Long
 *
 * Definition of HMM class.
 */
#ifndef __MLPACK_METHODS_HMM_HMM_HPP
#define __MLPACK_METHODS_HMM_HMM_HPP

#include <mlpack/core.h>

namespace mlpack {
namespace hmm {

/**
 * A class that represents a Hidden Markov Model.
 */
template<typename Distribution>
class HMM
{
 private:
  //! Transition probability matrix.
  arma::mat transition;

  //! Emission probability matrix (for each state).
  arma::mat emission;

 public:
  /**
   * Create the Hidden Markov Model with the given number of hidden states and
   * the given number of emission states.
   *
   * @param states Number of states.
   * @param emissions Number of possible emissions.
   */
  HMM(const size_t states, const size_t emissions);

  /**
   * Create the Hidden Markov Model with the given transition matrix and the
   * given emission probability matrix.
   *
   * The transition matrix should be such that T(i, j) is the probability of
   * transition to state i from state j.  The columns of the matrix should sum
   * to 1.
   *
   * The emission matrix should be such that E(i, j) is the probability of
   * emission i while in state j.  The columns of the matrix should sum to 1.
   *
   * @param transition Transition matrix.
   * @param emission Emission probability matrix.
   */
  HMM(const arma::mat& transition, const arma::mat& emission);

  /**
   * Train the model using the Baum-Welch algorithm, with only the given
   * unlabeled observations.  Instead of giving a guess transition and emission
   * matrix here, do that in the constructor.
   *
   * @param dataSeq Vector of observation sequences.
   */
  void Train(const std::vector<arma::vec>& dataSeq);

  /**
   * Train the model using the given labeled observations; the transition and
   * emission matrices are directly estimated.
   *
   * @param dataSeq Vector of observation sequences.
   * @param stateSeq Vector of state sequences, corresponding to each
   *    observation.
   */
  void Train(const std::vector<arma::vec>& dataSeq,
             const std::vector<arma::Col<size_t> >& stateSeq);

  /**
   * Generate a random data sequence of the given length.  The data sequence is
   * stored in the data_sequence parameter, and the state sequence is stored in
   * the state_sequence parameter.
   *
   * @param length Length of random sequence to generate.
   * @param data_sequence Vector to store data in.
   * @param state_sequence Vector to store states in.
   */
  void GenerateSequence(const size_t length,
                        arma::vec& data_sequence,
                        arma::vec& state_sequence) const;

  /**
   * Estimate the probabilities of each hidden state at each time step for each
   * given data observation.
   */
  double Estimate(const arma::vec& data_seq,
                  arma::mat& state_prob_mat,
                  arma::mat& forward_prob_mat,
                  arma::mat& backward_prob_mat,
                  arma::vec& scale_vec) const;

  /**
   * Compute the log-likelihood of a sequence.
   */
  double LogLikelihood(const arma::vec& data_seq) const;

  /**
   * Compute the most probable hidden state sequence for the given data
   * sequence, using the Viterbi algorithm, returning the log-likelihood of the
   * most likely state sequence.
   *
   * @param dataSeq Sequence of observations.
   * @param stateSeq Vector in which the most probable state sequence will be
   *    stored.
   * @return Log-likelihood of most probable state sequence.
   */
  double Predict(const arma::vec& data_seq, arma::Col<size_t>& stateSeq) const;

  /**
   * Return the transition matrix.
   */
  const arma::mat& Transition() const { return transition; }

  /**
   * Return a modifiable transition matrix reference.
   */
  arma::mat& Transition() { return transition; }

  /**
   * Return the emission probability matrix.
   */
  const arma::mat& Emission() const { return emission; }

  /**
   * Return a modifiable emission probability matrix reference.
   */
  arma::mat& Emission() { return emission; }

 private:
  // Helper functions.

  void Forward(const arma::vec& data_seq, arma::vec& scales, arma::mat& fs)
      const;
  void Backward(const arma::vec& data_seq, const arma::vec& scales,
                arma::mat& bs) const;
};

}; // namespace hmm
}; // namespace mlpack

// Include implementation.
#include "hmm_impl.hpp"

#endif
