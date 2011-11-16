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
#include "distributions/discrete_distribution.hpp"

namespace mlpack {
namespace hmm {

/**
 * A class that represents a Hidden Markov Model with an arbitrary type of
 * emission distribution.  This HMM class supports training (supervised and
 * unsupervised), prediction of state sequences via the Viterbi algorithm,
 * estimation of state probabilities, generation of random sequences, and
 * calculation of the log-likelihood of a given sequence.
 *
 * The template parameter, Distribution, specifies the distribution which the
 * emissions follow.  The class should implement the following functions:
 *
 * @code
 * class Distribution
 * {
 *  public:
 *   // The type of observation used by this distribution.
 *   typedef something DataType;
 *
 *   // Return the probability of the given observation.
 *   double Probability(const DataType& observation) const;
 *
 *   // Estimate the distribution based on the given observations.
 *   void Estimate(const std::vector<DataType>& observations);
 *
 *   // Estimate the distribution based on the given observations, given also
 *   // the probability of each observation coming from this distribution.
 *   void Estimate(const std::vector<DataType>& observations,
 *                 const std::vector<double>& probabilities);
 * };
 * @endcode
 *
 * See the mlpack::distribution::DiscreteDistribution class for an example.  One
 * would use the DiscreteDistribution class when the observations are
 * non-negative integers.  Other distributions could be Gaussians, a mixture of
 * Gaussians (GMM), or any other probability distribution.
 */
template<typename Distribution = distribution::DiscreteDistribution>
class HMM
{
 private:
  //! Transition probability matrix.
  arma::mat transition;

  //! Set of emission probability distributions; one for each state.
  std::vector<Distribution> emission;

 public:
  //! Convenience typedef for the type of observation we are using.
  typedef typename Distribution::DataType Observation;

  /**
   * Create the Hidden Markov Model with the given number of hidden states and
   * the given default distribution for emissions.
   *
   * @param states Number of states.
   * @param emissions Default distribution for emissions.
   */
  HMM(const size_t states, const Distribution emissions);

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
  HMM(const arma::mat& transition, const std::vector<Distribution>& emission);

  /**
   * Train the model using the Baum-Welch algorithm, with only the given
   * unlabeled observations.  Instead of giving a guess transition and emission
   * matrix here, do that in the constructor.
   *
   * @param dataSeq Vector of observation sequences.
   */
  void Train(const std::vector<std::vector<Observation> >& dataSeq);

  /**
   * Train the model using the given labeled observations; the transition and
   * emission matrices are directly estimated.
   *
   * @param dataSeq Vector of observation sequences.
   * @param stateSeq Vector of state sequences, corresponding to each
   *    observation.
   */
  void Train(const std::vector<std::vector<Observation> >& dataSeq,
             const std::vector<std::vector<size_t> >& stateSeq);

  /**
   * Estimate the probabilities of each hidden state at each time step for each
   * given data observation, using the Forward-Backward algorithm.  Each matrix
   * which is returned has columns equal to the number of data observations, and
   * rows equal to the number of hidden states in the model.  The log-likelihood
   * of the most probable sequence is returned.
   *
   * @param dataSeq Sequence of observations.
   * @param stateProb Matrix in which the probabilities of each state at each
   *    time interval will be stored.
   * @param forwardProb Matrix in which the forward probabilities of each state
   *    at each time interval will be stored.
   * @param backwardProb Matrix in which the backward probabilities of each
   *    state at each time interval will be stored.
   * @param scales Vector in which the scaling factors at each time interval
   *    will be stored.
   * @return Log-likelihood of most likely state sequence.
   */
  double Estimate(const std::vector<Observation>& dataSeq,
                  arma::mat& stateProb,
                  arma::mat& forwardProb,
                  arma::mat& backwardProb,
                  arma::vec& scales) const;

  /**
   * Estimate the probabilities of each hidden state at each time step of each
   * given data observation, using the Forward-Backward algorithm.  The returned
   * matrix of state probabilities has columns equal to the number of data
   * observations, and rows equal to the number of hidden states in the model.
   * The log-likelihood of the most probable sequence is returned.
   *
   * @param dataSeq Sequence of observations.
   * @param stateProb Probabilities of each state at each time interval.
   * @return Log-likelihood of most likely state sequence.
   */
  double Estimate(const std::vector<Observation>& dataSeq,
                  arma::mat& stateProb) const;

  /**
   * Generate a random data sequence of the given length.  The data sequence is
   * stored in the data_sequence parameter, and the state sequence is stored in
   * the state_sequence parameter.
   *
   * @param length Length of random sequence to generate.
   * @param dataSequence Vector to store data in.
   * @param stateSequence Vector to store states in.
   * @param startState Hidden state to start sequence in (default 0).
   */
  void Generate(const size_t length,
                std::vector<Observation>& dataSequence,
                std::vector<size_t>& stateSequence,
                const size_t startState = 0) const;

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
  double Predict(const std::vector<Observation>& dataSeq,
                 std::vector<size_t>& stateSeq) const;

  /**
   * Compute the log-likelihood of the given data sequence.
   *
   * @param dataSeq Data sequence to evaluate the likelihood of.
   * @return Log-likelihood of the given sequence.
   */
  double LogLikelihood(const std::vector<Observation>& dataSeq) const;

  /**
   * Return the transition matrix.
   */
  const arma::mat& Transition() const { return transition; }

  /**
   * Return a modifiable transition matrix reference.
   */
  arma::mat& Transition() { return transition; }

  /**
   * Return the emission distributions.
   */
  const std::vector<Distribution>& Emission() const { return emission; }

  /**
   * Return a modifiable emission probability matrix reference.
   */
  std::vector<Distribution>& Emission() { return emission; }

 private:
  // Helper functions.

  /**
   * The Forward algorithm (part of the Forward-Backward algorithm).  Computes
   * forward probabilities for each state for each observation in the given data
   * sequence.  The returned matrix has rows equal to the number of hidden
   * states and columns equal to the number of observations.
   *
   * @param dataSeq Data sequence to compute probabilities for.
   * @param scales Vector in which scaling factors will be saved.
   * @param forwardProb Matrix in which forward probabilities will be saved.
   */
  void Forward(const std::vector<Observation>& dataSeq,
               arma::vec& scales,
               arma::mat& forwardProb) const;

  /**
   * The Backward algorithm (part of the Forward-Backward algorithm).  Computes
   * backward probabilities for each state for each observation in the given
   * data sequence, using the scaling factors found (presumably) by Forward().
   * The returned matrix has rows equal to the number of hidden states and
   * columns equal to the number of observations.
   *
   * @param dataSeq Data sequence to compute probabilities for.
   * @param scales Vector of scaling factors.
   * @param backwardProb Matrix in which backward probabilities will be saved.
   */
  void Backward(const std::vector<Observation>& dataSeq,
                const arma::vec& scales,
                arma::mat& backwardProb) const;
};

}; // namespace hmm
}; // namespace mlpack

// Include implementation.
#include "hmm_impl.hpp"

#endif
