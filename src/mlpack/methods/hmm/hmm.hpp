/**
 * @file hmm.hpp
 * @author Ryan Curtin
 * @author Tran Quoc Long
 *
 * Definition of HMM class.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_HMM_HMM_HPP
#define __MLPACK_METHODS_HMM_HMM_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace hmm /** Hidden Markov Models. */ {

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
 * Gaussians (GMM), or any other probability distribution implementing the
 * four Distribution functions.
 *
 * Usage of the HMM class generally involves either training an HMM or loading
 * an already-known HMM and taking probability measurements of sequences.
 * Example code for supervised training of a Gaussian HMM (that is, where the
 * emission output distribution is a single Gaussian for each hidden state) is
 * given below.
 *
 * @code
 * extern arma::mat observations; // Each column is an observation.
 * extern arma::Col<size_t> states; // Hidden states for each observation.
 * // Create an untrained HMM with 5 hidden states and default (N(0, 1))
 * // Gaussian distributions with the dimensionality of the dataset.
 * HMM<GaussianDistribution> hmm(5, GaussianDistribution(observations.n_rows));
 *
 * // Train the HMM (the labels could be omitted to perform unsupervised
 * // training).
 * hmm.Train(observations, states);
 * @endcode
 *
 * Once initialized, the HMM can evaluate the probability of a certain sequence
 * (with LogLikelihood()), predict the most likely sequence of hidden states
 * (with Predict()), generate a sequence (with Generate()), or estimate the
 * probabilities of each state for a sequence of observations (with Estimate()).
 *
 * @tparam Distribution Type of emission distribution for this HMM.
 */
template<typename Distribution = distribution::DiscreteDistribution>
class HMM
{
 public:
  /**
   * Create the Hidden Markov Model with the given number of hidden states and
   * the given default distribution for emissions.  The dimensionality of the
   * observations is taken from the emissions variable, so it is important that
   * the given default emission distribution is set with the correct
   * dimensionality.  Alternately, set the dimensionality with Dimensionality().
   * Optionally, the tolerance for convergence of the Baum-Welch algorithm can
   * be set.
   *
   * By default, the transition matrix and initial probability vector are set to
   * contain equal probability for each state.
   *
   * @param states Number of states.
   * @param emissions Default distribution for emissions.
   * @param tolerance Tolerance for convergence of training algorithm
   *      (Baum-Welch).
   */
  HMM(const size_t states,
      const Distribution emissions,
      const double tolerance = 1e-5);

  /**
   * Create the Hidden Markov Model with the given initial probability vector,
   * the given transition matrix, and the given emission distributions.  The
   * dimensionality of the observations of the HMM are taken from the given
   * emission distributions.  Alternately, the dimensionality can be set with
   * Dimensionality().
   *
   * The initial state probability vector should have length equal to the number
   * of states, and each entry represents the probability of being in the given
   * state at time T = 0 (the beginning of a sequence).
   *
   * The transition matrix should be such that T(i, j) is the probability of
   * transition to state i from state j.  The columns of the matrix should sum
   * to 1.
   *
   * The emission matrix should be such that E(i, j) is the probability of
   * emission i while in state j.  The columns of the matrix should sum to 1.
   *
   * Optionally, the tolerance for convergence of the Baum-Welch algorithm can
   * be set.
   *
   * @param initial Initial state probabilities.
   * @param transition Transition matrix.
   * @param emission Emission distributions.
   * @param tolerance Tolerance for convergence of training algorithm
   *      (Baum-Welch).
   */
  HMM(const arma::vec& initial,
      const arma::mat& transition,
      const std::vector<Distribution>& emission,
      const double tolerance = 1e-5);

  /**
   * Train the model using the Baum-Welch algorithm, with only the given
   * unlabeled observations.  Instead of giving a guess transition and emission
   * matrix here, do that in the constructor.  Each matrix in the vector of data
   * sequences holds an individual data sequence; each point in each individual
   * data sequence should be a column in the matrix.  The number of rows in each
   * matrix should be equal to the dimensionality of the HMM (which is set in
   * the constructor).
   *
   * It is preferable to use the other overload of Train(), with labeled data.
   * That will produce much better results.  However, if labeled data is
   * unavailable, this will work.  In addition, it is possible to use Train()
   * with labeled data first, and then continue to train the model using this
   * overload of Train() with unlabeled data.
   *
   * The tolerance of the Baum-Welch algorithm can be set either in the
   * constructor or with the Tolerance() method.  When the change in
   * log-likelihood of the model between iterations is less than the tolerance,
   * the Baum-Welch algorithm terminates.
   *
   * @note
   * Train() can be called multiple times with different sequences; each time it
   * is called, it uses the current parameters of the HMM as a starting point
   * for training.
   * @endnote
   *
   * @param dataSeq Vector of observation sequences.
   */
  void Train(const std::vector<arma::mat>& dataSeq);

  /**
   * Train the model using the given labeled observations; the transition and
   * emission matrices are directly estimated.  Each matrix in the vector of
   * data sequences corresponds to a vector in the vector of state sequences.
   * Each point in each individual data sequence should be a column in the
   * matrix, and its state should be the corresponding element in the state
   * sequence vector.  For instance, dataSeq[0].col(3) corresponds to the fourth
   * observation in the first data sequence, and its state is stateSeq[0][3].
   * The number of rows in each matrix should be equal to the dimensionality of
   * the HMM (which is set in the constructor).
   *
   * @note
   * Train() can be called multiple times with different sequences; each time it
   * is called, it uses the current parameters of the HMM as a starting point
   * for training.
   * @endnote
   *
   * @param dataSeq Vector of observation sequences.
   * @param stateSeq Vector of state sequences, corresponding to each
   *     observation.
   */
  void Train(const std::vector<arma::mat>& dataSeq,
             const std::vector<arma::Col<size_t> >& stateSeq);

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
  double Estimate(const arma::mat& dataSeq,
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
  double Estimate(const arma::mat& dataSeq,
                  arma::mat& stateProb) const;

  /**
   * Generate a random data sequence of the given length.  The data sequence is
   * stored in the dataSequence parameter, and the state sequence is stored in
   * the stateSequence parameter.  Each column of dataSequence represents a
   * random observation.
   *
   * @param length Length of random sequence to generate.
   * @param dataSequence Vector to store data in.
   * @param stateSequence Vector to store states in.
   * @param startState Hidden state to start sequence in (default 0).
   */
  void Generate(const size_t length,
                arma::mat& dataSequence,
                arma::Col<size_t>& stateSequence,
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
  double Predict(const arma::mat& dataSeq,
                 arma::Col<size_t>& stateSeq) const;

  /**
   * Compute the log-likelihood of the given data sequence.
   *
   * @param dataSeq Data sequence to evaluate the likelihood of.
   * @return Log-likelihood of the given sequence.
   */
  double LogLikelihood(const arma::mat& dataSeq) const;

  //! Return the vector of initial state probabilities.
  const arma::vec& Initial() const { return initial; }
  //! Modify the vector of initial state probabilities.
  arma::vec& Initial() { return initial; }

  //! Return the transition matrix.
  const arma::mat& Transition() const { return transition; }
  //! Return a modifiable transition matrix reference.
  arma::mat& Transition() { return transition; }

  //! Return the emission distributions.
  const std::vector<Distribution>& Emission() const { return emission; }
  //! Return a modifiable emission probability matrix reference.
  std::vector<Distribution>& Emission() { return emission; }

  //! Get the dimensionality of observations.
  size_t Dimensionality() const { return dimensionality; }
  //! Set the dimensionality of observations.
  size_t& Dimensionality() { return dimensionality; }

  //! Get the tolerance of the Baum-Welch algorithm.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance of the Baum-Welch algorithm.
  double& Tolerance() { return tolerance; }

  /**
   * Returns a string representation of this object.
   */
  std::string ToString() const;

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
  void Forward(const arma::mat& dataSeq,
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
  void Backward(const arma::mat& dataSeq,
                const arma::vec& scales,
                arma::mat& backwardProb) const;

  //! Initial state probability vector.
  arma::vec initial;

  //! Transition probability matrix.
  arma::mat transition;

  //! Set of emission probability distributions; one for each state.
  std::vector<Distribution> emission;

  //! Dimensionality of observations.
  size_t dimensionality;

  //! Tolerance of Baum-Welch algorithm.
  double tolerance;
};

}; // namespace hmm
}; // namespace mlpack

// Include implementation.
#include "hmm_impl.hpp"

#endif
