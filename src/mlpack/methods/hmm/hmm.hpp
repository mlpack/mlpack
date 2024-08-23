/**
 * @file methods/hmm/hmm.hpp
 * @author Ryan Curtin
 * @author Tran Quoc Long
 * @author Michael Fox
 *
 * Definition of HMM class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HMM_HMM_HPP
#define MLPACK_METHODS_HMM_HMM_HPP

#include <mlpack/core.hpp>

namespace mlpack {

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
 *   double Train(const std::vector<DataType>& observations);
 *
 *   // Estimate the distribution based on the given observations, given also
 *   // the probability of each observation coming from this distribution.
 *   double Train(const std::vector<DataType>& observations,
 *                const std::vector<double>& probabilities);
 * };
 * @endcode
 *
 * See the DiscreteDistribution class for an example.  One
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
 * extern arma::Row<size_t> states; // Hidden states for each observation.
 * // Create an untrained HMM with 5 hidden states and default (N(0, 1))
 * // Gaussian distributions with the dimensionality of the dataset.
 * HMM<GaussianDistribution<>> hmm(5,
 *     GaussianDistribution<>(observations.n_rows));
 *
 * // Train the HMM (the labels could be omitted to perform unsupervised
 * // training).
 * hmm.Train(observations, states);
 * @endcode
 *
 * Once initialized, the HMM can evaluate the probability of a certain sequence
 * (with LogLikelihood()), predict the most likely sequence of hidden states
 * (with Predict()), generate a sequence (with Generate()), or estimate the
 * probabilities of each state for a sequence of observations (with Train()).
 *
 * @tparam Distribution Type of emission distribution for this HMM.
 */
template<typename Distribution = DiscreteDistribution<>>
class HMM
{
 public:
  /**
   * Create the Hidden Markov Model with the given number of hidden states and
   * the given default distribution for emissions.  The dimensionality of the
   * observations is taken from the emissions variable, so it is important that
   * the given default emission distribution is set with the correct
   * dimensionality.  Alternately, set the dimensionality with Dimensionality(),
   * and then use Emission() to access and set the dimensionality of each
   * individual distribution correctly.
   *
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
  HMM(const size_t states = 0,
      const Distribution emissions = Distribution(),
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
   *
   * @param dataSeq Vector of observation sequences.
   * @return Log-likelihood of state sequence.
   */
  double Train(const std::vector<arma::mat>& dataSeq);

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
   *
   * @param dataSeq Vector of observation sequences.
   * @param stateSeq Vector of state sequences, corresponding to each
   *     observation.
   */
  void Train(const std::vector<arma::mat>& dataSeq,
             const std::vector<arma::Row<size_t> >& stateSeq);

  /**
   * Estimate the probabilities of each hidden state at each time step for each
   * given data observation, using the Forward-Backward algorithm.  Each matrix
   * which is returned has columns equal to the number of data observations, and
   * rows equal to the number of hidden states in the model.  The log-likelihood
   * of the most probable sequence is returned.
   *
   * @param dataSeq Sequence of observations.
   * @param stateLogProb Matrix in which the log probabilities of each state at
   *    each time interval will be stored.
   * @param forwardLogProb Matrix in which the forward log probabilities of each
   *    state at each time interval will be stored.
   * @param backwardLogProb Matrix in which the backward log probabilities of each
   *    state at each time interval will be stored.
   * @param logScales Vector in which the log of scaling factors at each time
   *    interval will be stored.
   * @return Log-likelihood of most likely state sequence.
   */
  double LogEstimate(const arma::mat& dataSeq,
                     arma::mat& stateLogProb,
                     arma::mat& forwardLogProb,
                     arma::mat& backwardLogProb,
                     arma::vec& logScales) const;

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
                arma::Row<size_t>& stateSequence,
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
                 arma::Row<size_t>& stateSeq) const;

  /**
   * Compute the log-likelihood of the given data sequence.
   *
   * @param dataSeq Data sequence to evaluate the likelihood of.
   * @return Log-likelihood of the given sequence.
   */
  double LogLikelihood(const arma::mat& dataSeq) const;

  /**
   * Compute the log of the scaling factor of the given emission probability
   * at time t. To calculate the log-likelihood for the whole sequence,
   * accumulate log scale over the entire sequence
   * This is meant for incremental or streaming computation of the
   * log-likelihood of a sequence. For the first data point, provide an empty
   * forwardLogProb vector.
   *
   * @param emissionLogProb emission probability at time t.
   * @param forwardLogProb Vector in which forward probabilities will be saved.
   *     Passing forwardLogProb as an empty vector indicates the start of the
   *     sequence (i.e. time t=0).
   * @return Log scale factor of the given sequence of emission at time t.
   */
  double EmissionLogScaleFactor(const arma::vec& emissionLogProb,
                                arma::vec& forwardLogProb) const;

  /**
   * Compute the log-likelihood of the given emission probability up to time t,
   * storing the result in logLikelihood.
   * This is meant for incremental or streaming computation of the
   * log-likelihood of a sequence. For the first data point, provide an empty
   * forwardLogProb vector.
   *
   * @param emissionLogProb emission probability at time t.
   * @param logLikelihood Log-likelihood of the given sequence of emission
   *     probability up to time t-1.  This will be overwritten with the
   *     log-likelihood of the given emission probability up to time t.
   * @param forwardLogProb Vector in which forward probabilities will be saved.
   *     Passing forwardLogProb as an empty vector indicates the start of the
   *     sequence (i.e. time t=0).
   * @return Log-likelihood of the given sequence of emission up to time t.
   */
  double EmissionLogLikelihood(const arma::vec& emissionLogProb,
                               double &logLikelihood,
                               arma::vec& forwardLogProb) const;

  /**
   * Compute the log of the scaling factor of the given data at time t.
   * To calculate the log-likelihood for the whole sequence, accumulate the
   * log scale factor (the return value of this function) over the entire
   * sequence.
   * This is meant for incremental or streaming computation of the
   * log-likelihood of a sequence. For the first data point, provide an empty
   * forwardLogProb vector.
   *
   * @param data observation at time t.
   * @param forwardLogProb Vector in which forward probabilities will be saved.
   *     Passing forwardLogProb as an empty vector indicates the start of the
   *     sequence (i.e. time t=0).
   * @return Log scale factor of the given sequence of data up at time t.
   */
  double LogScaleFactor(const arma::vec &data,
                        arma::vec& forwardLogProb) const;

  /**
   * Compute the log-likelihood of the given data up to time t, storing the
   * result in logLikelihood.
   * This is meant for incremental or streaming computation of the
   * log-likelihood of a sequence. For the first data point, provide an empty
   * forwardLogProb vector.
   *
   * @param data observation at time t.
   * @param logLikelihood Log-likelihood of the given sequence of data
   *     up to time t-1.
   * @param forwardLogProb Vector in which forward probabilities will be saved.
   *     Passing forwardLogProb as an empty vector indicates the start of the
   *     sequence (i.e. time t=0).
   * @return Log-likelihood of the given sequence of data up to time t.
   */
  double LogLikelihood(const arma::vec &data,
                       double &logLikelihood,
                       arma::vec& forwardLogProb) const;
  /**
   * HMM filtering. Computes the k-step-ahead expected emission at each time
   * conditioned only on prior observations. That is
   * E{ Y[t+k] | Y[0], ..., Y[t] }.
   * The returned matrix has columns equal to the number of observations. Note
   * that the expectation may not be meaningful for discrete emissions.
   *
   * @param dataSeq Sequence of observations.
   * @param filterSeq Vector in which the expected emission sequence will be
   *    stored.
   * @param ahead Number of steps ahead (k) for expectations.
   */
  void Filter(const arma::mat& dataSeq,
              arma::mat& filterSeq,
              size_t ahead = 0) const;

  /**
   * HMM smoothing. Computes expected emission at each time conditioned on all
   * observations. That is
   * E{ Y[t] | Y[0], ..., Y[T] }.
   * The returned matrix has columns equal to the number of observations. Note
   * that the expectation may not be meaningful for discrete emissions.
   *
   * @param dataSeq Sequence of observations.
   * @param smoothSeq Vector in which the expected emission sequence will be
   *    stored.
   */
  void Smooth(const arma::mat& dataSeq,
              arma::mat& smoothSeq) const;

  //! Return the vector of initial state probabilities.
  const arma::vec& Initial() const { return initialProxy; }
  //! Modify the vector of initial state probabilities.
  arma::vec& Initial()
  {
    recalculateInitial = true;
    return initialProxy;
  }

  //! Return the transition matrix.
  const arma::mat& Transition() const { return transitionProxy; }
  //! Return a modifiable transition matrix reference.
  arma::mat& Transition()
  {
    recalculateTransition = true;
    return transitionProxy;
  }

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
   * Load the object.
   */
  template<typename Archive>
  void load(Archive& ar, const uint32_t version);

  /**
   * Save the object.
   */
  template<typename Archive>
  void save(Archive& ar, const uint32_t version) const;

 protected:
  /**
   * Given emission probabilities, computes forward probabilities at time t=0.
   *
   * @param emissionLogProb Emission probability at time t=0.
   * @param logScales Vector in which the log of scaling factors will be saved.
   * @return Forward probabilities
   */
  arma::vec ForwardAtT0(const arma::vec& emissionLogProb,
                        double& logScales) const;

  /**
   * Given emission probabilities, computes forward probabilities for time t>0.
   *
   * @param emissionLogProb Emission probability at time t>0.
   * @param logScales Vector in which the log of scaling factors will be saved.
   * @param prevForwardLogProb Previous forward probabilities.
   * @return Forward probabilities
   */
  arma::vec ForwardAtTn(const arma::vec& emissionLogProb,
                        double& logScales,
                        const arma::vec& prevForwardLogProb) const;

  // Helper functions.
  /**
   * The Forward algorithm (part of the Forward-Backward algorithm).  Computes
   * forward probabilities for each state for each observation in the given data
   * sequence.  The returned matrix has rows equal to the number of hidden
   * states and columns equal to the number of observations.
   *
   * @param dataSeq Data sequence to compute probabilities for.
   * @param logScales Vector in which the log of scaling factors will be saved.
   * @param forwardLogProb Matrix in which forward probabilities will be saved.
   */
  void Forward(const arma::mat& dataSeq,
               arma::vec& logScales,
               arma::mat& forwardLogProb,
               arma::mat& logProbs) const;

  /**
   * The Backward algorithm (part of the Forward-Backward algorithm).  Computes
   * backward probabilities for each state for each observation in the given
   * data sequence, using the scaling factors found (presumably) by Forward().
   * The returned matrix has rows equal to the number of hidden states and
   * columns equal to the number of observations.
   *
   * @param dataSeq Data sequence to compute probabilities for.
   * @param logScales Vector of log of scaling factors.
   * @param backwardLogProb Matrix in which backward probabilities will be saved.
   */
  void Backward(const arma::mat& dataSeq,
                const arma::vec& logScales,
                arma::mat& backwardLogProb,
                arma::mat& logProbs) const;

  //! Set of emission probability distributions; one for each state.
  std::vector<Distribution> emission;

  /**
   * A proxy variable in linear space for logTransition.
   * Should be removed in mlpack 4.0.
   */
  arma::mat transitionProxy;

  //! Transition probability matrix. No need to be mutable in mlpack 4.0.
  mutable arma::mat logTransition;

 private:
  /**
   * Make sure the variables in log space are in sync
   * with the linear counter parts.
   * Should be removed in mlpack 4.0.
   */
  void ConvertToLogSpace() const;

  /**
   * A proxy vriable in linear space for logInitial.
   * Should be removed in mlpack 4.0.
   */
  arma::vec initialProxy;

  //! Initial state probability vector. No need to be mutable in mlpack 4.0.
  mutable arma::vec logInitial;

  //! Dimensionality of observations.
  size_t dimensionality;

  //! Tolerance of Baum-Welch algorithm.
  double tolerance;

  /**
   * Whether or not we need to update the logInitial from initialProxy.
   * Should be removed in mlpack 4.0.
   */
  mutable bool recalculateInitial;

  /**
   * Whether or not we need to update the logTransition from transitionProxy.
   * Should be removed in mlpack 4.0.
   */
  mutable bool recalculateTransition;
};

} // namespace mlpack

// Include implementation.
#include "hmm_impl.hpp"

#endif
