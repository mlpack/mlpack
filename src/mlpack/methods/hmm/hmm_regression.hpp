/**
 * @file hmm_regression.hpp
 * @author Michael Fox
 *
 * Definition of HMMRegression class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HMM_HMM_REGRESSION_HPP
#define MLPACK_METHODS_HMM_HMM_REGRESSION_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/dists/regression_distribution.hpp>
#include "hmm.hpp"

namespace mlpack {
namespace hmm /** Hidden Markov Models. */ {

/**
 * A class that represents a Hidden Markov Model Regression (HMMR). HMMR is an
 * extension of Hidden Markov Models to regression analysis. The method is
 * described in (Fridman, 1993)
 * https://www.ima.umn.edu/preprints/January1994/1195.pdf
 * An HMMR is a linear regression model whose coefficients are determined by a
 * finite-state Markov chain. The error terms are conditionally independently
 * normally distributed with zero mean and state-dependent variance. Let Q_t be
 * a finite-state Markov chain, X_t a vector of predictors and Y_t a response.
 * The HMMR is
 * Y_t = X_t \beta_{Q_t} + \sigma_{Q_t} \epsilon_t
 *
 * This HMMR class supports training (supervised and unsupervised), prediction
 * of state sequences via the Viterbi algorithm, estimation of state
 * probabilities, filtering and smoothing of responses, and calculation of the
 * log-likelihood of a given sequence.
 *
 * Usage of the HMMR class generally involves either training an HMMR or loading
 * an already-known HMMR and using to filter a sequence.
 * Example code for supervised training of an HMMR is given below.
 *
 * @code
 * // Each column is a vector of predictors for a single observation.
 * arma::mat predictors(5, 100, arma::fill::randn);
 * // Responses for each observation
 * arma::vec responses(100, arma::fill::randn);
 *
 * // Create an untrained HMMR with 3 hidden states
 * RegressionDistribution rd(predictors, responses);
 * arma::mat transition("0.5 0.5;" "0.5 0.5;");
 * std::vector<RegressionDistribution> emissions(2,rd);
 * HMMRegression hmmr("0.9 0.1", transition, emissions);
 *
 *  // Train the HMM (supply a state sequence to perform supervised training)
 * std::vector<arma::mat> predictorsSeq(1, predictors);
 * std::vector< arma::vec> responsesSeq(1, responses);
 * hmmr.Train(predictorsSeq, responsesSeq);
 * hmm.Train(observations, states);
 * @endcode
 *
 * Once initialized, the HMMR can evaluate the probability of a certain sequence
 * (with LogLikelihood()), predict the most likely sequence of hidden states
 * (with Predict()), estimate the probabilities of each state for a sequence of
 * observations (with Estimate()), or perform filtering or smoothing of
 * observations.
 *
 */
class HMMRegression : public HMM<distribution::RegressionDistribution>
{
 public:
  /**
   * Create the Hidden Markov Model Regression with the given number of hidden
   * states and the given default regression emission. The dimensionality of the
   * observations is taken from the emissions variable, so it is important that
   * the given default emission distribution is set with the correct
   * dimensionality. Alternately, set the dimensionality with Dimensionality().
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
  HMMRegression(const size_t states,
      const distribution::RegressionDistribution emissions,
      const double tolerance = 1e-5) :
      HMM<distribution::RegressionDistribution>(states, emissions, tolerance)
  { /* nothing to do */ }

  /**
   * Create the Hidden Markov Model Regression with the given initial
   * probability vector, the given transition matrix, and the given regression
   * emission distributions. The dimensionality of the observations of the HMMR
   * are taken from the given emission distributions. Alternately, the
   * dimensionality can be set with Dimensionality().
   *
   * The initial state probability vector should have length equal to the number
   * of states, and each entry represents the probability of being in the given
   * state at time T = 0 (the beginning of a sequence).
   *
   * The transition matrix should be such that T(i, j) is the probability of
   * transition to state i from state j.  The columns of the matrix should sum
   * to 1.
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
  HMMRegression(const arma::vec& initial,
      const arma::mat& transition,
      const std::vector<distribution::RegressionDistribution>& emission,
      const double tolerance = 1e-5) :
      HMM<distribution::RegressionDistribution>(initial, transition, emission,
       tolerance)
  { /* nothing to do */ }



  /**
   * Train the model using the Baum-Welch algorithm, with only the given
   * predictors and responses. Instead of giving a guess transition and emission
   * here, do that in the constructor.  Each matrix in the vector of predictors
   * corresponds to an individual data sequence, and likewise for each vec in
   * the vector of responses. The number of rows in each matrix of predictors
   * plus one should be equal to the dimensionality of the HMM (which is set in
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
   * @param predictors Vector of predictor sequences.
   * @param responses Vector of response sequences.
   */
  void Train(const std::vector<arma::mat>& predictors,
             const std::vector<arma::vec>& responses);

  /**
   * Train the model using the given labeled observations; the transition and
   * regression emissions are directly estimated. Each matrix in the vector of
   * predictors corresponds to an individual data sequence, and likewise for
   * each vec in the vector of responses. The number of rows in each matrix of
   * predictors plus one should be equal to the dimensionality of the HMM
   * (which is set in the constructor).
   *
   * @note
   * Train() can be called multiple times with different sequences; each time it
   * is called, it uses the current parameters of the HMMR as a starting point
   * for training.
   * @endnote
   *
   * @param predictors Vector of predictor sequences.
   * @param responses Vector of response sequences.
   * @param stateSeq Vector of state sequences, corresponding to each
   *     observation.
   */
  void Train(const std::vector<arma::mat>& predictors,
             const std::vector<arma::vec>& responses,
             const std::vector<arma::Row<size_t> >& stateSeq);

  /**
   * Estimate the probabilities of each hidden state at each time step for each
   * given data observation, using the Forward-Backward algorithm.  Each matrix
   * which is returned has columns equal to the number of data observations, and
   * rows equal to the number of hidden states in the model.  The log-likelihood
   * of the most probable sequence is returned.
   *
   * @param predictors Vector of predictor sequences.
   * @param responses Vector of response sequences.
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
  double Estimate(const arma::mat& predictors,
                  const arma::vec& responses,
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
   * @param predictors Vector of predictor sequences.
   * @param responses Vector of response sequences.
   * @param stateProb Probabilities of each state at each time interval.
   * @return Log-likelihood of most likely state sequence.
   */
  double Estimate(const arma::mat& predictors,
                  const arma::vec& responses,
                  arma::mat& stateProb) const;

  /**
   * Compute the most probable hidden state sequence for the given predictors
   * and responses, using the Viterbi algorithm, returning the log-likelihood of
   * the most likely state sequence.
   *
   * @param predictors Vector of predictor sequences.
   * @param responses Vector of response sequences.
   * @param stateSeq Vector in which the most probable state sequence will be
   *    stored.
   * @return Log-likelihood of most probable state sequence.
   */
  double Predict(const arma::mat& predictors,
                 const arma::vec& responses,
                 arma::Row<size_t>& stateSeq) const;

  /**
   * Compute the log-likelihood of the given predictors and responses.
   *
   * @param predictors Vector of predictor sequences.
   * @param responses Vector of response sequences.
   * @return Log-likelihood of the given sequence.
   */
  double LogLikelihood(const arma::mat& predictors,
                       const arma::vec& responses) const;
  /**
   * HMMR filtering. Computes the k-step-ahead expected response at each time
   * conditioned only on prior observations. That is
   * E{ Y[t+k] | Y[0], ..., Y[t] }.
   * The returned matrix has columns equal to the number of observations. Note
   * that the expectation may not be meaningful for discrete emissions.
   *
   * @param predictors Vector of predictor sequences.
   * @param responses Vector of response sequences.
   * @param initial Distribution of initial state.
   * @param ahead Number of steps ahead (k) for expectations.
   * @param filterSeq Vector in which the expected emission sequence will be
   *    stored.
   */
  void Filter(const arma::mat& predictors,
              const arma::vec& responses,
              arma::vec& filterSeq,
              size_t ahead = 0) const;

  /**
   * HMM smoothing. Computes expected emission at each time conditioned on all
   * observations. That is
   * E{ Y[t] | Y[0], ..., Y[T] }.
   * The returned matrix has columns equal to the number of observations. Note
   * that the expectation may not be meaningful for discrete emissions.
   *
   * @param predictors Vector of predictor sequences.
   * @param responses Vector of response sequences..
   * @param initial Distribution of initial state.
   * @param smoothSeq Vector in which the expected emission sequence will be
   *    stored.
   */
  void Smooth(const arma::mat& predictors,
              const arma::vec& responses,
              arma::vec& smoothSeq) const;

 private:
  /**
   * Utility functions to facilitate the use of the HMM class for HMMR.
   */
   void StackData(const std::vector<arma::mat>& predictors,
                  const std::vector<arma::vec>& responses,
                  std::vector<arma::mat>& dataSeq) const;

   void StackData(const arma::mat& predictors,
                  const arma::vec& responses,
                  arma::mat& dataSeq) const;

  /**
   * The Forward algorithm (part of the Forward-Backward algorithm).  Computes
   * forward probabilities for each state for each observation in the given data
   * sequence.  The returned matrix has rows equal to the number of hidden
   * states and columns equal to the number of observations.
   *
   * @param predictors Vector of predictor sequences.
   * @param responses Vector of response sequences.
   * @param scales Vector in which scaling factors will be saved.
   * @param forwardProb Matrix in which forward probabilities will be saved.
   */
  void Forward(const arma::mat& predictors,
               const arma::vec& responses,
               arma::vec& scales,
               arma::mat& forwardProb) const;

  /**
   * The Backward algorithm (part of the Forward-Backward algorithm).  Computes
   * backward probabilities for each state for each observation in the given
   * data sequence, using the scaling factors found (presumably) by Forward().
   * The returned matrix has rows equal to the number of hidden states and
   * columns equal to the number of observations.
   *
   * @param predictors Vector of predictor sequences.
   * @param responses Vector of response sequences.
   * @param scales Vector of scaling factors.
   * @param backwardProb Matrix in which backward probabilities will be saved.
   */
  void Backward(const arma::mat& predictors,
                const arma::vec& responses,
                const arma::vec& scales,
                arma::mat& backwardProb) const;


};

} // namespace hmm
} // namespace mlpack

// Include implementation.
#include "hmm_regression_impl.hpp"

#endif
