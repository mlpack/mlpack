/**
 * @file hmm_impl.hpp
 * @author Ryan Curtin
 * @author Tran Quoc Long
 * @author Michael Fox
 * @author Rohan Raj
 *
 * Implementation of HMM class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HMM_HMM_IMPL_HPP
#define MLPACK_METHODS_HMM_HMM_IMPL_HPP

// Just in case...
#include "hmm.hpp"
#include <mlpack/core/math/log_add.hpp>

namespace mlpack {
namespace hmm {

/**
 * Create the Hidden Markov Model with the given number of hidden states and the
 * given number of emission states.
 */
template<typename Distribution>
HMM<Distribution>::HMM(const size_t states,
                       const Distribution emissions,
                       const double tolerance) :
    emission(states, /* default distribution */ emissions),
    transition(arma::randu<arma::mat>(states, states)),
    initial(arma::randu<arma::vec>(states) / (double) states),
    dimensionality(emissions.Dimensionality()),
    tolerance(tolerance)
{
  // Normalize the transition probabilities and initial state probabilities.
  initial /= arma::accu(initial);
  for (size_t i = 0; i < transition.n_cols; ++i)
    transition.col(i) /= arma::accu(transition.col(i));
}

/**
 * Create the Hidden Markov Model with the given transition matrix and the given
 * emission probability matrix.
 */
template<typename Distribution>
HMM<Distribution>::HMM(const arma::vec& initial,
                       const arma::mat& transition,
                       const std::vector<Distribution>& emission,
                       const double tolerance) :
    emission(emission),
    transition(transition),
    initial(initial),
    tolerance(tolerance)
{
  // Set the dimensionality, if we can.
  if (emission.size() > 0)
    dimensionality = emission[0].Dimensionality();
  else
  {
    Log::Warn << "HMM::HMM(): no emission distributions given; assuming a "
        << "dimensionality of 0 and hoping it gets set right later."
        << std::endl;
    dimensionality = 0;
  }
}

/**
 * Train the model using the Baum-Welch algorithm, with only the given unlabeled
 * observations.  Each matrix in the vector of data sequences holds an
 * individual data sequence; each point in each individual data sequence should
 * be a column in the matrix.  The number of rows in each matrix should be equal
 * to the dimensionality of the HMM.
 *
 * It is preferable to use the other overload of Train(), with labeled data.
 * That will produce much better results.  However, if labeled data is
 * unavailable, this will work.  In addition, it is possible to use Train() with
 * labeled data first, and then continue to train the model using this overload
 * of Train() with unlabeled data.
 *
 * @param dataSeq Set of data sequences to train on.
 */
template<typename Distribution>
double HMM<Distribution>::Train(const std::vector<arma::mat>& dataSeq)
{
  // We should allow a guess at the transition and emission matrices.
  double loglik = 0;
  double oldLoglik = 0;

  // Maximum iterations?
  size_t iterations = 1000;

  // Find length of all sequences and ensure they are the correct size.
  size_t totalLength = 0;
  for (size_t seq = 0; seq < dataSeq.size(); seq++)
  {
    totalLength += dataSeq[seq].n_cols;

    if (dataSeq[seq].n_rows != dimensionality)
      Log::Fatal << "HMM::Train(): data sequence " << seq << " has "
          << "dimensionality " << dataSeq[seq].n_rows << " (expected "
          << dimensionality << " dimensions)." << std::endl;
  }

  // These are used later for training of each distribution.  We initialize it
  // all now so we don't have to do any allocation later on.
  std::vector<arma::vec> emissionProb(transition.n_cols,
      arma::vec(totalLength));
  arma::mat emissionList(dimensionality, totalLength);

  // This should be the Baum-Welch algorithm (EM for HMM estimation). This
  // follows the procedure outlined in Elliot, Aggoun, and Moore's book "Hidden
  // Markov Models: Estimation and Control", pp. 36-40.
  for (size_t iter = 0; iter < iterations; iter++)
  {
    // Clear new transition matrix and emission probabilities.
    arma::vec newLogInitial(transition.n_rows);
    newLogInitial.fill(-std::numeric_limits<double>::infinity());
    arma::mat newLogTransition(transition.n_rows, transition.n_cols);
    newLogTransition.fill(-std::numeric_limits<double>::infinity());

    // Reset log likelihood.
    loglik = 0;

    // Sum over time.
    size_t sumTime = 0;

    // Loop over each sequence.
    for (size_t seq = 0; seq < dataSeq.size(); seq++)
    {
      arma::mat stateLogProb;
      arma::mat forwardLog;
      arma::mat backwardLog;
      arma::vec logScales;

      // Add the log-likelihood of this sequence.  This is the E-step.
      loglik += LogEstimate(dataSeq[seq], stateLogProb, forwardLog,
          backwardLog, logScales);

      // Add to estimate of initial probability for state j.
      for (size_t j = 0; j < transition.n_cols; ++j)
        newLogInitial[j] = math::LogAdd(newLogInitial[j], stateLogProb(j, 0));

      // Define a variable to store the value of log-probability for data.
      arma::mat logProbs(dataSeq[seq].n_cols, transition.n_rows);
      // Save the values of log-probability to logProbs.
      for (size_t i = 0; i < transition.n_rows; i++)
      {
        // Use advanced constructor for using logProbs directly.
        emission[i].LogProbability(dataSeq[seq], arma::vec(logProbs.colptr(i),
            logProbs.n_rows, false, true));
      }

      // Now re-estimate the parameters.  This is the M-step.
      //   pi_i = sum_d ((1 / P(seq[d])) sum_t (f(i, 0) b(i, 0))
      //   T_ij = sum_d ((1 / P(seq[d])) sum_t (f(i, t) T_ij E_i(seq[d][t]) b(i,
      //           t + 1)))
      //   E_ij = sum_d ((1 / P(seq[d])) sum_{t | seq[d][t] = j} f(i, t) b(i, t)
      // We store the new estimates in a different matrix.
      for (size_t t = 0; t < dataSeq[seq].n_cols; ++t)
      {
        for (size_t j = 0; j < transition.n_cols; ++j)
        {
          if (t < dataSeq[seq].n_cols - 1)
          {
            // Estimate of T_ij (probability of transition from state j to state
            // i).  We postpone multiplication of the old T_ij until later.
            for (size_t i = 0; i < transition.n_rows; i++)
            {
              newLogTransition(i, j) = math::LogAdd(newLogTransition(i, j),
                  forwardLog(j, t) + backwardLog(i, t + 1) + logProbs(t + 1, i)
                  - logScales[t + 1]);
            }
          }

          // Add to list of emission observations, for Distribution::Train().
          emissionList.col(sumTime) = dataSeq[seq].col(t);
          emissionProb[j][sumTime] = exp(stateLogProb(j, t));
        }
        sumTime++;
      }
    }

    if (std::abs(oldLoglik - loglik) < tolerance)
    {
      Log::Debug << "Converged after " << iter << " iterations." << std::endl;
      break;
    }

    oldLoglik = loglik;

    // Normalize the new initial probabilities.
    if (dataSeq.size() > 1)
      initial = exp(newLogInitial) / dataSeq.size();
    else
      initial = exp(newLogInitial);

    // Assign the new transition matrix.  We use %= (element-wise
    // multiplication) because every element of the new transition matrix must
    // still be multiplied by the old elements (this is the multiplication we
    // earlier postponed).
    transition %= exp(newLogTransition);

    // Now we normalize the transition matrix.
    for (size_t i = 0; i < transition.n_cols; i++)
    {
      const double sum = accu(transition.col(i));
      if (sum > 0.0)
        transition.col(i) /= sum;
      else
        transition.col(i).fill(1.0 / (double) transition.n_rows);
    }

    // Now estimate emission probabilities.
    for (size_t state = 0; state < transition.n_cols; state++)
      emission[state].Train(emissionList, emissionProb[state]);

    Log::Debug << "Iteration " << iter << ": log-likelihood " << loglik
        << "." << std::endl;
  }
  return loglik;
}

/**
 * Train the model using the given labeled observations; the transition and
 * emission matrices are directly estimated.
 */
template<typename Distribution>
void HMM<Distribution>::Train(const std::vector<arma::mat>& dataSeq,
                              const std::vector<arma::Row<size_t> >& stateSeq)
{
  // Simple error checking.
  if (dataSeq.size() != stateSeq.size())
  {
    Log::Fatal << "HMM::Train(): number of data sequences (" << dataSeq.size()
        << ") not equal to number of state sequences (" << stateSeq.size()
        << ")." << std::endl;
  }

  initial.zeros();
  transition.zeros();

  // Estimate the transition and emission matrices directly from the
  // observations.  The emission list holds the time indices for observations
  // from each state.
  std::vector<std::vector<std::pair<size_t, size_t> > >
      emissionList(transition.n_cols);
  for (size_t seq = 0; seq < dataSeq.size(); seq++)
  {
    // Simple error checking.
    if (dataSeq[seq].n_cols != stateSeq[seq].n_elem)
    {
      Log::Fatal << "HMM::Train(): number of observations ("
          << dataSeq[seq].n_cols << ") in sequence " << seq
          << " not equal to number of states (" << stateSeq[seq].n_cols
          << ") in sequence " << seq << "." << std::endl;
    }

    if (dataSeq[seq].n_rows != dimensionality)
    {
      Log::Fatal << "HMM::Train(): data sequence " << seq << " has "
          << "dimensionality " << dataSeq[seq].n_rows << " (expected "
          << dimensionality << " dimensions)." << std::endl;
    }

    // Loop over each observation in the sequence.  For estimation of the
    // transition matrix, we must ignore the last observation.
    initial[stateSeq[seq][0]]++;
    for (size_t t = 0; t < dataSeq[seq].n_cols - 1; t++)
    {
      transition(stateSeq[seq][t + 1], stateSeq[seq][t])++;
      emissionList[stateSeq[seq][t]].push_back(std::make_pair(seq, t));
    }

    // Last observation.
    emissionList[stateSeq[seq][stateSeq[seq].n_elem - 1]].push_back(
        std::make_pair(seq, stateSeq[seq].n_elem - 1));
  }

  // Normalize initial weights.
  initial /= accu(initial);

  // Normalize transition matrix.
  for (size_t col = 0; col < transition.n_cols; col++)
  {
    // If the transition probability sum is greater than 0 in this column, the
    // emission probability sum will also be greater than 0.  We want to avoid
    // division by 0.
    double sum = accu(transition.col(col));
    if (sum > 0)
      transition.col(col) /= sum;
  }

  // Estimate emission matrix.
  for (size_t state = 0; state < transition.n_cols; state++)
  {
    // Generate full sequence of observations for this state from the list of
    // emissions that are from this state.
    if (emissionList[state].size() > 0)
    {
      arma::mat emissions(dimensionality, emissionList[state].size());
      for (size_t i = 0; i < emissions.n_cols; i++)
      {
        emissions.col(i) = dataSeq[emissionList[state][i].first].col(
            emissionList[state][i].second);
      }

      emission[state].Train(emissions);
    }
    else
    {
      Log::Warn << "There are no observations in training data with hidden "
          << "state " << state << "!  The corresponding emission distribution "
          << "is likely to be meaningless." << std::endl;
    }
  }
}

/**
 * Estimate the probabilities of each hidden state at each time step for each
 * given data observation.
 */
template<typename Distribution>
double HMM<Distribution>::LogEstimate(const arma::mat& dataSeq,
                                      arma::mat& stateLogProb,
                                      arma::mat& forwardLogProb,
                                      arma::mat& backwardLogProb,
                                      arma::vec& logScales) const
{
  // First run the forward-backward algorithm.
  Forward(dataSeq, logScales, forwardLogProb);
  Backward(dataSeq, logScales, backwardLogProb);

  // Now assemble the state probability matrix based on the forward and backward
  // probabilities.
  stateLogProb = forwardLogProb + backwardLogProb;

  // Finally assemble the log-likelihood and return it.
  return accu(logScales);
}

/**
 * Estimate the probabilities of each hidden state at each time step for each
 * given data observation.
 */
template<typename Distribution>
double HMM<Distribution>::Estimate(const arma::mat& dataSeq,
                                   arma::mat& stateProb,
                                   arma::mat& forwardProb,
                                   arma::mat& backwardProb,
                                   arma::vec& scales) const
{
  arma::mat stateLogProb;
  arma::mat forwardLogProb;
  arma::mat backwardLogProb;
  arma::vec logScales;

  const double loglikelihood = LogEstimate(dataSeq, stateLogProb,
      forwardLogProb, backwardLogProb, logScales);

  stateProb = exp(stateLogProb);
  forwardProb = exp(forwardLogProb);
  backwardProb = exp(backwardLogProb);
  scales = exp(logScales);

  return loglikelihood;
}

/**
 * Estimate the probabilities of each hidden state at each time step for each
 * given data observation.
 */
template<typename Distribution>
double HMM<Distribution>::Estimate(const arma::mat& dataSeq,
                                   arma::mat& stateProb) const
{
  // We don't need to save these.
  arma::mat stateLogProb;
  arma::mat forwardLogProb;
  arma::mat backwardLogProb;
  arma::vec logScales;

  const double loglikelihood = LogEstimate(dataSeq, stateLogProb,
      forwardLogProb, backwardLogProb, logScales);

  stateProb = exp(stateLogProb);

  return loglikelihood;
}

/**
 * Generate a random data sequence of a given length.  The data sequence is
 * stored in the dataSequence parameter, and the state sequence is stored in
 * the stateSequence parameter.
 */
template<typename Distribution>
void HMM<Distribution>::Generate(const size_t length,
                                 arma::mat& dataSequence,
                                 arma::Row<size_t>& stateSequence,
                                 const size_t startState) const
{
  // Set vectors to the right size.
  stateSequence.set_size(length);
  dataSequence.set_size(dimensionality, length);

  // Set start state (default is 0).
  stateSequence[0] = startState;

  // Choose first emission state.
  double randValue = math::Random();

  // We just have to find where our random value sits in the probability
  // distribution of emissions for our starting state.
  dataSequence.col(0) = emission[startState].Random();

  // Now choose the states and emissions for the rest of the sequence.
  for (size_t t = 1; t < length; t++)
  {
    // First choose the hidden state.
    randValue = math::Random();

    // Now find where our random value sits in the probability distribution of
    // state changes.
    double probSum = 0;
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
    dataSequence.col(t) = emission[stateSequence[t]].Random();
  }
}

/**
 * Compute the most probable hidden state sequence for the given observation
 * using the Viterbi algorithm. Returns the log-likelihood of the most likely
 * sequence.
 */
template<typename Distribution>
double HMM<Distribution>::Predict(const arma::mat& dataSeq,
                                  arma::Row<size_t>& stateSeq) const
{
  // This is an implementation of the Viterbi algorithm for finding the most
  // probable sequence of states to produce the observed data sequence.  We
  // don't use log-likelihoods to save that little bit of time, but we'll
  // calculate the log-likelihood at the end of it all.
  stateSeq.set_size(dataSeq.n_cols);
  arma::mat logStateProb(transition.n_rows, dataSeq.n_cols);
  arma::mat stateSeqBack(transition.n_rows, dataSeq.n_cols);

  // Store the logs of the transposed transition matrix.  This is because we
  // will be using the rows of the transition matrix.
  arma::mat logTrans(log(trans(transition)));

  // The calculation of the first state is slightly different; the probability
  // of the first state being state j is the maximum probability that the state
  // came to be j from another state.
  logStateProb.col(0).zeros();
  for (size_t state = 0; state < transition.n_rows; state++)
  {
    logStateProb(state, 0) = log(initial[state]) +
        emission[state].LogProbability(dataSeq.unsafe_col(0));
    stateSeqBack(state, 0) = state;
  }

  // Store the best first state.
  arma::uword index;
  // Define a variable to store the value of log-probability for dataSeq.
  arma::mat logProbs(dataSeq.n_cols, transition.n_rows);

  // Save the values of log-probability to logProbs.
  for (size_t i = 0; i < transition.n_rows; i++)
  {
    // Use advanced constructor for using logProbs directly.
    emission[i].LogProbability(dataSeq, arma::vec(logProbs.colptr(i),
        logProbs.n_rows, false, true));
  }

  for (size_t t = 1; t < dataSeq.n_cols; t++)
  {
    // Assemble the state probability for this element.
    // Given that we are in state j, we use state with the highest probability
    // of being the previous state.
    for (size_t j = 0; j < transition.n_rows; j++)
    {
      arma::vec prob = logStateProb.col(t - 1) + logTrans.col(j);
      logStateProb(j, t) = prob.max(index) + logProbs(t, j);
      stateSeqBack(j, t) = index;
    }
  }

  // Backtrack to find the most probable state sequence.
  logStateProb.unsafe_col(dataSeq.n_cols - 1).max(index);
  stateSeq[dataSeq.n_cols - 1] = index;
  for (size_t t = 2; t <= dataSeq.n_cols; t++)
  {
    stateSeq[dataSeq.n_cols - t] =
        stateSeqBack(stateSeq[dataSeq.n_cols - t + 1], dataSeq.n_cols - t + 1);
  }

  return logStateProb(stateSeq(dataSeq.n_cols - 1), dataSeq.n_cols - 1);
}

/**
 * Compute the log-likelihood of the given data sequence.
 */
template<typename Distribution>
double HMM<Distribution>::LogLikelihood(const arma::mat& dataSeq) const
{
  arma::mat forwardLog;
  arma::vec logScales;

  Forward(dataSeq, logScales, forwardLog);

  // The log-likelihood is the log of the scales for each time step.
  return accu(logScales);
}

/**
 * HMM filtering.
 */
template<typename Distribution>
void HMM<Distribution>::Filter(const arma::mat& dataSeq,
                               arma::mat& filterSeq,
                               size_t ahead) const
{
  // First run the forward algorithm.
  arma::mat forwardLogProb;
  arma::vec logScales;
  Forward(dataSeq, logScales, forwardLogProb);

  arma::mat forwardProb = exp(forwardLogProb);

  // Propagate state ahead.
  if (ahead != 0)
    forwardProb = pow(transition, ahead) * forwardProb;

  // Compute expected emissions.
  // Will not work for distributions without a Mean() function.
  filterSeq.zeros(dimensionality, dataSeq.n_cols);
  for (size_t i = 0; i < emission.size(); i++)
    filterSeq += emission[i].Mean() * forwardProb.row(i);
}

/**
 * HMM smoothing.
 */
template<typename Distribution>
void HMM<Distribution>::Smooth(const arma::mat& dataSeq,
                               arma::mat& smoothSeq) const
{
  // First run the forward algorithm.
  arma::mat stateLogProb;
  arma::mat forwardLogProb;
  arma::mat backwardLogProb;
  arma::mat logScales;
  LogEstimate(dataSeq, stateLogProb, forwardLogProb, backwardLogProb,
      logScales);

  // Compute expected emissions.
  // Will not work for distributions without a Mean() function.
  smoothSeq.zeros(dimensionality, dataSeq.n_cols);
  for (size_t i = 0; i < emission.size(); i++)
    smoothSeq += emission[i].Mean() * exp(stateLogProb.row(i));
}

/**
 * The Forward procedure (part of the Forward-Backward algorithm).
 */
template<typename Distribution>
void HMM<Distribution>::Forward(const arma::mat& dataSeq,
                                arma::vec& logScales,
                                arma::mat& forwardLogProb) const
{
  // Our goal is to calculate the forward probabilities:
  //  P(X_k | o_{1:k}) for all possible states X_k, for each time point k.
  forwardLogProb.resize(transition.n_rows, dataSeq.n_cols);
  forwardLogProb.fill(-std::numeric_limits<double>::infinity());
  logScales.resize(dataSeq.n_cols);
  logScales.fill(-std::numeric_limits<double>::infinity());

  arma::mat logTrans = trans(log(transition));

  // Define a variable to store the value of log-probability for dataSeq.
  arma::mat logProbs(dataSeq.n_cols, transition.n_rows);

  // Save the values of log-probability to logProbs.
  for (size_t i = 0; i < transition.n_rows; i++)
  {
    // Use advanced constructor for using logProbs directly.
    emission[i].LogProbability(dataSeq, arma::vec(logProbs.colptr(i),
        logProbs.n_rows, false, true));
  }
  // The first entry in the forward algorithm uses the initial state
  // probabilities.  Note that MATLAB assumes that the starting state (at
  // t = -1) is state 0; this is not our assumption here.  To force that
  // behavior, you could append a single starting state to every single data
  // sequence and that should produce results in line with MATLAB.
  for (size_t state = 0; state < transition.n_rows; state++)
  {
    forwardLogProb(state, 0) = log(initial(state)) + logProbs(0, state);
  }

  // Then normalize the column.
  logScales[0] = math::AccuLog(forwardLogProb.col(0));
  if (std::isfinite(logScales[0]))
    forwardLogProb.col(0) -= logScales[0];

  // Now compute the probabilities for each successive observation.
  for (size_t t = 1; t < dataSeq.n_cols; t++)
  {
    for (size_t j = 0; j < transition.n_rows; j++)
    {
      // The forward probability of state j at time t is the sum over all states
      // of the probability of the previous state transitioning to the current
      // state and emitting the given observation.
      arma::vec tmp = forwardLogProb.col(t - 1) + logTrans.col(j);
      forwardLogProb(j, t) = math::AccuLog(tmp) + logProbs(t, j);
    }

    // Normalize probability.
    logScales[t] = math::AccuLog(forwardLogProb.col(t));
    if (std::isfinite(logScales[t]))
        forwardLogProb.col(t) -= logScales[t];
  }
}

template<typename Distribution>
void HMM<Distribution>::Backward(const arma::mat& dataSeq,
                                 const arma::vec& logScales,
                                 arma::mat& backwardLogProb) const
{
  // Our goal is to calculate the backward probabilities:
  //  P(X_k | o_{k + 1:T}) for all possible states X_k, for each time point k.
  backwardLogProb.resize(transition.n_rows, dataSeq.n_cols);
  backwardLogProb.fill(-std::numeric_limits<double>::infinity());
  arma::mat logTrans = log(transition);

  // The last element probability is 1.
  backwardLogProb.col(dataSeq.n_cols - 1).fill(0);

  // Define a variable to store the value of log-probability for dataSeq.
  arma::mat logProbs(dataSeq.n_cols, transition.n_rows);

  // Save the values of log-probability to logProbs.
  for (size_t i = 0; i < transition.n_rows; i++)
  {
    // Use advanced constructor for using logProbs directly.
    emission[i].LogProbability(dataSeq, arma::vec(logProbs.colptr(i),
        logProbs.n_rows, false, true));
  }

  // Now step backwards through all other observations.
  for (size_t t = dataSeq.n_cols - 2; t + 1 > 0; t--)
  {
    for (size_t j = 0; j < transition.n_rows; j++)
    {
      // The backward probability of state j at time t is the sum over all state
      // of the probability of the next state having been a transition from the
      // current state multiplied by the probability of each of those states
      // emitting the given observation.
      for (size_t state = 0; state < transition.n_rows; state++)
      {
        backwardLogProb(j, t) = math::LogAdd(backwardLogProb(j, t),
            logTrans(state, j) + backwardLogProb(state, t + 1) +
            logProbs(t + 1, state));
      }

      // Normalize by the weights from the forward algorithm.
      if (std::isfinite(logScales[t + 1]))
        backwardLogProb(j, t) -= logScales[t + 1];
    }
  }
}

//! Serialize the HMM.
template<typename Distribution>
template<typename Archive>
void HMM<Distribution>::serialize(Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(dimensionality);
  ar & BOOST_SERIALIZATION_NVP(tolerance);
  ar & BOOST_SERIALIZATION_NVP(transition);
  ar & BOOST_SERIALIZATION_NVP(initial);

  // Now serialize each emission.  If we are loading, we must resize the vector
  // of emissions correctly.
  if (Archive::is_loading::value)
    emission.resize(transition.n_rows);

  // Load the emissions; generate the correct name for each one.
    ar & BOOST_SERIALIZATION_NVP(emission);
}

} // namespace hmm
} // namespace mlpack

#endif
