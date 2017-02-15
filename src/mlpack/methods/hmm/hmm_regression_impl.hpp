/**
 * @file hmm_regression_impl.hpp
 * @author Ryan Curtin
 * @author Tran Quoc Long
 * @author Michael Fox
 *
 * Implementation of HMMRegression class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HMM_HMM_REGRESSION_IMPL_HPP
#define MLPACK_METHODS_HMM_HMM_REGRESSION_IMPL_HPP

// Just in case...
#include "hmm_regression.hpp"

namespace mlpack {
namespace hmm {

void HMMRegression::Train(const std::vector<arma::mat>& predictors,
                          const std::vector<arma::vec>& responses)
{
  std::vector<arma::mat> dataSeq;
  StackData(predictors, responses, dataSeq);
  this->HMM::Train(dataSeq);
}

void HMMRegression::Train(const std::vector<arma::mat>& predictors,
                          const std::vector<arma::vec>& responses,
                          const std::vector<arma::Row<size_t> >& stateSeq)
{
  std::vector<arma::mat> dataSeq;
  StackData(predictors, responses, dataSeq);
  this->HMM::Train(dataSeq, stateSeq);
}

/**
 * Estimate the probabilities of each hidden state at each time step for each
 * given data observation.
 */
double HMMRegression::Estimate(const arma::mat& predictors,
                               const arma::vec& responses,
                               arma::mat& stateProb,
                               arma::mat& forwardProb,
                               arma::mat& backwardProb,
                               arma::vec& scales) const
{
  arma::mat dataSeq;
  StackData(predictors, responses, dataSeq);
  return this->HMM::Estimate(dataSeq, stateProb, forwardProb,
      backwardProb, scales);
}

/**
 * Estimate the probabilities of each hidden state at each time step for each
 * given data observation.
 */
double HMMRegression::Estimate(const arma::mat& predictors,
                               const arma::vec& responses,
                               arma::mat& stateProb) const
{
  arma::mat dataSeq;
  StackData(predictors, responses, dataSeq);
  return this->HMM::Estimate(dataSeq, stateProb);
}

/**
 * Compute the most probable hidden state sequence for the given observation
 * using the Viterbi algorithm. Returns the log-likelihood of the most likely
 * sequence.
 */
double HMMRegression::Predict(const arma::mat& predictors,
                              const arma::vec& responses,
                              arma::Row<size_t>& stateSeq) const
{
  arma::mat dataSeq;
  StackData(predictors, responses, dataSeq);
  return this->HMM::Predict(dataSeq, stateSeq);
}

/**
 * Compute the log-likelihood of the given data sequence.
 */
double HMMRegression::LogLikelihood(const arma::mat& predictors,
                                    const arma::vec& responses) const
{
  arma::mat dataSeq;
  StackData(predictors, responses, dataSeq);
  return this->HMM::LogLikelihood(dataSeq);
}

/**
 * HMMRegression filtering.
 */
void HMMRegression::Filter(const arma::mat& predictors,
                           const arma::vec& responses,
                           arma::vec& filterSeq,
                           size_t ahead) const
{
  // First run the forward algorithm
  arma::mat forwardProb;
  arma::vec scales;
  Forward(predictors, responses, scales, forwardProb);

  // Propagate state, predictors ahead
  if (ahead != 0) {
    forwardProb = pow(transition, ahead)*forwardProb;
    forwardProb = forwardProb.cols(0, forwardProb.n_cols-ahead-1);
  }

  // Compute expected emissions.
  filterSeq.resize(responses.n_elem - ahead);
  filterSeq.zeros();
  arma::vec nextSeq;
  for(size_t i = 0; i < emission.size(); i++)
  {
    emission[i].Predict(predictors.cols(ahead, predictors.n_cols-1), nextSeq);
    filterSeq = filterSeq + nextSeq%(forwardProb.row(i).t());
  }
}

/**
 * HMM smoothing.
 */
void HMMRegression::Smooth(const arma::mat& predictors,
                           const arma::vec& responses,
                           arma::vec& smoothSeq) const
{
  // First run the forward algorithm
  arma::mat stateProb;
  Estimate(predictors, responses, stateProb);

  // Compute expected emissions.
  smoothSeq.resize(responses.n_elem);
  smoothSeq.zeros();
  arma::vec nextSeq;
  for(size_t i = 0; i < emission.size(); i++)
  {
    emission[i].Predict(predictors, nextSeq);
    smoothSeq = smoothSeq + nextSeq%(stateProb.row(i).t());
  }

}

/**
 * The Forward procedure (part of the Forward-Backward algorithm).
 */
void HMMRegression::Forward(const arma::mat& predictors,
                            const arma::vec& responses,
                            arma::vec& scales,
                            arma::mat& forwardProb) const
{
  arma::mat dataSeq;
  StackData(predictors, responses, dataSeq);
  this->HMM::Forward(dataSeq, scales, forwardProb);
}


void HMMRegression::Backward(const arma::mat& predictors,
                             const arma::vec& responses,
                             const arma::vec& scales,
                             arma::mat& backwardProb) const
{
  arma::mat dataSeq;
  StackData(predictors, responses, dataSeq);
  this->HMM::Backward(dataSeq, scales, backwardProb);
}

void HMMRegression::StackData(const std::vector<arma::mat>& predictors,
                              const std::vector<arma::vec>& responses,
                              std::vector<arma::mat>& dataSeq) const
{
  arma::mat nextSeq;
  for(size_t i = 0; i < predictors.size(); i++)
  {
    nextSeq = predictors[i];
    nextSeq.insert_rows(0, responses[i].t());
    dataSeq.push_back(nextSeq);
  }
}

void HMMRegression::StackData(const arma::mat& predictors,
                              const arma::vec& responses,
                              arma::mat& dataSeq) const
{
  dataSeq = predictors;
  dataSeq.insert_rows(0, responses.t());
}

} // namespace hmm
} // namespace mlpack

#endif
