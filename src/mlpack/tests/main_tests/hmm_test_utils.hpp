#ifndef MLPACK_TESTS_MAIN_TESTS_HMM_TEST_UTILS_HPP
#define MLPACK_TESTS_MAIN_TESTS_HMM_TEST_UTILS_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/hmm/hmm.hpp>

struct Init
{
  template<typename HMMType>
  static void Apply(HMMType& hmm, vector<mat>* trainSeq)
  {
    const size_t states = 2;

    // Create the initialized-to-zero model.
    Create(hmm, *trainSeq, states);

    // Initializing the emission distribution depends on the distribution.
    // Therefore we have to use the helper functions.
    RandomInitialize(hmm.Emission());
  }

  //! Helper function to create discrete HMM.
  static void Create(HMM<DiscreteDistribution>& hmm,
                     vector<mat>& trainSeq,
                     size_t states,
                     double tolerance = 1e-05)
  {
    // Maximum observation is necessary so we know how to train the discrete
    // distribution.
    arma::Col<size_t> maxEmissions(trainSeq[0].n_rows);
    maxEmissions.zeros();
    for (vector<mat>::iterator it = trainSeq.begin(); it != trainSeq.end();
         ++it)
    {
      arma::Col<size_t> maxSeqs =
          arma::conv_to<arma::Col<size_t>>::from(arma::max(*it, 1)) + 1;
      maxEmissions = arma::max(maxEmissions, maxSeqs);
    }

    hmm = HMM<DiscreteDistribution>(size_t(states),
        DiscreteDistribution(maxEmissions), tolerance);
  }

  static void Create(HMM<GaussianDistribution>& hmm,
                     vector<mat>& trainSeq,
                     size_t states,
                     double tolerance = 1e-05)
  {
    // Not implemented
    // Prevent unused parameter warning
    (void)hmm;
    (void)trainSeq;
    (void)states;
    (void)tolerance;
  }

  static void Create(HMM<GMM>& hmm,
                     vector<mat>& trainSeq,
                     size_t states,
                     double tolerance = 1e-05)
  {
    // Not implemented
    // Prevent unused parameter warning
    (void)hmm;
    (void)trainSeq;
    (void)states;
    (void)tolerance;
  }

  //! Helper function for discrete emission distributions.
  static void RandomInitialize(vector<DiscreteDistribution>& e)
  {
    for (size_t i = 0; i < e.size(); ++i)
    {
      e[i].Probabilities().randu();
      e[i].Probabilities() /= arma::accu(e[i].Probabilities());
    }
  }

  static void RandomInitialize(vector<GaussianDistribution>& e)
  {
    // Not implemented
    // Prevent unused parameter warning
    (void)e;
  }

  static void RandomInitialize(vector<GMM>& e)
  {
    // Not implemented
    // Prevent unused parameter warning
    (void)e;
  }
};

struct Train
{
  template<typename HMMType>
  static void Apply(HMMType& hmm, vector<arma::mat>* trainSeq)
  {
    // For now, perform unsupervised (Baum-Welch) training
    hmm.Train(*trainSeq);
  }
};

#endif
