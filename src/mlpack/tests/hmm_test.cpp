/**
 * @file hmm_test.cpp
 *
 * Test file for HMMs.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/hmm/hmm.hpp>
#include <mlpack/methods/gmm/gmm.hpp>
#include <mlpack/methods/gmm/diagonal_gmm.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::hmm;
using namespace mlpack::distribution;
using namespace mlpack::gmm;

BOOST_AUTO_TEST_SUITE(HMMTest);

/**
 * We will use the simple case proposed by Russell and Norvig in Artificial
 * Intelligence: A Modern Approach, 2nd Edition, around p.549.
 */
BOOST_AUTO_TEST_CASE(SimpleDiscreteHMMTestViterbi)
{
  // We have two hidden states: rain/dry.  Two emission states: umbrella/no
  // umbrella.
  // In this example, the transition matrix is
  //  rain  dry
  // [[0.7 0.3]  rain
  //  [0.3 0.7]] dry
  // and the emission probability is
  //  rain dry
  // [[0.9 0.2]  umbrella
  //  [0.1 0.8]] no umbrella
  arma::vec initial("1 0"); // Default MATLAB initial states.
  arma::mat transition("0.7 0.3; 0.3 0.7");
  std::vector<DiscreteDistribution> emission(2);
  emission[0] = DiscreteDistribution(std::vector<arma::vec>{"0.9 0.1"});
  emission[1] = DiscreteDistribution(std::vector<arma::vec>{"0.2 0.8"});

  HMM<DiscreteDistribution> hmm(initial, transition, emission);

  // Now let's take a sequence and find what the most likely state is.
  // We'll use the sequence [U U N U U] (U = umbrella, N = no umbrella) like on
  // p. 547.
  arma::mat observation = "0 0 1 0 0";
  arma::Row<size_t> states;
  hmm.Predict(observation, states);

  // Check each state.
  BOOST_REQUIRE_EQUAL(states[0], 0); // Rain.
  BOOST_REQUIRE_EQUAL(states[1], 0); // Rain.
  BOOST_REQUIRE_EQUAL(states[2], 1); // No rain.
  BOOST_REQUIRE_EQUAL(states[3], 0); // Rain.
  BOOST_REQUIRE_EQUAL(states[4], 0); // Rain.
}

/**
 * This example is from Borodovsky & Ekisheva, p. 80-81.  It is just slightly
 * more complex.
 */
BOOST_AUTO_TEST_CASE(BorodovskyHMMTestViterbi)
{
  // Equally probable initial states.
  arma::vec initial(3);
  initial.fill(1.0 / 3.0);

  // Two hidden states: H (high GC content) and L (low GC content), as well as a
  // start state.
  arma::mat transition("0.0 0.0 0.0;"
                       "0.5 0.5 0.4;"
                       "0.5 0.5 0.6");
  // Four emission states: A, C, G, T.  Start state doesn't emit...
  std::vector<DiscreteDistribution> emission(3);
  emission[0] = DiscreteDistribution(
      std::vector<arma::vec>{"0.25 0.25 0.25 0.25"});
  emission[1] = DiscreteDistribution(
      std::vector<arma::vec>{"0.20 0.30 0.30 0.20"});
  emission[2] = DiscreteDistribution(
      std::vector<arma::vec>{"0.30 0.20 0.20 0.30"});

  HMM<DiscreteDistribution> hmm(initial, transition, emission);

  // GGCACTGAA.
  arma::mat observation("2 2 1 0 1 3 2 0 0");
  arma::Row<size_t> states;
  hmm.Predict(observation, states);

  // Most probable path is HHHLLLLLL.
  BOOST_REQUIRE_EQUAL(states[0], 1);
  BOOST_REQUIRE_EQUAL(states[1], 1);
  BOOST_REQUIRE_EQUAL(states[2], 1);
  BOOST_REQUIRE_EQUAL(states[3], 2);
  // This could actually be one of two states (equal probability).
  BOOST_REQUIRE((states[4] == 1) || (states[4] == 2));
  BOOST_REQUIRE_EQUAL(states[5], 2);
  // This could also be one of two states.
  BOOST_REQUIRE((states[6] == 1) || (states[6] == 2));
  BOOST_REQUIRE_EQUAL(states[7], 2);
  BOOST_REQUIRE_EQUAL(states[8], 2);
}

/**
 * Ensure that the forward-backward algorithm is correct.
 */
BOOST_AUTO_TEST_CASE(ForwardBackwardTwoState)
{
  arma::mat obs("3 3 2 1 1 1 1 3 3 1");

  // The values used for the initial distribution here don't entirely make
  // sense.  I am not sure how the output came from hmmdecode(), and the
  // documentation below doesn't completely say.  It seems like maybe the
  // transition matrix needs to be transposed and the results recalculated, but
  // I am not certain.
  arma::vec initial("0.1 0.4");
  arma::mat transition("0.1 0.9; 0.4 0.6");
  std::vector<DiscreteDistribution> emis(2);
  emis[0] = DiscreteDistribution(std::vector<arma::vec>{"0.85 0.15 0.00 0.00"});
  emis[1] = DiscreteDistribution(std::vector<arma::vec>{"0.00 0.00 0.50 0.50"});

  HMM<DiscreteDistribution> hmm(initial, transition, emis);

  // Now check we are getting the same results as MATLAB for this sequence.
  arma::mat stateProb;
  arma::mat forwardProb;
  arma::mat backwardProb;
  arma::vec scales;

  const double log = hmm.Estimate(obs, stateProb, forwardProb, backwardProb,
      scales);

  // All values obtained from MATLAB hmmdecode().
  BOOST_REQUIRE_CLOSE(log, -23.4349, 1e-3);

  BOOST_REQUIRE_SMALL(stateProb(0, 0), 1e-5);
  BOOST_REQUIRE_CLOSE(stateProb(1, 0), 1.0, 1e-5);
  BOOST_REQUIRE_SMALL(stateProb(0, 1), 1e-5);
  BOOST_REQUIRE_CLOSE(stateProb(1, 1), 1.0, 1e-5);
  BOOST_REQUIRE_SMALL(stateProb(0, 2), 1e-5);
  BOOST_REQUIRE_CLOSE(stateProb(1, 2), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(stateProb(0, 3), 1.0, 1e-5);
  BOOST_REQUIRE_SMALL(stateProb(1, 3), 1e-5);
  BOOST_REQUIRE_CLOSE(stateProb(0, 4), 1.0, 1e-5);
  BOOST_REQUIRE_SMALL(stateProb(1, 4), 1e-5);
  BOOST_REQUIRE_CLOSE(stateProb(0, 5), 1.0, 1e-5);
  BOOST_REQUIRE_SMALL(stateProb(1, 5), 1e-5);
  BOOST_REQUIRE_CLOSE(stateProb(0, 6), 1.0, 1e-5);
  BOOST_REQUIRE_SMALL(stateProb(1, 6), 1e-5);
  BOOST_REQUIRE_SMALL(stateProb(0, 7), 1e-5);
  BOOST_REQUIRE_CLOSE(stateProb(1, 7), 1.0, 1e-5);
  BOOST_REQUIRE_SMALL(stateProb(0, 8), 1e-5);
  BOOST_REQUIRE_CLOSE(stateProb(1, 8), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(stateProb(0, 9), 1.0, 1e-5);
  BOOST_REQUIRE_SMALL(stateProb(1, 9), 1e-5);
}

/**
 * In this example we try to estimate the transmission and emission matrices
 * based on some observations.  We use the simplest possible model.
 */
BOOST_AUTO_TEST_CASE(SimplestBaumWelchDiscreteHMM)
{
  // Don't yet require a useful distribution.  1 state, 1 emission.
  HMM<DiscreteDistribution> hmm(1, DiscreteDistribution(1));

  std::vector<arma::mat> observations;
  // Different lengths for each observation sequence.
  observations.push_back("0 0 0 0 0 0 0 0"); // 8 zeros.
  observations.push_back("0 0 0 0 0 0 0"); // 7 zeros.
  observations.push_back("0 0 0 0 0 0 0 0 0 0 0 0"); // 12 zeros.
  observations.push_back("0 0 0 0 0 0 0 0 0 0"); // 10 zeros.

  hmm.Train(observations);

  BOOST_REQUIRE_CLOSE(hmm.Initial()[0], 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(hmm.Emission()[0].Probability("0"), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(hmm.Transition()(0, 0), 1.0, 1e-5);
}

/**
 * A slightly more complex model to estimate.
 */
BOOST_AUTO_TEST_CASE(SimpleBaumWelchDiscreteHMM)
{
  HMM<DiscreteDistribution> hmm(1, 2); // 1 state, 2 emissions.
  // Randomize the emission matrix.
  hmm.Emission()[0].Probabilities() = arma::randu<arma::vec>(2);
  hmm.Emission()[0].Probabilities() /= accu(hmm.Emission()[0].Probabilities());

  // P(each emission) = 0.5.
  // I've been careful to make P(first emission = 0) = P(first emission = 1).
  std::vector<arma::mat> observations;
  observations.push_back("0 1 0 1 0 1 0 1 0 1 0 1");
  observations.push_back("0 0 0 0 0 0 1 1 1 1 1 1");
  observations.push_back("1 1 1 1 1 1 0 0 0 0 0 0");
  observations.push_back("1 1 1 0 0 0 1 1 1 0 0 0");
  observations.push_back("0 0 1 1 0 0 0 0 1 1 1 1");
  observations.push_back("1 1 1 0 0 0 1 1 1 0 0 0");
  observations.push_back("0 1 0 1 0 1 0 1 0 1 0 1");
  observations.push_back("0 0 0 0 0 0 1 1 1 1 1 1");
  observations.push_back("1 1 1 1 1 0 1 0 0 0 0 0");
  observations.push_back("1 1 1 0 0 1 0 1 1 0 0 0");
  observations.push_back("0 0 1 1 0 0 0 1 0 1 1 1");
  observations.push_back("1 1 1 0 0 1 0 1 1 0 0 0");

  hmm.Train(observations);

  BOOST_REQUIRE_CLOSE(hmm.Emission()[0].Probability("0"), 0.5, 1e-5);
  BOOST_REQUIRE_CLOSE(hmm.Emission()[0].Probability("1"), 0.5, 1e-5);
  BOOST_REQUIRE_CLOSE(hmm.Transition()(0, 0), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(hmm.Initial()[0], 1.0, 1e-5);
}

/**
 * Increasing complexity, but still simple; 4 emissions, 2 states; the state can
 * be determined directly by the emission.
 */
BOOST_AUTO_TEST_CASE(SimpleBaumWelchDiscreteHMM_2)
{
  HMM<DiscreteDistribution> hmm(2, DiscreteDistribution(4));

  // A little bit of obfuscation to the solution.
  hmm.Transition() = arma::mat("0.1 0.4; 0.9 0.6");
  hmm.Emission()[0].Probabilities() = "0.85 0.15 0.00 0.00";
  hmm.Emission()[1].Probabilities() = "0.00 0.00 0.50 0.50";

  // True emission matrix:
  //  [[0.4 0  ]
  //   [0.6 0  ]
  //   [0   0.2]
  //   [0   0.8]]

  // True transmission matrix:
  //  [[0.5 0.5]
  //   [0.5 0.5]]

  // Generate observations randomly by hand.  This is kinda ugly, but it works.
  std::vector<arma::mat> observations;
  size_t obsNum = 250; // Number of observations.
  size_t obsLen = 500; // Number of elements in each observation.
  size_t stateZeroStarts = 0; // Number of times we start in state 0.
  for (size_t i = 0; i < obsNum; i++)
  {
    arma::mat observation(1, obsLen);

    size_t state = 0;
    size_t emission = 0;

    for (size_t obs = 0; obs < obsLen; obs++)
    {
      // See if state changed.
      double r = math::Random();

      if (r <= 0.5)
      {
        if (obs == 0)
          ++stateZeroStarts;
        state = 0;
      }
      else
      {
        state = 1;
      }

      // Now set the observation.
      r = math::Random();

      switch (state)
      {
        // case 0 is not possible.
        case 0:
          if (r <= 0.4)
            emission = 0;
          else
            emission = 1;
          break;
        case 1:
          if (r <= 0.2)
            emission = 2;
          else
            emission = 3;
          break;
      }

      observation(0, obs) = emission;
    }

    observations.push_back(observation);
  }

  hmm.Train(observations);

  // Calculate true probability of class 0 at the start.
  double prob = double(stateZeroStarts) / observations.size();

  // Only require 2.5% tolerance, because this is a little fuzzier.
  BOOST_REQUIRE_CLOSE(hmm.Initial()[0], prob, 2.5);
  BOOST_REQUIRE_CLOSE(hmm.Initial()[1], 1.0 - prob, 2.5);

  BOOST_REQUIRE_CLOSE(hmm.Transition()(0, 0), 0.5, 2.5);
  BOOST_REQUIRE_CLOSE(hmm.Transition()(1, 0), 0.5, 2.5);
  BOOST_REQUIRE_CLOSE(hmm.Transition()(0, 1), 0.5, 2.5);
  BOOST_REQUIRE_CLOSE(hmm.Transition()(1, 1), 0.5, 2.5);

  BOOST_REQUIRE_CLOSE(hmm.Emission()[0].Probability("0"), 0.4, 4.0);
  BOOST_REQUIRE_CLOSE(hmm.Emission()[0].Probability("1"), 0.6, 4.0);
  BOOST_REQUIRE_SMALL(hmm.Emission()[0].Probability("2"), 2.5);
  BOOST_REQUIRE_SMALL(hmm.Emission()[0].Probability("3"), 2.5);
  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Probability("0"), 2.5);
  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Probability("1"), 2.5);
  BOOST_REQUIRE_CLOSE(hmm.Emission()[1].Probability("2"), 0.2, 4.0);
  BOOST_REQUIRE_CLOSE(hmm.Emission()[1].Probability("3"), 0.8, 4.0);
}

BOOST_AUTO_TEST_CASE(DiscreteHMMLabeledTrainTest)
{
  // Generate a random Markov model with 3 hidden states and 6 observations.
  arma::mat transition;
  std::vector<DiscreteDistribution> emission(3);

  transition.randu(3, 3);
  emission[0].Probabilities() = arma::randu<arma::vec>(6);
  emission[0].Probabilities() /= accu(emission[0].Probabilities());
  emission[1].Probabilities() = arma::randu<arma::vec>(6);
  emission[1].Probabilities() /= accu(emission[1].Probabilities());
  emission[2].Probabilities() = arma::randu<arma::vec>(6);
  emission[2].Probabilities() /= accu(emission[2].Probabilities());

  // Normalize so they we have a correct transition matrix.
  for (size_t col = 0; col < 3; col++)
    transition.col(col) /= accu(transition.col(col));

  // Now generate sequences.
  size_t obsNum = 250;
  size_t obsLen = 800;

  std::vector<arma::mat> observations(obsNum);
  std::vector<arma::Row<size_t> > states(obsNum);

  for (size_t n = 0; n < obsNum; n++)
  {
    observations[n].set_size(1, obsLen);
    states[n].set_size(obsLen);

    // Random starting state.
    states[n][0] = math::RandInt(3);

    // Random starting observation.
    observations[n].col(0) = emission[states[n][0]].Random();

    // Now the rest of the observations.
    for (size_t t = 1; t < obsLen; t++)
    {
      // Choose random number for state transition.
      double state = math::Random();

      // Decide next state.
      double sumProb = 0;
      for (size_t st = 0; st < 3; st++)
      {
        sumProb += transition(st, states[n][t - 1]);
        if (sumProb >= state)
        {
          states[n][t] = st;
          break;
        }
      }

      // Decide observation.
      observations[n].col(t) = emission[states[n][t]].Random();
    }
  }

  // Now that our data is generated, we give the HMM the labeled data to train
  // on.
  HMM<DiscreteDistribution> hmm(3, DiscreteDistribution(6));

  hmm.Train(observations, states);

  // Make sure the initial weights are fine.  They should be equal (or close).
  arma::vec initial(3);
  initial.fill(1.0 / 3.0);
  BOOST_REQUIRE_LT(arma::norm(hmm.Initial() - initial), 0.2);

  // Check that the transition matrix is close.
  BOOST_REQUIRE_LT(arma::norm(hmm.Transition() - transition), 0.1);

  for (size_t col = 0; col < hmm.Emission().size(); col++)
  {
    for (size_t row = 0; row < hmm.Emission()[col].Probabilities().n_elem;
        row++)
    {
      arma::vec obs(1);
      obs[0] = row;
      BOOST_REQUIRE_SMALL(hmm.Emission()[col].Probability(obs) -
          emission[col].Probability(obs), 0.07);
    }
  }
}

/**
 * Make sure the Generate() function works for a uniformly distributed HMM;
 * we'll take many samples just to make sure.
 */
BOOST_AUTO_TEST_CASE(DiscreteHMMSimpleGenerateTest)
{
  // Very simple HMM.  4 emissions with equal probability and 2 states with
  // equal probability.
  HMM<DiscreteDistribution> hmm(2, DiscreteDistribution(4));
  hmm.Initial() = arma::ones<arma::vec>(2) / 2.0;
  hmm.Transition() = arma::ones<arma::mat>(2, 2) / 2.0;

  // Now generate a really, really long sequence.
  arma::mat dataSeq;
  arma::Row<size_t> stateSeq;

  hmm.Generate(100000, dataSeq, stateSeq);

  // Now find the empirical probabilities of each state.
  arma::vec emissionProb(4);
  arma::vec stateProb(2);
  emissionProb.zeros();
  stateProb.zeros();
  for (size_t i = 0; i < 100000; i++)
  {
    emissionProb[(size_t) dataSeq.col(i)[0] + 0.5]++;
    stateProb[stateSeq[i]]++;
  }

  // Normalize so these are probabilities.
  emissionProb /= accu(emissionProb);
  stateProb /= accu(stateProb);

  // Now check that the probabilities are right.  3% tolerance.
  BOOST_REQUIRE_CLOSE(emissionProb[0], 0.25, 3.0);
  BOOST_REQUIRE_CLOSE(emissionProb[1], 0.25, 3.0);
  BOOST_REQUIRE_CLOSE(emissionProb[2], 0.25, 3.0);
  BOOST_REQUIRE_CLOSE(emissionProb[3], 0.25, 3.0);

  BOOST_REQUIRE_CLOSE(stateProb[0], 0.50, 3.0);
  BOOST_REQUIRE_CLOSE(stateProb[1], 0.50, 3.0);
}

/**
 * More complex test for Generate().
 */
BOOST_AUTO_TEST_CASE(DiscreteHMMGenerateTest)
{
  // 6 emissions, 4 states.  Random transition and emission probability.
  arma::vec initial("1 0 0 0");
  arma::mat transition(4, 4);
  std::vector<DiscreteDistribution> emission(4);
  emission[0].Probabilities() = arma::randu<arma::vec>(6);
  emission[0].Probabilities() /= accu(emission[0].Probabilities());
  emission[1].Probabilities() = arma::randu<arma::vec>(6);
  emission[1].Probabilities() /= accu(emission[1].Probabilities());
  emission[2].Probabilities() = arma::randu<arma::vec>(6);
  emission[2].Probabilities() /= accu(emission[2].Probabilities());
  emission[3].Probabilities() = arma::randu<arma::vec>(6);
  emission[3].Probabilities() /= accu(emission[3].Probabilities());

  transition.randu();

  // Normalize matrix.
  for (size_t col = 0; col < 4; col++)
    transition.col(col) /= accu(transition.col(col));

  // Create HMM object.
  HMM<DiscreteDistribution> hmm(initial, transition, emission);

  // We'll create a bunch of sequences.
  int numSeq = 400;
  int numObs = 3000;
  std::vector<arma::mat> sequences(numSeq);
  std::vector<arma::Row<size_t> > states(numSeq);
  for (int i = 0; i < numSeq; i++)
  {
    // Random starting state.
    size_t startState = math::RandInt(4);

    hmm.Generate(numObs, sequences[i], states[i], startState);
  }

  // Now we will calculate the full probabilities.
  HMM<DiscreteDistribution> hmm2(4, 6);
  hmm2.Train(sequences, states);

  // Check that training gives the same result.
  BOOST_REQUIRE_LT(arma::norm(hmm.Transition() - hmm2.Transition()), 0.02);

  for (size_t row = 0; row < 6; row++)
  {
    arma::vec obs(1);
    obs[0] = row;
    for (size_t col = 0; col < 4; col++)
    {
      BOOST_REQUIRE_SMALL(hmm.Emission()[col].Probability(obs) -
          hmm2.Emission()[col].Probability(obs), 0.02);
    }
  }
}

BOOST_AUTO_TEST_CASE(DiscreteHMMLogLikelihoodTest)
{
  // Create a simple HMM with three states and four emissions.
  arma::vec initial("0.5 0.2 0.3"); // Default MATLAB initial states.
  arma::mat transition("0.5 0.0 0.1;"
                       "0.2 0.6 0.2;"
                       "0.3 0.4 0.7");
  std::vector<DiscreteDistribution> emission(3);
  emission[0].Probabilities() = "0.75 0.25 0.00 0.00";
  emission[1].Probabilities() = "0.00 0.25 0.25 0.50";
  emission[2].Probabilities() = "0.10 0.40 0.40 0.10";

  HMM<DiscreteDistribution> hmm(initial, transition, emission);

  // Now generate some sequences and check that the log-likelihood is the same
  // as MATLAB gives for this HMM.
  BOOST_REQUIRE_CLOSE(hmm.LogLikelihood("0 1 2 3"), -4.9887223949, 1e-5);
  BOOST_REQUIRE_CLOSE(hmm.LogLikelihood("1 2 0 0"), -6.0288487077, 1e-5);
  BOOST_REQUIRE_CLOSE(hmm.LogLikelihood("3 3 3 3"), -5.5544000018, 1e-5);
  BOOST_REQUIRE_CLOSE(hmm.LogLikelihood("0 2 2 1 2 3 0 0 1 3 1 0 0 3 1 2 2"),
      -24.51556128368, 1e-5);
}

/**
 * A simple test to make sure HMMs with Gaussian output distributions work.
 */
BOOST_AUTO_TEST_CASE(GaussianHMMSimpleTest)
{
  // We'll have two Gaussians, far away from each other, one corresponding to
  // each state.
  //  E(0) ~ N([ 5.0  5.0], eye(2)).
  //  E(1) ~ N([-5.0 -5.0], eye(2)).
  // The transition matrix is simple:
  //  T = [[0.75 0.25]
  //       [0.25 0.75]]
  GaussianDistribution g1("5.0 5.0", "1.0 0.0; 0.0 1.0");
  GaussianDistribution g2("-5.0 -5.0", "1.0 0.0; 0.0 1.0");

  arma::vec initial("1 0"); // Default MATLAB initial states.
  arma::mat transition("0.75 0.25; 0.25 0.75");

  std::vector<GaussianDistribution> emission;
  emission.push_back(g1);
  emission.push_back(g2);

  HMM<GaussianDistribution> hmm(initial, transition, emission);

  // Now, generate some sequences.
  arma::mat observations(2, 1000);
  arma::Row<size_t> classes(1000);

  // 1000-observations sequence.
  classes[0] = 0;
  observations.col(0) = g1.Random();
  for (size_t i = 1; i < 1000; i++)
  {
    double randValue = math::Random();

    if (randValue > 0.75) // Then we change state.
      classes[i] = (classes[i - 1] + 1) % 2;
    else
      classes[i] = classes[i - 1];

    if (classes[i] == 0)
      observations.col(i) = g1.Random();
    else
      observations.col(i) = g2.Random();
  }

  // Now predict the sequence.
  arma::Row<size_t> predictedClasses;
  arma::mat stateProb;

  hmm.Predict(observations, predictedClasses);
  hmm.Estimate(observations, stateProb);

  // Check that each prediction is right.
  for (size_t i = 0; i < 1000; i++)
  {
    BOOST_REQUIRE_EQUAL(predictedClasses[i], classes[i]);

    // The probability of the wrong class should be infinitesimal.
    BOOST_REQUIRE_SMALL(stateProb((classes[i] + 1) % 2, i), 0.001);
  }
}

/**
 * Ensure that Gaussian HMMs can be trained properly, for the labeled training
 * case and also for the unlabeled training case.
 */
BOOST_AUTO_TEST_CASE(GaussianHMMTrainTest)
{
  // Four emission Gaussians and three internal states.  The goal is to estimate
  // the transition matrix correctly, and each distribution correctly.
  std::vector<GaussianDistribution> emission;
  emission.push_back(GaussianDistribution("0.0 0.0 0.0", "1.0 0.2 0.2;"
                                                         "0.2 1.5 0.0;"
                                                         "0.2 0.0 1.1"));
  emission.push_back(GaussianDistribution("2.0 1.0 5.0", "0.7 0.3 0.0;"
                                                         "0.3 2.6 0.0;"
                                                         "0.0 0.0 1.0"));
  emission.push_back(GaussianDistribution("5.0 0.0 0.5", "1.0 0.0 0.0;"
                                                         "0.0 1.0 0.0;"
                                                         "0.0 0.0 1.0"));

  arma::mat transition("0.3 0.5 0.7;"
                       "0.3 0.4 0.1;"
                       "0.4 0.1 0.2");

  // Now generate observations.
  std::vector<arma::mat> observations(100);
  std::vector<arma::Row<size_t> > states(100);

  for (size_t obs = 0; obs < 100; obs++)
  {
    observations[obs].set_size(3, 1000);
    states[obs].set_size(1000);

    // Always start in state zero.
    states[obs][0] = 0;
    observations[obs].col(0) = emission[0].Random();

    for (size_t t = 1; t < 1000; t++)
    {
      // Choose the state.
      double randValue = math::Random();
      double probSum = 0;
      for (size_t state = 0; state < 3; state++)
      {
        probSum += transition(state, states[obs][t - 1]);
        if (probSum >= randValue)
        {
          states[obs][t] = state;
          break;
        }
      }

      // Now choose the emission.
      observations[obs].col(t) = emission[states[obs][t]].Random();
    }
  }

  // Now that the data is generated, train the HMM.
  HMM<GaussianDistribution> hmm(3, GaussianDistribution(3));

  hmm.Train(observations, states);

  // Check initial weights.
  BOOST_REQUIRE_CLOSE(hmm.Initial()[0], 1.0, 1e-5);
  BOOST_REQUIRE_SMALL(hmm.Initial()[1], 1e-3);
  BOOST_REQUIRE_SMALL(hmm.Initial()[2], 1e-3);

  // We use a tolerance of 0.05 for the transition matrices.
  // Check that the transition matrix is correct.
  BOOST_REQUIRE_LT(arma::norm(hmm.Transition() - transition), 0.05);

  // Check that each distribution is correct.
  for (size_t dist = 0; dist < 3; dist++)
  {
    BOOST_REQUIRE_LT(arma::norm(hmm.Emission()[dist].Mean() -
        emission[dist].Mean()), 0.05);
    BOOST_REQUIRE_LT(arma::norm(hmm.Emission()[dist].Covariance() -
        emission[dist].Covariance()), 0.1);
  }

  // Now let's try it all again, but this time, unlabeled.  Everything will fail
  // if we don't have a decent guess at the Gaussians, so we'll take a "poor"
  // guess at it ourselves.  I won't use K-Means because we can't afford to add
  // the instability of that to our test.  We'll leave the covariances as the
  // identity.
  HMM<GaussianDistribution> hmm2(3, GaussianDistribution(3));
  hmm2.Emission()[0].Mean() = "0.3 -0.2 0.1"; // Actual: [0 0 0].
  hmm2.Emission()[1].Mean() = "1.0 1.4 3.2";  // Actual: [2 1 5].
  hmm2.Emission()[2].Mean() = "3.1 -0.2 6.1"; // Actual: [5 0 5].

  // We'll only use 20 observation sequences to try and keep training time
  // shorter.
  observations.resize(20);

  hmm.Train(observations);

  BOOST_REQUIRE_CLOSE(hmm.Initial()[0], 1.0, 0.1);
  BOOST_REQUIRE_SMALL(hmm.Initial()[1], 0.05);
  BOOST_REQUIRE_SMALL(hmm.Initial()[2], 0.05);

  // The tolerances are increased because there is more error in unlabeled
  // training; we use an absolute tolerance of 0.03 for the transition matrices.
  // Check that the transition matrix is correct.
  for (size_t row = 0; row < 3; row++)
    for (size_t col = 0; col < 3; col++)
      BOOST_REQUIRE_SMALL(transition(row, col) - hmm.Transition()(row, col),
          0.03);

  // Check that each distribution is correct.
  for (size_t dist = 0; dist < 3; dist++)
  {
    BOOST_REQUIRE_LT(arma::norm(hmm.Emission()[dist].Mean() -
        emission[dist].Mean()), 0.1);
    BOOST_REQUIRE_LT(arma::norm(hmm.Emission()[dist].Covariance() -
        emission[dist].Covariance()), 0.25);
  }
}

/**
 * Make sure that a random sequence generated by a Gaussian HMM fits the
 * distribution correctly.
 */
BOOST_AUTO_TEST_CASE(GaussianHMMGenerateTest)
{
  // Our distribution will have three two-dimensional output Gaussians.
  HMM<GaussianDistribution> hmm(3, GaussianDistribution(2));
  hmm.Transition() = arma::mat("0.4 0.6 0.8; 0.2 0.2 0.1; 0.4 0.2 0.1");
  hmm.Emission()[0] = GaussianDistribution("0.0 0.0", "1.0 0.0; 0.0 1.0");
  hmm.Emission()[1] = GaussianDistribution("2.0 2.0", "1.0 0.5; 0.5 1.2");
  hmm.Emission()[2] = GaussianDistribution("-2.0 1.0", "2.0 0.1; 0.1 1.0");

  // Now we will generate a long sequence.
  std::vector<arma::mat> observations(1);
  std::vector<arma::Row<size_t> > states(1);

  // Start in state 1 (no reason).
  hmm.Generate(10000, observations[0], states[0], 1);

  HMM<GaussianDistribution> hmm2(3, GaussianDistribution(2));

  // Now estimate the HMM from the generated sequence.
  hmm2.Train(observations, states);

  // Check that the estimated matrices are the same.
  BOOST_REQUIRE_LT(arma::norm(hmm.Transition() - hmm2.Transition()), 0.1);

  // Check that each Gaussian is the same.
  for (size_t dist = 0; dist < 3; dist++)
  {
    BOOST_REQUIRE_LT(arma::norm(hmm.Emission()[dist].Mean() -
        hmm2.Emission()[dist].Mean()), 0.2);
    BOOST_REQUIRE_LT(arma::norm(hmm.Emission()[dist].Covariance() -
        hmm2.Emission()[dist].Covariance()), 0.3);
  }
}

/**
 * Make sure that Predict() is numerically stable.
 */
BOOST_AUTO_TEST_CASE(GaussianHMMPredictTest)
{
  size_t numState = 10;
  size_t obsDimension = 2;
  HMM<GaussianDistribution> hmm(numState, GaussianDistribution(obsDimension));

  arma::vec initial = {1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  arma::mat transition = {{0.9149, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                          {0.0851, 0.8814, 0, 0, 0, 0, 0, 0, 0, 0},
                          {0, 0.1186, 0.9031, 0, 0, 0, 0, 0, 0, 0},
                          {0, 0, 0.0969, 0.903, 0, 0, 0, 0, 0, 0},
                          {0, 0, 0, 0.097, 0.8941, 0, 0, 0, 0, 0},
                          {0, 0, 0, 0, 0.1059, 0.9024, 0, 0, 0, 0},
                          {0, 0, 0, 0, 0, 0.0976, 0.8902, 0, 0, 0},
                          {0, 0, 0, 0, 0, 0, 0.1098, 0.9107, 0, 0},
                          {0, 0, 0, 0, 0, 0, 0, 0.0893, 0.8964, 0},
                          {0, 0, 0, 0, 0, 0, 0, 0, 0.1036, 1}};

  std::vector<arma::vec> mean = {{0, 0.259},
                                 {0.0372, 0.2063},
                                 {0.1496, -0.3075},
                                 {-0.0366, -0.3255},
                                 {-0.2866, -0.0202},
                                 {0.1804, 0.1385},
                                 {0.1922, -0.0616},
                                 {-0.378, -0.1751},
                                 {-0.1346, 0.1357},
                                 {0.338, 0.183}};

  std::vector<arma::mat> cov = {
      {{3.2837e-07, 0}, {0, 0.032837}},
      {{0.0154, -0.0093}, {-0.0093, 0.0358}},
      {{0.1087, -0.0032}, {-0.0032, 0.0587}},
      {{0.3185, -0.0069}, {-0.0069, 0.0396}},
      {{0.3472, 0.0484}, {0.0484, 0.0706}},
      {{0.39, 0.0406}, {0.0406, 0.0653}},
      {{0.4502, 0.0718}, {0.0718, 0.0705}},
      {{0.3253, 0.0312}, {0.0312, 0.0783}},
      {{0.2355, 0.0195}, {0.0195, 0.0276}},
      {{0.0818, 0.022}, {0.022, 0.0282}}};

  hmm.Initial() = initial;
  hmm.Transition() = transition;

  for (size_t i = 0; i < numState; ++i)
  {
    GaussianDistribution& emission = hmm.Emission().at(i);
    emission.Mean() = mean.at(i);
    emission.Covariance(cov.at(i));
  }

  arma::mat obs = {
      {
          -0.0424, -0.0395, -0.0336, -0.0294, -0.0299, -0.032, -0.0289, -0.0148,
          0.0095, 0.0416, 0.0795, 0.1173, 0.1491, 0.1751, 0.1999, 0.2277,
          0.2586, 0.2858, 0.3019, 0.303, 0.289, 0.2632, 0.2301, 0.1923, 0.1498,
          0.1021, 0.0471, -0.0191, -0.0969, -0.1795, -0.2559, -0.323, -0.3882,
          -0.4582, -0.5334, -0.609, -0.6778999999999999, -0.7278, -0.7481,
          -0.7356, -0.6953, -0.635, -0.5617, -0.478, -0.3833, -0.2721, -0.1365,
          0.0283, 0.217, 0.4148, 0.6028, 0.7664, 0.8937, 0.9737, 1, 0.972,
          0.8972, 0.7891, 0.6613, 0.524, 0.3847, 0.2489, 0.1187, -0.0045,
          -0.1214, -0.2316, -0.3328, -0.4211, -0.4963, -0.5607, -0.6136,
          -0.6532, -0.6777, -0.6867, -0.6807, -0.6612, -0.6345, -0.6075,
          -0.5748, -0.5278, -0.4747, -0.4176, -0.33, -0.2036, -0.0597,
          0.07240000000000001, 0.1754, 0.2471, 0.295, 0.3356, 0.3809, 0.4299,
          0.4737, 0.4987, 0.4958, 0.4676, 0.4253, 0.3802, 0.342, 0.3183
      },
      {
          0.2355, 0.2639, 0.2971, 0.3301, 0.3598, 0.3842, 0.3995, 0.4019, 0.39,
          0.3624, 0.3201, 0.2658, 0.203, 0.1341, 0.06, -0.0179, -0.1006,
          -0.1869, -0.2719, -0.35, -0.4176, -0.4739, -0.52, -0.5584, -0.5913,
          -0.6196, -0.642, -0.6554, -0.6567, -0.6459, -0.6271, -0.6029, -0.5722,
          -0.5318000000000001, -0.4802, -0.4174, -0.3449, -0.2685, -0.1927,
          -0.1201, -0.0532, 0.008699999999999999, 0.0673, 0.1204, 0.1647,
          0.2008, 0.2284, 0.2447, 0.2504, 0.2479, 0.2373, 0.2148, 0.1781,
          0.1283, 0.06710000000000001, -0.0022, -0.0743, -0.1463, -0.2149,
          -0.2784, -0.3362, -0.3867, -0.4297, -0.4651, -0.4924, -0.5101,
          -0.5168, -0.5117, -0.496, -0.4706, -0.4358, -0.3923, -0.3419, -0.2868,
          -0.2289, -0.1702, -0.1094, -0.0421, 0.0311, 0.1047, 0.1732, 0.2257,
          0.254, 0.2532, 0.2308, 0.2017, 0.1724, 0.1425, 0.1195, 0.099, 0.0759,
          0.0521, 0.0313, 0.0188, 0.0113, 0.0068, 0.0042, 0.0026, 0.0018, 0.0014
      }
  };

  arma::Row<size_t> stateSeq;
  auto likelihood = hmm.LogLikelihood(obs);
  hmm.Predict(obs, stateSeq);

  arma::Row<size_t> stateSeqRef = { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4,
      4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9,
      9, 9, 9, 9, 9, 9, 9, 9, 9, 9 };

  BOOST_REQUIRE_CLOSE(likelihood, -2734.43, 1e-3);

  for (size_t i = 0; i < stateSeqRef.n_cols; ++i)
  {
    BOOST_REQUIRE_EQUAL(stateSeqRef.at(i), stateSeq.at(i));
  }
}

/**
 * Test that HMMs work with Gaussian mixture models.  We'll try putting in a
 * simple model by hand and making sure that prediction of observation sequences
 * works correctly.
 */
BOOST_AUTO_TEST_CASE(GMMHMMPredictTest)
{
  // It's possible, but extremely unlikely, that this test can fail.  So we are
  // willing to do three trials in case the first two fail.
  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    // We will use two GMMs; one with two components and one with three.
    std::vector<GMM> gmms(2);
    gmms[0] = GMM(2, 2);
    gmms[0].Weights() = arma::vec("0.75 0.25");

    // N([2.25 3.10], [1.00 0.20; 0.20 0.89])
    gmms[0].Component(0) = GaussianDistribution("4.25 3.10",
                                                "1.00 0.20; 0.20 0.89");

    // N([4.10 1.01], [1.00 0.00; 0.00 1.01])
    gmms[0].Component(1) = GaussianDistribution("7.10 5.01",
                                                "1.00 0.00; 0.00 1.01");

    gmms[1] = GMM(3, 2);
    gmms[1].Weights() = arma::vec("0.4 0.2 0.4");

    gmms[1].Component(0) = GaussianDistribution("-3.00 -6.12",
                                                "1.00 0.00; 0.00 1.00");

    gmms[1].Component(1) = GaussianDistribution("-4.25 -7.12",
                                                "1.50 0.60; 0.60 1.20");

    gmms[1].Component(2) = GaussianDistribution("-6.15 -2.00",
                                                "1.00 0.80; 0.80 1.00");

    // Default MATLAB initial probabilities.
    arma::vec initial("1 0");

    // Transition matrix.
    arma::mat trans("0.30 0.50;"
                    "0.70 0.50");

    // Now build the model.
    HMM<GMM> hmm(initial, trans, gmms);

    // Make a sequence of observations.
    arma::mat observations(2, 1000);
    arma::Row<size_t> states(1000);
    states[0] = 0;
    observations.col(0) = gmms[0].Random();

    for (size_t i = 1; i < 1000; i++)
    {
      double randValue = math::Random();

      if (randValue <= trans(0, states[i - 1]))
        states[i] = 0;
      else
        states[i] = 1;

      observations.col(i) = gmms[states[i]].Random();
    }

    // Run the prediction.
    arma::Row<size_t> predictions;
    hmm.Predict(observations, predictions);

    // Check that the predictions were correct.
    success = true;
    for (size_t i = 0; i < 1000; i++)
    {
      if (predictions[i] != states[i])
      {
        success = false;
        break;
      }
    }

    if (success)
      break;
  }

  BOOST_REQUIRE_EQUAL(success, true);
}

/**
 * Test that GMM-based HMMs can train on models correctly using labeled training
 * data.
 */
BOOST_AUTO_TEST_CASE(GMMHMMLabeledTrainingTest)
{
  // We will use two GMMs; one with two components and one with three.
  std::vector<GMM> gmms(2, GMM(2, 2));
  gmms[0].Weights() = arma::vec("0.3 0.7");

  // N([2.25 3.10], [1.00 0.20; 0.20 0.89])
  gmms[0].Component(0) = GaussianDistribution("4.25 3.10",
                                              "1.00 0.20; 0.20 0.89");

  // N([4.10 1.01], [1.00 0.00; 0.00 1.01])
  gmms[0].Component(1) = GaussianDistribution("7.10 5.01",
                                              "1.00 0.00; 0.00 1.01");

  gmms[1].Weights() = arma::vec("0.20 0.80");

  gmms[1].Component(0) = GaussianDistribution("-3.00 -6.12",
                                              "1.00 0.00; 0.00 1.00");

  gmms[1].Component(1) = GaussianDistribution("-4.25 -2.12",
                                              "1.50 0.60; 0.60 1.20");

  // Transition matrix.
  arma::mat transMat("0.40 0.60;"
                     "0.60 0.40");

  // Make a sequence of observations.
  std::vector<arma::mat> observations(5, arma::mat(2, 2500));
  std::vector<arma::Row<size_t> > states(5, arma::Row<size_t>(2500));
  for (size_t obs = 0; obs < 5; obs++)
  {
    states[obs][0] = 0;
    observations[obs].col(0) = gmms[0].Random();

    for (size_t i = 1; i < 2500; i++)
    {
      double randValue = (double) rand() / (double) RAND_MAX;

      if (randValue <= transMat(0, states[obs][i - 1]))
        states[obs][i] = 0;
      else
        states[obs][i] = 1;

      observations[obs].col(i) = gmms[states[obs][i]].Random();
    }
  }

  // Set up the GMM for training.
  HMM<GMM> hmm(2, GMM(2, 2));

  // Train the HMM.
  hmm.Train(observations, states);

  // Check the initial weights.  The dataset was generated with 100% probability
  // of a sequence starting in state 0.
  BOOST_REQUIRE_CLOSE(hmm.Initial()[0], 1.0, 0.01);
  BOOST_REQUIRE_SMALL(hmm.Initial()[1], 0.01);

  // Check the results.  Use absolute tolerances instead of percentages.
  BOOST_REQUIRE_SMALL(hmm.Transition()(0, 0) - transMat(0, 0), 0.03);
  BOOST_REQUIRE_SMALL(hmm.Transition()(0, 1) - transMat(0, 1), 0.03);
  BOOST_REQUIRE_SMALL(hmm.Transition()(1, 0) - transMat(1, 0), 0.03);
  BOOST_REQUIRE_SMALL(hmm.Transition()(1, 1) - transMat(1, 1), 0.03);

  // Now the emission probabilities (the GMMs).
  // We have to sort each GMM for comparison.
  arma::uvec sortedIndices = sort_index(hmm.Emission()[0].Weights());

  BOOST_REQUIRE_SMALL(hmm.Emission()[0].Weights()[sortedIndices[0]] -
      gmms[0].Weights()[0], 0.08);
  BOOST_REQUIRE_SMALL(hmm.Emission()[0].Weights()[sortedIndices[1]] -
      gmms[0].Weights()[1], 0.08);

  BOOST_REQUIRE_LT(arma::norm(
      hmm.Emission()[0].Component(sortedIndices[0]).Mean() -
      gmms[0].Component(0).Mean()), 0.2);
  BOOST_REQUIRE_LT(arma::norm(
      hmm.Emission()[0].Component(sortedIndices[1]).Mean() -
      gmms[0].Component(1).Mean()), 0.2);

  BOOST_REQUIRE_LT(arma::norm(
      hmm.Emission()[0].Component(sortedIndices[0]).Covariance() -
      gmms[0].Component(0).Covariance()), 0.5);
  BOOST_REQUIRE_LT(arma::norm(
      hmm.Emission()[0].Component(sortedIndices[1]).Covariance() -
      gmms[0].Component(0).Covariance()), 0.5);

  // Sort the GMM.
  sortedIndices = sort_index(hmm.Emission()[1].Weights());

  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Weights()[sortedIndices[0]] -
      gmms[1].Weights()[0], 0.08);
  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Weights()[sortedIndices[1]] -
      gmms[1].Weights()[1], 0.08);

  BOOST_REQUIRE_LT(arma::norm(
      hmm.Emission()[1].Component(sortedIndices[0]).Mean() -
      gmms[1].Component(0).Mean()), 0.2);
  BOOST_REQUIRE_LT(arma::norm(
      hmm.Emission()[1].Component(sortedIndices[1]).Mean() -
      gmms[1].Component(1).Mean()), 0.2);

  BOOST_REQUIRE_LT(arma::norm(
      hmm.Emission()[1].Component(sortedIndices[0]).Covariance() -
      gmms[1].Component(0).Covariance()), 0.5);
  BOOST_REQUIRE_LT(arma::norm(
      hmm.Emission()[1].Component(sortedIndices[1]).Covariance() -
      gmms[1].Component(1).Covariance()), 0.5);
}

/**
 * Test saving and loading of GMM HMMs
 */
BOOST_AUTO_TEST_CASE(GMMHMMLoadSaveTest)
{
  // Create a GMM HMM, save it, and load it.
  HMM<GMM> hmm(3, GMM(4, 3));

  for (size_t j = 0; j < hmm.Emission().size(); ++j)
  {
    hmm.Emission()[j].Weights().randu();
    for (size_t i = 0; i < hmm.Emission()[j].Gaussians(); ++i)
    {
      hmm.Emission()[j].Component(i).Mean().randu();
      arma::mat covariance = arma::randu<arma::mat>(
          hmm.Emission()[j].Component(i).Covariance().n_rows,
          hmm.Emission()[j].Component(i).Covariance().n_cols);
      covariance *= covariance.t();
      covariance += arma::eye<arma::mat>(covariance.n_rows, covariance.n_cols);
      hmm.Emission()[j].Component(i).Covariance(std::move(covariance));
    }
  }

  // Save the HMM.
  {
    std::ofstream ofs("test-hmm-save.xml");
    boost::archive::xml_oarchive ar(ofs);
    ar << BOOST_SERIALIZATION_NVP(hmm);
  }

  // Load the HMM.
  HMM<GMM> hmm2(3, GMM(4, 3));
  {
    std::ifstream ifs("test-hmm-save.xml");
    boost::archive::xml_iarchive ar(ifs);
    ar >> BOOST_SERIALIZATION_NVP(hmm2);
  }

  // Remove clutter.
  remove("test-hmm-save.xml");

  for (size_t j = 0; j < hmm.Emission().size(); ++j)
  {
    BOOST_REQUIRE_EQUAL(hmm.Emission()[j].Gaussians(),
                        hmm2.Emission()[j].Gaussians());
    BOOST_REQUIRE_EQUAL(hmm.Emission()[j].Dimensionality(),
                        hmm2.Emission()[j].Dimensionality());

    for (size_t i = 0; i < hmm.Emission()[j].Dimensionality(); ++i)
      BOOST_REQUIRE_CLOSE(hmm.Emission()[j].Weights()[i],
                          hmm2.Emission()[j].Weights()[i], 1e-3);

    for (size_t i = 0; i < hmm.Emission()[j].Gaussians(); ++i)
    {
      for (size_t l = 0; l < hmm.Emission()[j].Dimensionality(); ++l)
      {
        BOOST_REQUIRE_CLOSE(hmm.Emission()[j].Component(i).Mean()[l],
            hmm2.Emission()[j].Component(i).Mean()[l], 1e-3);

        for (size_t k = 0; k < hmm.Emission()[j].Dimensionality(); ++k)
        {
          BOOST_REQUIRE_CLOSE(hmm.Emission()[j].Component(i).Covariance()(l, k),
              hmm2.Emission()[j].Component(i).Covariance()(l, k), 1e-3);
        }
      }
    }
  }
}

/**
 * Test saving and loading of Gaussian HMMs
 */
BOOST_AUTO_TEST_CASE(GaussianHMMLoadSaveTest)
{
  // Create a Gaussian HMM, save it, and load it.
  HMM<GaussianDistribution> hmm(3, GaussianDistribution(2));

  for (size_t j = 0; j < hmm.Emission().size(); ++j)
  {
    hmm.Emission()[j].Mean().randu();
    arma::mat covariance = arma::randu<arma::mat>(
        hmm.Emission()[j].Covariance().n_rows,
        hmm.Emission()[j].Covariance().n_cols);
    covariance *= covariance.t();
    covariance += arma::eye<arma::mat>(covariance.n_rows, covariance.n_cols);
    hmm.Emission()[j].Covariance(std::move(covariance));
  }

  // Save the HMM.
  {
    std::ofstream ofs("test-hmm-save.xml");
    boost::archive::xml_oarchive ar(ofs);
    ar << BOOST_SERIALIZATION_NVP(hmm);
  }

  // Load the HMM.
  HMM<GaussianDistribution> hmm2(3, GaussianDistribution(2));
  {
    std::ifstream ifs("test-hmm-save.xml");
    boost::archive::xml_iarchive ar(ifs);
    ar >> BOOST_SERIALIZATION_NVP(hmm2);
  }

  // Remove clutter.
  remove("test-hmm-save.xml");

  for (size_t j = 0; j < hmm.Emission().size(); ++j)
  {
    BOOST_REQUIRE_EQUAL(hmm.Emission()[j].Dimensionality(),
                        hmm2.Emission()[j].Dimensionality());

    for (size_t i = 0; i < hmm.Emission()[j].Dimensionality(); ++i)
    {
      BOOST_REQUIRE_CLOSE(hmm.Emission()[j].Mean()[i],
          hmm2.Emission()[j].Mean()[i], 1e-3);
      for (size_t k = 0; k < hmm.Emission()[j].Dimensionality(); ++k)
      {
        BOOST_REQUIRE_CLOSE(hmm.Emission()[j].Covariance()(i, k),
            hmm2.Emission()[j].Covariance()(i, k), 1e-3);
      }
    }
  }
}

/**
 * Test saving and loading of Discrete HMMs
 */
BOOST_AUTO_TEST_CASE(DiscreteHMMLoadSaveTest)
{
  // Create a Discrete HMM, save it, and load it.
  std::vector<DiscreteDistribution> emission(4);
  emission[0].Probabilities() = arma::randu<arma::vec>(6);
  emission[0].Probabilities() /= accu(emission[0].Probabilities());
  emission[1].Probabilities() = arma::randu<arma::vec>(6);
  emission[1].Probabilities() /= accu(emission[1].Probabilities());
  emission[2].Probabilities() = arma::randu<arma::vec>(6);
  emission[2].Probabilities() /= accu(emission[2].Probabilities());
  emission[3].Probabilities() = arma::randu<arma::vec>(6);
  emission[3].Probabilities() /= accu(emission[3].Probabilities());


  // Create HMM object.
  HMM<DiscreteDistribution> hmm(3, DiscreteDistribution(3));


  for (size_t j = 0; j < hmm.Emission().size(); ++j)
  {
    hmm.Emission()[j].Probabilities() = arma::randu<arma::vec>(3);
    hmm.Emission()[j].Probabilities() /= accu(emission[j].Probabilities());
  }

  // Save the HMM.
  {
    std::ofstream ofs("test-hmm-save.xml");
    boost::archive::xml_oarchive ar(ofs);
    ar << BOOST_SERIALIZATION_NVP(hmm);
  }

  // Load the HMM.
  HMM<DiscreteDistribution> hmm2(3, DiscreteDistribution(3));
  {
    std::ifstream ifs("test-hmm-save.xml");
    boost::archive::xml_iarchive ar(ifs);
    ar >> BOOST_SERIALIZATION_NVP(hmm2);
  }

  // Remove clutter.
  remove("test-hmm-save.xml");

  for (size_t j = 0; j < hmm.Emission().size(); ++j)
    for (size_t i = 0; i < hmm.Emission()[j].Probabilities().n_elem; ++i)
      BOOST_REQUIRE_CLOSE(hmm.Emission()[j].Probabilities()[i],
          hmm2.Emission()[j].Probabilities()[i], 1e-3);
}

/**
 * Test that HMM::Train() returns finite log-likelihood.
 */
BOOST_AUTO_TEST_CASE(HMMTrainReturnLogLikelihood)
{
  HMM<DiscreteDistribution> hmm(1, 2); // 1 state, 2 emissions.
  // Randomize the emission matrix.
  hmm.Emission()[0].Probabilities() = arma::randu<arma::vec>(2);
  hmm.Emission()[0].Probabilities() /= accu(hmm.Emission()[0].Probabilities());

  std::vector<arma::mat> observations;
  observations.push_back("0 1 0 1 0 1 0 1 0 1 0 1");
  observations.push_back("0 0 0 0 0 0 1 1 1 1 1 1");
  observations.push_back("1 1 1 1 1 1 0 0 0 0 0 0");
  observations.push_back("1 1 1 0 0 0 1 1 1 0 0 0");
  observations.push_back("0 0 1 1 0 0 0 0 1 1 1 1");
  observations.push_back("1 1 1 0 0 0 1 1 1 0 0 0");
  observations.push_back("0 1 0 1 0 1 0 1 0 1 0 1");
  observations.push_back("0 0 0 0 0 0 1 1 1 1 1 1");
  observations.push_back("1 1 1 1 1 0 1 0 0 0 0 0");
  observations.push_back("1 1 1 0 0 1 0 1 1 0 0 0");
  observations.push_back("0 0 1 1 0 0 0 1 0 1 1 1");
  observations.push_back("1 1 1 0 0 1 0 1 1 0 0 0");

  double loglik = hmm.Train(observations);

  BOOST_REQUIRE_EQUAL(std::isfinite(loglik), true);
}

/********************************************/
/** DiagonalGMM Hidden Markov Models Tests **/
/********************************************/

//! Make sure the prediction of DiagonalGMM HMMs is reasonable.
BOOST_AUTO_TEST_CASE(DiagonalGMMHMMPredictTest)
{
  // This test is probabilistic, so we perform it three times to make it robust.
  bool success = false;
  for (size_t trial = 0; trial < 3; trial++)
  {
    std::vector<DiagonalGMM> gmms(2);
    gmms[0] = DiagonalGMM(2, 2);

    gmms[0].Component(0) = DiagonalGaussianDistribution("3.25 2.10",
        "0.97 1.00");
    gmms[0].Component(1) = DiagonalGaussianDistribution("5.03 7.28",
        "1.20 0.89");

    gmms[1] = DiagonalGMM(3, 2);
    gmms[1].Weights() = arma::vec("0.3 0.2 0.5");
    gmms[1].Component(0) = DiagonalGaussianDistribution("-2.48 -3.02",
        "1.02 0.80");
    gmms[1].Component(1) = DiagonalGaussianDistribution("-1.24 -2.40",
        "0.85 0.78");
    gmms[1].Component(2) = DiagonalGaussianDistribution("-5.68 -4.83",
        "1.42 0.96");

    // Initial probabilities.
    arma::vec initial("1 0");

    // Transition matrix.
    arma::mat transProb("0.40 0.70;"
                        "0.60 0.30");

    // Build the model.
    HMM<DiagonalGMM> hmm(initial, transProb, gmms);

    // Make a sequence of observations according to transition probabilities.
    arma::mat observations(2, 1000);
    arma::Row<size_t> states(1000);

    // Set initial state to zero.
    states[0] = 0;
    observations.col(0) = gmms[0].Random();

    for (size_t i = 1; i < 1000; i++)
    {
      double randValue = math::Random();

      if (randValue <= transProb(0, states[i - 1]))
        states[i] = 0;
      else
        states[i] = 1;

      observations.col(i) = gmms[states[i]].Random();
    }

    // Predict the most probable hidden state sequence.
    arma::Row<size_t> predictions;
    hmm.Predict(observations, predictions);

    // Check them.
    success = true;
    for (size_t i = 0; i < 1000; i++)
    {
      if (predictions[i] != states[i])
      {
        success = false;
        break;
      }
    }

    if (success)
      break;
  }

  BOOST_REQUIRE_EQUAL(success, true);
}

/**
 * Make sure a random data sequence generation is correct when the emission
 * distribution is DiagonalGMM.
 */
BOOST_AUTO_TEST_CASE(DiagonalGMMHMMGenerateTest)
{
  // Build the model.
  HMM<DiagonalGaussianDistribution> hmm(3, DiagonalGaussianDistribution(2));
  hmm.Transition() = arma::mat("0.2 0.3 0.8;"
                               "0.4 0.5 0.1;"
                               "0.4 0.2 0.1");

  hmm.Emission()[0] = DiagonalGaussianDistribution("0.0 0.0", "1.0 0.7");
  hmm.Emission()[1] = DiagonalGaussianDistribution("1.0 1.0", "0.7 0.5");
  hmm.Emission()[2] = DiagonalGaussianDistribution("-3.0 2.0", "2.0 0.3");

  // Now we will generate a long sequence.
  std::vector<arma::mat> observations(1);
  std::vector<arma::Row<size_t> > states(1);

  // Generate a random data sequence.
  hmm.Generate(10000, observations[0], states[0], 1);

  // Build the hmm2.
  HMM<DiagonalGaussianDistribution> hmm2(3, DiagonalGaussianDistribution(2));

  // Now estimate the HMM from the generated sequence.
  hmm2.Train(observations, states);

  // Check that the estimated matrices are the same.
  BOOST_REQUIRE_LT(arma::norm(hmm.Transition() - hmm2.Transition()), 0.05);

  // Check that each Gaussian is the same.
  for (size_t dist = 0; dist < 3; dist++)
  {
    BOOST_REQUIRE_LT(arma::norm(hmm.Emission()[dist].Mean() -
        hmm2.Emission()[dist].Mean()), 0.1);
    BOOST_REQUIRE_LT(arma::norm(hmm.Emission()[dist].Covariance() -
        hmm2.Emission()[dist].Covariance()), 0.2);
  }
}

/**
 * Make sure the unlabeled 1-state training works reasonably given a single
 * distribution with diagonal covariance.
 */
BOOST_AUTO_TEST_CASE(DiagonalGMMHMMOneGaussianOneStateTrainingTest)
{
  // Create a Gaussian distribution with diagonal covariance.
  DiagonalGaussianDistribution d("2.05 3.45", "0.89 1.05");

  // Make a sequence of observations.
  std::vector<arma::mat> observations(1, arma::mat(2, 5000));
  for (size_t obs = 0; obs < 1; obs++)
  {
    observations[obs].col(0) = d.Random();

    for (size_t i = 1; i < 5000; i++)
    {
      observations[obs].col(i) = d.Random();
    }
  }

  // Build the model.
  HMM<DiagonalGMM> hmm(1, DiagonalGMM(1, 2));

  // Train with observations.
  hmm.Train(observations);

  // Generate the ground truth values.
  arma::vec actualMean = arma::mean(observations[0], 1);
  arma::vec actualCovar = arma::diagvec(
      mlpack::math::ColumnCovariance(observations[0],
      1 /* biased estimator */));

  // Check the model to see that it is correct.
  CheckMatrices(hmm.Emission()[0].Component(0).Mean(), actualMean);
  CheckMatrices(hmm.Emission()[0].Component(0).Covariance(), actualCovar);
}

/**
 * Make sure the unlabeled training works reasonably given a single
 * distribution with diagonal covariance.
 */
BOOST_AUTO_TEST_CASE(DiagonalGMMHMMOneGaussianUnlabeledTrainingTest)
{
  // Create a sequence of DiagonalGMMs. Each GMM has one gaussian distribution.
  std::vector<DiagonalGMM> gmms(2, DiagonalGMM(1, 2));
  gmms[0].Component(0) = DiagonalGaussianDistribution("1.25 2.10",
      "0.97 1.00");

  gmms[1].Component(0) = DiagonalGaussianDistribution("-2.48 -3.02",
      "1.02 0.80");

  // Transition matrix.
  arma::mat transProbs("0.30 0.80;"
                       "0.70 0.20");

  arma::vec initialProb("1 0");

  // Make a sequence of observations.
  std::vector<arma::mat> observations(2, arma::mat(2, 500));
  std::vector<arma::Row<size_t>> states(2, arma::Row<size_t>(500));
  for (size_t obs = 0; obs < 2; obs++)
  {
    states[obs][0] = 0;
    observations[obs].col(0) = gmms[0].Random();

    for (size_t i = 1; i < 500; i++)
    {
      double randValue = math::Random();

      if (randValue <= transProbs(0, states[obs][i - 1]))
        states[obs][i] = 0;
      else
        states[obs][i] = 1;

      observations[obs].col(i) = gmms[states[obs][i]].Random();
    }
  }

  // Build the model.
  HMM<DiagonalGMM> hmm(initialProb, transProbs, gmms);

  // Train the model. If labels are not given, when training GMM, the estimated
  // probabilities based on the forward and backward probabilities is used.
  hmm.Train(observations);

  // Check the initial weights.
  BOOST_REQUIRE_CLOSE(hmm.Initial()[0], 1.0, 0.01);
  BOOST_REQUIRE_SMALL(hmm.Initial()[1], 0.01);

  // Check the transition probability matrix.
  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      BOOST_REQUIRE_SMALL(hmm.Transition()(i, j) - transProbs(i, j), 0.08);

  // Check the estimated weights of the each emission distribution.
  for (size_t i = 0; i < 2; i++)
    BOOST_REQUIRE_SMALL(hmm.Emission()[i].Weights()[0] - gmms[i].Weights()[0],
        0.08);

  // Check the estimated means of the each emission distribution.
  for (size_t i = 0; i < 2; i++)
    BOOST_REQUIRE_LT(arma::norm(hmm.Emission()[i].Component(0).Mean() -
        gmms[i].Component(0).Mean()), 0.2);

  // Check the estimated covariances of the each emission distribution.
  for (size_t i = 0; i < 2; i++)
    BOOST_REQUIRE_LT(arma::norm(hmm.Emission()[i].Component(0).Covariance() -
        gmms[i].Component(0).Covariance()), 0.5);
}

/**
 * Make sure the labeled training works reasonably given a single distribution
 * with diagonal covariance.
 */
BOOST_AUTO_TEST_CASE(DiagonalGMMHMMOneGaussianLabeledTrainingTest)
{
  // Create a sequence of DiagonalGMMs.
  std::vector<DiagonalGMM> gmms(3, DiagonalGMM(1, 2));
  gmms[0].Component(0) = DiagonalGaussianDistribution("5.25 7.10",
      "0.97 1.00");

  gmms[1].Component(0) = DiagonalGaussianDistribution("4.48 6.02",
      "1.02 0.80");

  gmms[2].Component(0) = DiagonalGaussianDistribution("-3.28 -5.30",
      "0.87 1.05");

  // Transition matrix.
  arma::mat transProbs("0.2 0.4 0.4;"
                       "0.3 0.4 0.3;"
                       "0.5 0.2 0.3");

  arma::vec initialProb("1 0 0");

  // Make a sequence of observations.
  std::vector<arma::mat> observations(3, arma::mat(2, 5000));
  std::vector<arma::Row<size_t>> states(3, arma::Row<size_t>(5000));
  for (size_t obs = 0; obs < 3; obs++)
  {
    states[obs][0] = 0;
    observations[obs].col(0) = gmms[0].Random();

    for (size_t i = 1; i < 5000; i++)
    {
      double randValue = math::Random();
      double probSum = 0;
      for (size_t state = 0; state < 3; state++)
      {
        probSum += transProbs(state, states[obs][i - 1]);
        if (randValue <= probSum)
        {
          states[obs][i] = state;
          break;
        }
      }

      observations[obs].col(i) = gmms[states[obs][i]].Random();
    }
  }

  // Build the model.
  HMM<DiagonalGMM> hmm(3, DiagonalGMM(1, 2));

  // Train the model.
  hmm.Train(observations, states);

  // Check the initial weights.
  BOOST_REQUIRE_CLOSE(hmm.Initial()[0], 1.0, 0.01);
  BOOST_REQUIRE_SMALL(hmm.Initial()[1], 0.01);
  BOOST_REQUIRE_SMALL(hmm.Initial()[2], 0.01);

  // Check the transition probability matrix.
  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++)
      BOOST_REQUIRE_SMALL(hmm.Transition()(i, j) - transProbs(i, j), 0.03);

  // Check the estimated weights of the each emission distribution.
  for (size_t i = 0; i < 3; i++)
    BOOST_REQUIRE_SMALL(hmm.Emission()[i].Weights()[0] - gmms[i].Weights()[0],
        0.08);

  // Check the estimated means of the each emission distribution.
  for (size_t i = 0; i < 3; i++)
    BOOST_REQUIRE_LT(arma::norm(hmm.Emission()[i].Component(0).Mean() -
        gmms[i].Component(0).Mean()), 0.2);

  // Check the estimated covariances of the each emission distribution.
  for (size_t i = 0; i < 3; i++)
    BOOST_REQUIRE_LT(arma::norm(hmm.Emission()[i].Component(0).Covariance() -
        gmms[i].Component(0).Covariance()), 0.5);
}

/**
 * Make sure the unlabeled training works reasonably given multiple
 * distributions with diagonal covariance.
 */
BOOST_AUTO_TEST_CASE(DiagonalGMMHMMMultipleGaussiansUnlabeledTrainingTest)
{
  // Create a sequence of DiagonalGMMs.
  std::vector<DiagonalGMM> gmms(2, DiagonalGMM(2, 2));
  gmms[0].Weights() = arma::vec("0.3 0.7");
  gmms[0].Component(0) = DiagonalGaussianDistribution("8.25 7.10",
      "0.97 1.00");
  gmms[0].Component(1) = DiagonalGaussianDistribution("-3.03 -2.28",
      "1.20 0.89");

  gmms[1].Weights() = arma::vec("0.4 0.6");
  gmms[1].Component(0) = DiagonalGaussianDistribution("4.48 6.02",
        "1.02 0.80");
  gmms[1].Component(1) = DiagonalGaussianDistribution("-9.24 -8.40",
        "0.85 1.58");

  // Transition matrix.
  arma::mat transProbs("0.30 0.40;"
                       "0.70 0.60");

  arma::vec initialProb("1 0");

  // Make a sequence of observations.
  std::vector<arma::mat> observations(2, arma::mat(2, 1000));
  std::vector<arma::Row<size_t>> states(2, arma::Row<size_t>(1000));
  for (size_t obs = 0; obs < 2; obs++)
  {
    states[obs][0] = 0;
    observations[obs].col(0) = gmms[0].Random();

    for (size_t i = 1; i < 1000; i++)
    {
      double randValue = math::Random();

      if (randValue <= transProbs(0, states[obs][i - 1]))
        states[obs][i] = 0;
      else
        states[obs][i] = 1;

      observations[obs].col(i) = gmms[states[obs][i]].Random();
    }
  }

  // Build the model.
  HMM<DiagonalGMM> hmm(initialProb, transProbs, gmms);

  // Train the model. If labels are not given, when training GMM, the estimated
  // probabilities based on the forward and backward probabilities is used.
  hmm.Train(observations);

  // Check the initial weights.
  BOOST_REQUIRE_CLOSE(hmm.Initial()[0], 1.0, 0.01);
  BOOST_REQUIRE_SMALL(hmm.Initial()[1], 0.01);

  // Check the transition probability matrix.
  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      BOOST_REQUIRE_SMALL(hmm.Transition()(i, j) - transProbs(i, j), 0.08);

  // Sort by the estimated weights of the first emission distribution.
  arma::uvec sortedIndices = sort_index(hmm.Emission()[0].Weights());

  // Check the first emission distribution.
  for (size_t i = 0; i < 2; i++)
  {
    // Check the estimated weights using the first DiagonalGMM.
    BOOST_REQUIRE_SMALL(hmm.Emission()[0].Weights()[sortedIndices[i]] -
        gmms[0].Weights()[i], 0.08);

    // Check the estimated means using the first DiagonalGMM.
    BOOST_REQUIRE_LT(arma::norm(
      hmm.Emission()[0].Component(sortedIndices[i]).Mean() -
      gmms[0].Component(i).Mean()), 0.35);

    // Check the estimated covariances using the first DiagonalGMM.
    BOOST_REQUIRE_LT(arma::norm(
      hmm.Emission()[0].Component(sortedIndices[i]).Covariance() -
      gmms[0].Component(i).Covariance()), 0.6);
  }

  // Sort by the estimated weights of the second emission distribution.
  sortedIndices = sort_index(hmm.Emission()[1].Weights());

  // Check the second emission distribution.
  for (size_t i = 0; i < 2; i++)
  {
    // Check the estimated weights using the second DiagonalGMM.
    BOOST_REQUIRE_SMALL(hmm.Emission()[1].Weights()[sortedIndices[i]] -
        gmms[1].Weights()[i], 0.08);

    // Check the estimated means using the second DiagonalGMM.
    BOOST_REQUIRE_LT(arma::norm(
      hmm.Emission()[1].Component(sortedIndices[i]).Mean() -
      gmms[1].Component(i).Mean()), 0.35);

    // Check the estimated covariances using the second DiagonalGMM.
    BOOST_REQUIRE_LT(arma::norm(
      hmm.Emission()[1].Component(sortedIndices[i]).Covariance() -
      gmms[1].Component(i).Covariance()), 0.6);
  }
}

/**
 * Make sure the labeled training works reasonably given multiple distributions
 * with diagonal covariance.
 */
BOOST_AUTO_TEST_CASE(DiagonalGMMHMMMultipleGaussiansLabeledTrainingTest)
{
  math::RandomSeed(std::time(NULL));
  // Create a sequence of DiagonalGMMs.
  std::vector<DiagonalGMM> gmms(2, DiagonalGMM(2, 2));
  gmms[0].Weights() = arma::vec("0.3 0.7");
  gmms[0].Component(0) = DiagonalGaussianDistribution("2.25 5.30",
      "0.97 1.00");
  gmms[0].Component(1) = DiagonalGaussianDistribution("-3.15 -2.50",
      "1.20 0.89");

  gmms[1].Weights() = arma::vec("0.4 0.6");
  gmms[1].Component(0) = DiagonalGaussianDistribution("-4.48 -6.30",
        "1.02 0.80");
  gmms[1].Component(1) = DiagonalGaussianDistribution("5.24 2.40",
        "0.85 1.58");

  // Transition matrix.
  arma::mat transProbs("0.30 0.80;"
                       "0.70 0.20");

  // Make a sequence of observations.
  std::vector<arma::mat> observations(5, arma::mat(2, 2500));
  std::vector<arma::Row<size_t>> states(5, arma::Row<size_t>(2500));
  for (size_t obs = 0; obs < 5; obs++)
  {
    states[obs][0] = 0;
    observations[obs].col(0) = gmms[0].Random();

    for (size_t i = 1; i < 2500; i++)
    {
      double randValue = math::Random();

      if (randValue <= transProbs(0, states[obs][i - 1]))
        states[obs][i] = 0;
      else
        states[obs][i] = 1;

      observations[obs].col(i) = gmms[states[obs][i]].Random();
    }
  }

  // Build the model.
  HMM<DiagonalGMM> hmm(2, DiagonalGMM(2, 2));

  // Train the model.
  hmm.Train(observations, states);

  // Check the initial weights.
  BOOST_REQUIRE_CLOSE(hmm.Initial()[0], 1.0, 0.01);
  BOOST_REQUIRE_SMALL(hmm.Initial()[1], 0.01);

  // Check the transition probability matrix.
  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      BOOST_REQUIRE_SMALL(hmm.Transition()(i, j) - transProbs(i, j), 0.03);

  // Sort by the estimated weights of the first emission distribution.
  arma::uvec sortedIndices = sort_index(hmm.Emission()[0].Weights());

  // Check the first emission distribution.
  for (size_t i = 0; i < 2; i++)
  {
    // Check the estimated weights using the first DiagonalGMM.
    BOOST_REQUIRE_SMALL(hmm.Emission()[0].Weights()[sortedIndices[i]] -
        gmms[0].Weights()[i], 0.08);

    // Check the estimated means using the first DiagonalGMM.
    BOOST_REQUIRE_LT(arma::norm(
      hmm.Emission()[0].Component(sortedIndices[i]).Mean() -
      gmms[0].Component(i).Mean()), 0.2);

    // Check the estimated covariances using the first DiagonalGMM.
    BOOST_REQUIRE_LT(arma::norm(
      hmm.Emission()[0].Component(sortedIndices[i]).Covariance() -
      gmms[0].Component(i).Covariance()), 0.5);
  }

  // Sort by the estimated weights of the second emission distribution.
  sortedIndices = sort_index(hmm.Emission()[1].Weights());

  // Check the second emission distribution.
  for (size_t i = 0; i < 2; i++)
  {
    // Check the estimated weights using the second DiagonalGMM.
    BOOST_REQUIRE_SMALL(hmm.Emission()[1].Weights()[sortedIndices[i]] -
        gmms[1].Weights()[i], 0.08);

    // Check the estimated means using the second DiagonalGMM.
    BOOST_REQUIRE_LT(arma::norm(
      hmm.Emission()[1].Component(sortedIndices[i]).Mean() -
      gmms[1].Component(i).Mean()), 0.2);

    // Check the estimated covariances using the second DiagonalGMM.
    BOOST_REQUIRE_LT(arma::norm(
      hmm.Emission()[1].Component(sortedIndices[i]).Covariance() -
      gmms[1].Component(i).Covariance()), 0.5);
  }
}

/**
 * Make sure loading and saving the model is correct.
 */
BOOST_AUTO_TEST_CASE(DiagonalGMMHMMLoadSaveTest)
{
  // Create a GMM HMM, save and load it.
  HMM<DiagonalGMM> hmm(3, DiagonalGMM(4, 3));

  // Generate intial random values.
  for (size_t j = 0; j < hmm.Emission().size(); j++)
  {
    hmm.Emission()[j].Weights().randu();
    for (size_t i = 0; i < hmm.Emission()[j].Gaussians(); i++)
    {
      hmm.Emission()[j].Component(i).Mean().randu();
      arma::vec covariance = arma::randu<arma::vec>(
          hmm.Emission()[j].Component(i).Covariance().n_elem);

      covariance += arma::ones<arma::vec>(covariance.n_elem);
      hmm.Emission()[j].Component(i).Covariance(std::move(covariance));
    }
  }

  // Save the HMM.
  {
    std::ofstream ofs("test-hmm-save.xml");
    boost::archive::xml_oarchive ar(ofs);
    ar << BOOST_SERIALIZATION_NVP(hmm);
  }

  // Load the HMM.
  HMM<DiagonalGMM> hmm2(3, DiagonalGMM(4, 3));
  {
    std::ifstream ifs("test-hmm-save.xml");
    boost::archive::xml_iarchive ar(ifs);
    ar >> BOOST_SERIALIZATION_NVP(hmm2);
  }

  // Remove clutter.
  remove("test-hmm-save.xml");

  for (size_t j = 0; j < hmm.Emission().size(); j++)
  {
    // Check the number of Gaussians.
    BOOST_REQUIRE_EQUAL(hmm.Emission()[j].Gaussians(),
                        hmm2.Emission()[j].Gaussians());

    // Check the dimensionality.
    BOOST_REQUIRE_EQUAL(hmm.Emission()[j].Dimensionality(),
                        hmm2.Emission()[j].Dimensionality());

    for (size_t i = 0; i < hmm.Emission()[j].Dimensionality(); i++)
      // Check the weights.
      BOOST_REQUIRE_CLOSE(hmm.Emission()[j].Weights()[i],
                          hmm2.Emission()[j].Weights()[i], 1e-3);

    for (size_t i = 0; i < hmm.Emission()[j].Gaussians(); i++)
    {
      for (size_t l = 0; l < hmm.Emission()[j].Dimensionality(); l++)
      {
        // Check the means.
        BOOST_REQUIRE_CLOSE(hmm.Emission()[j].Component(i).Mean()[l],
            hmm2.Emission()[j].Component(i).Mean()[l], 1e-3);

        // Check the covariances.
        BOOST_REQUIRE_CLOSE(hmm.Emission()[j].Component(i).Covariance()[l],
            hmm2.Emission()[j].Component(i).Covariance()[l], 1e-3);
      }
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
