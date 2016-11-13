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
  emission[0] = DiscreteDistribution("0.9 0.1");
  emission[1] = DiscreteDistribution("0.2 0.8");

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
  emission[0] = DiscreteDistribution("0.25 0.25 0.25 0.25");
  emission[1] = DiscreteDistribution("0.20 0.30 0.30 0.20");
  emission[2] = DiscreteDistribution("0.30 0.20 0.20 0.30");

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
  emis[0] = DiscreteDistribution("0.85 0.15 0.00 0.00");
  emis[1] = DiscreteDistribution("0.00 0.00 0.50 0.50");

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
  observations.push_back("1 1 1 1 1 1 0 0 0 0 0 0");
  observations.push_back("1 1 1 0 0 0 1 1 1 0 0 0");
  observations.push_back("0 0 1 1 0 0 0 0 1 1 1 1");
  observations.push_back("1 1 1 0 0 0 1 1 1 0 0 0");

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

  BOOST_REQUIRE_CLOSE(hmm.Emission()[0].Probability("0"), 0.4, 3.0);
  BOOST_REQUIRE_CLOSE(hmm.Emission()[0].Probability("1"), 0.6, 3.0);
  BOOST_REQUIRE_SMALL(hmm.Emission()[0].Probability("2"), 2.5);
  BOOST_REQUIRE_SMALL(hmm.Emission()[0].Probability("3"), 2.5);
  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Probability("0"), 2.5);
  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Probability("1"), 2.5);
  BOOST_REQUIRE_CLOSE(hmm.Emission()[1].Probability("2"), 0.2, 3.0);
  BOOST_REQUIRE_CLOSE(hmm.Emission()[1].Probability("3"), 0.8, 3.0);
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
  for (size_t row = 0; row < hmm.Transition().n_rows; ++row)
    BOOST_REQUIRE_SMALL(hmm.Initial()[row] - 1.0 / 3.0, 0.1);

  // We can't use % tolerance here because percent error increases as the actual
  // value gets very small.  So, instead, we just ensure that every value is no
  // more than 0.02 away from the actual value.
  for (size_t row = 0; row < hmm.Transition().n_rows; row++)
    for (size_t col = 0; col < hmm.Transition().n_cols; col++)
      BOOST_REQUIRE_SMALL(hmm.Transition()(row, col) - transition(row, col),
          0.025);

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
  // equal probability.  The default transition and emission matrices satisfy
  // this property.
  HMM<DiscreteDistribution> hmm(2, DiscreteDistribution(4));

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

  // Now check that the probabilities are right.  2% tolerance.
  BOOST_REQUIRE_CLOSE(emissionProb[0], 0.25, 2.0);
  BOOST_REQUIRE_CLOSE(emissionProb[1], 0.25, 2.0);
  BOOST_REQUIRE_CLOSE(emissionProb[2], 0.25, 2.0);
  BOOST_REQUIRE_CLOSE(emissionProb[3], 0.25, 2.0);

  BOOST_REQUIRE_CLOSE(stateProb[0], 0.50, 2.0);
  BOOST_REQUIRE_CLOSE(stateProb[1], 0.50, 2.0);
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

  // Check that training gives the same result.  Exact tolerance of 0.005.
  for (size_t row = 0; row < 4; row++)
    for (size_t col = 0; col < 4; col++)
      BOOST_REQUIRE_SMALL(hmm.Transition()(row, col) -
          hmm2.Transition()(row, col), 0.005);

  for (size_t row = 0; row < 6; row++)
  {
    arma::vec obs(1);
    obs[0] = row;
    for (size_t col = 0; col < 4; col++)
    {
      BOOST_REQUIRE_SMALL(hmm.Emission()[col].Probability(obs) -
          hmm2.Emission()[col].Probability(obs), 0.005);
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

  // We use an absolute tolerance of 0.01 for the transition matrices.
  // Check that the transition matrix is correct.
  for (size_t row = 0; row < 3; row++)
    for (size_t col = 0; col < 3; col++)
      BOOST_REQUIRE_SMALL(transition(row, col) - hmm.Transition()(row, col),
          0.01);

  // Check that each distribution is correct.
  for (size_t dist = 0; dist < 3; dist++)
  {
    // Check that the mean is correct.  Absolute tolerance of 0.04.
    for (size_t dim = 0; dim < 3; dim++)
      BOOST_REQUIRE_SMALL(hmm.Emission()[dist].Mean()(dim) -
          emission[dist].Mean()(dim), 0.04);

    // Check that the covariance is correct.  Absolute tolerance of 0.075.
    for (size_t row = 0; row < 3; row++)
      for (size_t col = 0; col < 3; col++)
        BOOST_REQUIRE_SMALL(hmm.Emission()[dist].Covariance()(row, col) -
            emission[dist].Covariance()(row, col), 0.075);
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
    // Check that the mean is correct.  Absolute tolerance of 0.09.
    for (size_t dim = 0; dim < 3; dim++)
      BOOST_REQUIRE_SMALL(hmm.Emission()[dist].Mean()(dim) -
          emission[dist].Mean()(dim), 0.09);

    // Check that the covariance is correct.  Absolute tolerance of 0.12.
    for (size_t row = 0; row < 3; row++)
      for (size_t col = 0; col < 3; col++)
        BOOST_REQUIRE_SMALL(hmm.Emission()[dist].Covariance()(row, col) -
            emission[dist].Covariance()(row, col), 0.14);
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
  for (size_t row = 0; row < 3; row++)
    for (size_t col = 0; col < 3; col++)
      BOOST_REQUIRE_SMALL(hmm.Transition()(row, col) - hmm2.Transition()(row,
          col), 0.04);

  // Check that each Gaussian is the same.
  for (size_t em = 0; em < 3; em++)
  {
    // Check that the mean is the same.
    BOOST_REQUIRE_SMALL(hmm.Emission()[em].Mean()(0) -
        hmm2.Emission()[em].Mean()(0), 0.1);
    BOOST_REQUIRE_SMALL(hmm.Emission()[em].Mean()(1) -
        hmm2.Emission()[em].Mean()(1), 0.1);

    // Check that the covariances are the same.
    BOOST_REQUIRE_SMALL(hmm.Emission()[em].Covariance()(0, 0) -
        hmm2.Emission()[em].Covariance()(0, 0), 0.2);
    BOOST_REQUIRE_SMALL(hmm.Emission()[em].Covariance()(0, 1) -
        hmm2.Emission()[em].Covariance()(0, 1), 0.2);
    BOOST_REQUIRE_SMALL(hmm.Emission()[em].Covariance()(1, 0) -
        hmm2.Emission()[em].Covariance()(1, 0), 0.2);
    BOOST_REQUIRE_SMALL(hmm.Emission()[em].Covariance()(1, 1) -
        hmm2.Emission()[em].Covariance()(1, 1), 0.2);
  }
}

/**
 * Test that HMMs work with Gaussian mixture models.  We'll try putting in a
 * simple model by hand and making sure that prediction of observation sequences
 * works correctly.
 */
BOOST_AUTO_TEST_CASE(GMMHMMPredictTest)
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
  for (size_t i = 0; i < 1000; i++)
    BOOST_REQUIRE_EQUAL(predictions[i], states[i]);
}

/**
 * Test that GMM-based HMMs can train on models correctly using labeled training
 * data.
 */
BOOST_AUTO_TEST_CASE(GMMHMMLabeledTrainingTest)
{
  srand(time(NULL));

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

  BOOST_REQUIRE_SMALL(hmm.Emission()[0].Component(sortedIndices[0]).Mean()[0] -
      gmms[0].Component(0).Mean()[0], 0.15);
  BOOST_REQUIRE_SMALL(hmm.Emission()[0].Component(sortedIndices[0]).Mean()[1] -
      gmms[0].Component(0).Mean()[1], 0.15);

  BOOST_REQUIRE_SMALL(hmm.Emission()[0].Component(sortedIndices[1]).Mean()[0] -
      gmms[0].Component(1).Mean()[0], 0.15);
  BOOST_REQUIRE_SMALL(hmm.Emission()[0].Component(sortedIndices[1]).Mean()[1] -
      gmms[0].Component(1).Mean()[1], 0.15);

  BOOST_REQUIRE_SMALL(hmm.Emission()[0].Component(sortedIndices[0]).
      Covariance()(0, 0) - gmms[0].Component(0).Covariance()(0, 0), 0.3);
  BOOST_REQUIRE_SMALL(hmm.Emission()[0].Component(sortedIndices[0]).
      Covariance()(0, 1) - gmms[0].Component(0).Covariance()(0, 1), 0.3);
  BOOST_REQUIRE_SMALL(hmm.Emission()[0].Component(sortedIndices[0]).
      Covariance()(1, 0) - gmms[0].Component(0).Covariance()(1, 0), 0.3);
  BOOST_REQUIRE_SMALL(hmm.Emission()[0].Component(sortedIndices[0]).
      Covariance()(1, 1) - gmms[0].Component(0).Covariance()(1, 1), 0.3);

  BOOST_REQUIRE_SMALL(hmm.Emission()[0].Component(sortedIndices[1]).
      Covariance()(0, 0) - gmms[0].Component(1).Covariance()(0, 0), 0.3);
  BOOST_REQUIRE_SMALL(hmm.Emission()[0].Component(sortedIndices[1]).
      Covariance()(0, 1) - gmms[0].Component(1).Covariance()(0, 1), 0.3);
  BOOST_REQUIRE_SMALL(hmm.Emission()[0].Component(sortedIndices[1]).
      Covariance()(1, 0) - gmms[0].Component(1).Covariance()(1, 0), 0.3);
  BOOST_REQUIRE_SMALL(hmm.Emission()[0].Component(sortedIndices[1]).
      Covariance()(1, 1) - gmms[0].Component(1).Covariance()(1, 1), 0.3);


  // Sort the GMM.
  sortedIndices = sort_index(hmm.Emission()[1].Weights());

  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Weights()[sortedIndices[0]] -
      gmms[1].Weights()[0], 0.08);
  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Weights()[sortedIndices[1]] -
      gmms[1].Weights()[1], 0.08);

  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Component(sortedIndices[0]).Mean()[0] -
      gmms[1].Component(0).Mean()[0], 0.15);
  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Component(sortedIndices[0]).Mean()[1] -
      gmms[1].Component(0).Mean()[1], 0.15);

  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Component(sortedIndices[1]).Mean()[0] -
      gmms[1].Component(1).Mean()[0], 0.15);
  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Component(sortedIndices[1]).Mean()[1] -
      gmms[1].Component(1).Mean()[1], 0.15);

  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Component(sortedIndices[0]).
      Covariance()(0, 0) - gmms[1].Component(0).Covariance()(0, 0), 0.3);
  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Component(sortedIndices[0]).
      Covariance()(0, 1) - gmms[1].Component(0).Covariance()(0, 1), 0.3);
  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Component(sortedIndices[0]).
      Covariance()(1, 0) - gmms[1].Component(0).Covariance()(1, 0), 0.3);
  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Component(sortedIndices[0]).
      Covariance()(1, 1) - gmms[1].Component(0).Covariance()(1, 1), 0.3);

  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Component(sortedIndices[1]).
      Covariance()(0, 0) - gmms[1].Component(1).Covariance()(0, 0), 0.3);
  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Component(sortedIndices[1]).
      Covariance()(0, 1) - gmms[1].Component(1).Covariance()(0, 1), 0.3);
  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Component(sortedIndices[1]).
      Covariance()(1, 0) - gmms[1].Component(1).Covariance()(1, 0), 0.3);
  BOOST_REQUIRE_SMALL(hmm.Emission()[1].Component(sortedIndices[1]).
      Covariance()(1, 1) - gmms[1].Component(1).Covariance()(1, 1), 0.3);
}

/**
 * Test saving and loading of GMM HMMs
 */
BOOST_AUTO_TEST_CASE(GMMHMMLoadSaveTest)
{
  // Create a GMM HMM, save it, and load it.
  HMM<GMM> hmm(3, GMM(4, 3));

  for(size_t j = 0; j < hmm.Emission().size(); ++j)
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
    ar << data::CreateNVP(hmm, "hmm");
  }

  // Load the HMM.
  HMM<GMM> hmm2(3, GMM(4, 3));
  {
    std::ifstream ifs("test-hmm-save.xml");
    boost::archive::xml_iarchive ar(ifs);
    ar >> data::CreateNVP(hmm2, "hmm");
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
          BOOST_REQUIRE_CLOSE(hmm.Emission()[j].Component(i).Covariance()(l,k),
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


  for(size_t j = 0; j < hmm.Emission().size(); ++j)
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
    ar << data::CreateNVP(hmm, "hmm");
  }

  // Load the HMM.
  HMM<GaussianDistribution> hmm2(3, GaussianDistribution(2));
  {
    std::ifstream ifs("test-hmm-save.xml");
    boost::archive::xml_iarchive ar(ifs);
    ar >> data::CreateNVP(hmm2, "hmm");
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
        BOOST_REQUIRE_CLOSE(hmm.Emission()[j].Covariance()(i,k),
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


  for(size_t j = 0; j < hmm.Emission().size(); ++j)
  {
    hmm.Emission()[j].Probabilities() = arma::randu<arma::vec>(3);
    hmm.Emission()[j].Probabilities() /= accu(emission[j].Probabilities());
  }

  // Save the HMM.
  {
    std::ofstream ofs("test-hmm-save.xml");
    boost::archive::xml_oarchive ar(ofs);
    ar << data::CreateNVP(hmm, "hmm");
  }

  // Load the HMM.
  HMM<DiscreteDistribution> hmm2(3, DiscreteDistribution(3));
  {
    std::ifstream ifs("test-hmm-save.xml");
    boost::archive::xml_iarchive ar(ifs);
    ar >> data::CreateNVP(hmm2, "hmm");
  }

  // Remove clutter.
  remove("test-hmm-save.xml");

  for (size_t j = 0; j < hmm.Emission().size(); ++j)
    for (size_t i = 0; i < hmm.Emission()[j].Probabilities().n_elem; ++i)
      BOOST_REQUIRE_CLOSE(hmm.Emission()[j].Probabilities()[i],
          hmm2.Emission()[j].Probabilities()[i], 1e-3);
}

BOOST_AUTO_TEST_SUITE_END();

