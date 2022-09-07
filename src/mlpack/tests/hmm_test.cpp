/**
 * @file tests/hmm_test.cpp
 *
 * Test file for HMMs.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/hmm.hpp>
#include <mlpack/methods/gmm.hpp>

#include "catch.hpp"
#include "test_catch_tools.hpp"

using namespace mlpack;

/**
 * We will use the simple case proposed by Russell and Norvig in Artificial
 * Intelligence: A Modern Approach, 2nd Edition, around p.549.
 */
TEST_CASE("SimpleDiscreteHMMTestViterbi", "[HMMTest]")
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
  REQUIRE(states[0] == 0); // Rain.
  REQUIRE(states[1] == 0); // Rain.
  REQUIRE(states[2] == 1); // No rain.
  REQUIRE(states[3] == 0); // Rain.
  REQUIRE(states[4] == 0); // Rain.
}

/**
 * This example is from Borodovsky & Ekisheva, p. 80-81.  It is just slightly
 * more complex.
 */
TEST_CASE("BorodovskyHMMTestViterbi", "[HMMTest]")
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
  REQUIRE(states[0] == 1);
  REQUIRE(states[1] == 1);
  REQUIRE(states[2] == 1);
  REQUIRE(states[3] == 2);
  // This could actually be one of two states (equal probability).
  REQUIRE(((states[4] == 1) || (states[4] == 2)) == true);
  REQUIRE(states[5] == 2);
  // This could also be one of two states.
  REQUIRE(((states[6] == 1) || (states[6] == 2)) == true);
  REQUIRE(states[7] == 2);
  REQUIRE(states[8] == 2);
}

/**
 * Ensure that the forward-backward algorithm is correct.
 */
TEST_CASE("ForwardBackwardTwoState", "[HMMTest]")
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
  REQUIRE(log == Approx(-23.4349).epsilon(1e-5));

  REQUIRE(stateProb(0, 0) == Approx(0.0).margin(1e-7));
  REQUIRE(stateProb(1, 0) == Approx(1.0).epsilon(1e-7));
  REQUIRE(stateProb(0, 1) == Approx(0.0).margin(1e-7));
  REQUIRE(stateProb(1, 1) == Approx(1.0).epsilon(1e-7));
  REQUIRE(stateProb(0, 2) == Approx(0.0).margin(1e-7));
  REQUIRE(stateProb(1, 2) == Approx(1.0).epsilon(1e-7));
  REQUIRE(stateProb(0, 3) == Approx(1.0).epsilon(1e-7));
  REQUIRE(stateProb(1, 3) == Approx(0.0).margin(1e-7));
  REQUIRE(stateProb(0, 4) == Approx(1.0).epsilon(1e-7));
  REQUIRE(stateProb(1, 4) == Approx(0.0).margin(1e-7));
  REQUIRE(stateProb(0, 5) == Approx(1.0).epsilon(1e-7));
  REQUIRE(stateProb(1, 5) == Approx(0.0).margin(1e-7));
  REQUIRE(stateProb(0, 6) == Approx(1.0).epsilon(1e-7));
  REQUIRE(stateProb(1, 6) == Approx(0.0).margin(1e-7));
  REQUIRE(stateProb(0, 7) == Approx(0.0).margin(1e-7));
  REQUIRE(stateProb(1, 7) == Approx(1.0).epsilon(1e-7));
  REQUIRE(stateProb(0, 8) == Approx(0.0).margin(1e-7));
  REQUIRE(stateProb(1, 8) == Approx(1.0).epsilon(1e-7));
  REQUIRE(stateProb(0, 9) == Approx(1.0).epsilon(1e-7));
  REQUIRE(stateProb(1, 9) == Approx(0.0).margin(1e-7));
}

/**
 * In this example we try to estimate the transmission and emission matrices
 * based on some observations.  We use the simplest possible model.
 */
TEST_CASE("SimplestBaumWelchDiscreteHMM", "[HMMTest]")
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

  REQUIRE(hmm.Initial()[0] == Approx(1.0).epsilon(1e-7));
  REQUIRE(hmm.Emission()[0].Probability("0") == Approx(1.0).epsilon(1e-7));
  REQUIRE(hmm.Transition()(0, 0) == Approx(1.0).epsilon(1e-7));
}

/**
 * A slightly more complex model to estimate.
 */
TEST_CASE("SimpleBaumWelchDiscreteHMM", "[HMMTest]")
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

  REQUIRE(hmm.Emission()[0].Probability("0") == Approx(0.5).epsilon(1e-7));
  REQUIRE(hmm.Emission()[0].Probability("1") == Approx(0.5).epsilon(1e-7));
  REQUIRE(hmm.Transition()(0, 0) == Approx(1.0).epsilon(1e-7));
  REQUIRE(hmm.Initial()[0] == Approx(1.0).epsilon(1e-7));
}

/**
 * Increasing complexity, but still simple; 4 emissions, 2 states; the state can
 * be determined directly by the emission.
 */
TEST_CASE("SimpleBaumWelchDiscreteHMM_2", "[HMMTest]")
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
  for (size_t i = 0; i < obsNum; ++i)
  {
    arma::mat observation(1, obsLen);

    size_t state = 0;
    size_t emission = 0;

    for (size_t obs = 0; obs < obsLen; obs++)
    {
      // See if state changed.
      double r = Random();

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
      r = Random();

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
  REQUIRE(hmm.Initial()[0] == Approx(prob).epsilon(0.025));
  REQUIRE(hmm.Initial()[1] == Approx(1.0 - prob).epsilon(0.025));

  REQUIRE(hmm.Transition()(0, 0) == Approx(0.5).epsilon(0.025));
  REQUIRE(hmm.Transition()(1, 0) == Approx(0.5).epsilon(0.025));
  REQUIRE(hmm.Transition()(0, 1) == Approx(0.5).epsilon(0.025));
  REQUIRE(hmm.Transition()(1, 1) == Approx(0.5).epsilon(0.025));

  REQUIRE(hmm.Emission()[0].Probability("0") == Approx(0.4).epsilon(0.04));
  REQUIRE(hmm.Emission()[0].Probability("1") == Approx(0.6).epsilon(0.04));
  REQUIRE(hmm.Emission()[0].Probability("2") == Approx(0.0).margin(2.5));
  REQUIRE(hmm.Emission()[0].Probability("3") == Approx(0.0).margin(2.5));
  REQUIRE(hmm.Emission()[1].Probability("0") == Approx(0.0).margin(2.5));
  REQUIRE(hmm.Emission()[1].Probability("1") == Approx(0.0).margin(2.5));
  REQUIRE(hmm.Emission()[1].Probability("2") == Approx(0.2).epsilon(0.04));
  REQUIRE(hmm.Emission()[1].Probability("3") == Approx(0.8).epsilon(0.04));
}

TEST_CASE("DiscreteHMMLabeledTrainTest", "[HMMTest]")
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
    states[n][0] = RandInt(3);

    // Random starting observation.
    observations[n].col(0) = emission[states[n][0]].Random();

    // Now the rest of the observations.
    for (size_t t = 1; t < obsLen; t++)
    {
      // Choose random number for state transition.
      double state = Random();

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
  REQUIRE(arma::norm(hmm.Initial() - initial) <  0.2);

  // Check that the transition matrix is close.
  REQUIRE(arma::norm(hmm.Transition() - transition) < 0.1);

  for (size_t col = 0; col < hmm.Emission().size(); col++)
  {
    for (size_t row = 0; row < hmm.Emission()[col].Probabilities().n_elem;
        row++)
    {
      arma::vec obs(1);
      obs[0] = row;
      REQUIRE(hmm.Emission()[col].Probability(obs) -
          emission[col].Probability(obs) == Approx(0.0).margin(0.07));
    }
  }
}

/**
 * Make sure the Generate() function works for a uniformly distributed HMM;
 * we'll take many samples just to make sure.
 */
TEST_CASE("DiscreteHMMSimpleGenerateTest", "[HMMTest]")
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
  for (size_t i = 0; i < 100000; ++i)
  {
    emissionProb[(size_t) dataSeq.col(i)[0] + 0.5]++;
    stateProb[stateSeq[i]]++;
  }

  // Normalize so these are probabilities.
  emissionProb /= accu(emissionProb);
  stateProb /= accu(stateProb);

  // Now check that the probabilities are right.  3% tolerance.
  REQUIRE(emissionProb[0] == Approx(0.25).epsilon(0.03));
  REQUIRE(emissionProb[1] == Approx(0.25).epsilon(0.03));
  REQUIRE(emissionProb[2] == Approx(0.25).epsilon(0.03));
  REQUIRE(emissionProb[3] == Approx(0.25).epsilon(0.03));

  REQUIRE(stateProb[0] == Approx(0.50).epsilon(0.03));
  REQUIRE(stateProb[1] == Approx(0.50).epsilon(0.03));
}

/**
 * More complex test for Generate().
 */
TEST_CASE("DiscreteHMMGenerateTest", "[HMMTest]")
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
  for (int i = 0; i < numSeq; ++i)
  {
    // Random starting state.
    size_t startState = RandInt(4);

    hmm.Generate(numObs, sequences[i], states[i], startState);
  }

  // Now we will calculate the full probabilities.
  HMM<DiscreteDistribution> hmm2(4, 6);
  hmm2.Train(sequences, states);

  // Check that training gives the same result.
  REQUIRE(arma::norm(hmm.Transition() - hmm2.Transition()) <  0.02);

  for (size_t row = 0; row < 6; row++)
  {
    arma::vec obs(1);
    obs[0] = row;
    for (size_t col = 0; col < 4; col++)
    {
      REQUIRE(hmm.Emission()[col].Probability(obs) -
          hmm2.Emission()[col].Probability(obs) == Approx(0.0).margin(0.02));
    }
  }
}

TEST_CASE("DiscreteHMMLogLikelihoodTest", "[HMMTest]")
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
  REQUIRE(hmm.LogLikelihood("0 1 2 3") == Approx(-4.9887223949).epsilon(1e-7));
  REQUIRE(hmm.LogLikelihood("1 2 0 0") == Approx(-6.0288487077).epsilon(1e-7));
  REQUIRE(hmm.LogLikelihood("3 3 3 3") == Approx(-5.5544000018).epsilon(1e-7));
  REQUIRE(hmm.LogLikelihood("0 2 2 1 2 3 0 0 1 3 1 0 0 3 1 2 2") ==
      Approx(-24.51556128368).epsilon(1e-7));
}

/**
 * A simple test to make sure HMMs with Gaussian output distributions work.
 */
TEST_CASE("GaussianHMMSimpleTest", "[HMMTest]")
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
  for (size_t i = 1; i < 1000; ++i)
  {
    double randValue = Random();

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
  for (size_t i = 0; i < 1000; ++i)
  {
    REQUIRE(predictedClasses[i] == classes[i]);

    // The probability of the wrong class should be infinitesimal.
    REQUIRE(stateProb((classes[i] + 1) % 2, i) == Approx(0.0).margin(0.001));
  }
}

/**
 * Ensure that Gaussian HMMs can be trained properly, for the labeled training
 * case and also for the unlabeled training case.
 */
TEST_CASE("GaussianHMMTrainTest", "[HMMTest]")
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
      double randValue = Random();
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
  REQUIRE(hmm.Initial()[0] == Approx(1.0).epsilon(1e-7));
  REQUIRE(hmm.Initial()[1] == Approx(0.0).margin(1e-3));
  REQUIRE(hmm.Initial()[2] == Approx(0.0).margin(1e-3));

  // We use a tolerance of 0.05 for the transition matrices.
  // Check that the transition matrix is correct.
  REQUIRE(arma::norm(hmm.Transition() - transition) < 0.05);

  // Check that each distribution is correct.
  for (size_t dist = 0; dist < 3; dist++)
  {
    REQUIRE(arma::norm(hmm.Emission()[dist].Mean() -
        emission[dist].Mean()) < 0.05);
    REQUIRE(arma::norm(hmm.Emission()[dist].Covariance() -
        emission[dist].Covariance()) < 0.1);
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

  REQUIRE(hmm.Initial()[0] == Approx(1.0).epsilon(0.001));
  REQUIRE(hmm.Initial()[1] == Approx(0.0).margin(0.05));
  REQUIRE(hmm.Initial()[2] == Approx(0.0).margin(0.05));

  // The tolerances are increased because there is more error in unlabeled
  // training; we use an absolute tolerance of 0.03 for the transition matrices.
  // Check that the transition matrix is correct.
  for (size_t row = 0; row < 3; row++)
    for (size_t col = 0; col < 3; col++)
      REQUIRE(transition(row, col) - hmm.Transition()(row, col) ==
          Approx(.0).margin(0.03));

  // Check that each distribution is correct.
  for (size_t dist = 0; dist < 3; dist++)
  {
    REQUIRE(arma::norm(hmm.Emission()[dist].Mean() -
        emission[dist].Mean()) < 0.1);
    REQUIRE(arma::norm(hmm.Emission()[dist].Covariance() -
        emission[dist].Covariance()) < 0.25);
  }
}

/**
 * Make sure that a random sequence generated by a Gaussian HMM fits the
 * distribution correctly.
 */
TEST_CASE("GaussianHMMGenerateTest", "[HMMTest]")
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
  REQUIRE(arma::norm(hmm.Transition() - hmm2.Transition()) < 0.1);

  // Check that each Gaussian is the same.
  for (size_t dist = 0; dist < 3; dist++)
  {
    REQUIRE(arma::norm(hmm.Emission()[dist].Mean() -
        hmm2.Emission()[dist].Mean()) < 0.2);
    REQUIRE(arma::norm(hmm.Emission()[dist].Covariance() -
        hmm2.Emission()[dist].Covariance()) < 0.3);
  }
}

/**
 * Make sure that Predict() is numerically stable.
 */
TEST_CASE("GaussianHMMPredictTest", "[HMMTest]")
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

  // 100 2D observations.
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

  // 100 pre-calculated emission probabilities each for 10 states.
  std::vector<arma::vec> emissionProb = {
    { -2.7301e+03, 1.7874e+00, -1.9428e+00, -3.6365e+00, -4.0397e-01,
            -1.5115e-01, -1.0328e+00, -1.1071e+00, 5.2876e-01, -1.0643e-01 },
    { -2.3684e+03, 1.8059e+00, -2.2058e+00, -4.0514e+00, -5.0935e-01,
            -2.1126e-01, -1.1962e+00, -1.2567e+00, 4.1247e-01, -3.0199e-01 },
    { -1.7117e+03, 1.7981e+00, -2.5275e+00, -4.5634e+00, -6.4839e-01,
            -2.9579e-01, -1.4000e+00, -1.4461e+00, 2.3795e-01, -5.5622e-01 },
    { -1.3089e+03, 1.7393e+00, -2.8685e+00, -5.0996e+00, -8.0288e-01,
            -3.9863e-01, -1.6229e+00, -1.6478e+00, 2.3300e-02, -8.6617e-01 },
    { -1.3541e+03, 1.6414e+00, -3.1971e+00, -5.6043e+00, -9.5605e-01,
            -5.1013e-01, -1.8460e+00, -1.8395e+00, -2.0603e-01, -1.2176e+00 },
    { -1.5521e+03, 1.5367e+00, -3.4806e+00, -6.0349e+00, -1.0924e+00,
            -6.1426e-01, -2.0436e+00, -2.0051e+00, -4.2045e-01, -1.5500e+00 },
    { -1.2647e+03, 1.4680e+00, -3.6577e+00, -6.3144e+00, -1.1823e+00,
            -6.8009e-01, -2.1646e+00, -2.1147e+00, -5.6512e-01, -1.7360e+00 },
    { -3.2650e+02, 1.4646e+00, -3.6693e+00, -6.3649e+00, -1.1957e+00,
            -6.7711e-01, -2.1592e+00, -2.1377e+00, -5.8400e-01, -1.6543e+00 },
    { -1.3035e+02, 1.5123e+00, -3.5018e+00, -6.1593e+00, -1.1254e+00,
            -6.0037e-01, -2.0181e+00, -2.0646e+00, -4.6413e-01, -1.3011e+00 },
    { -2.6279e+03, 1.5809e+00, -3.1559e+00, -5.6861e+00, -9.7699e-01,
            -4.5903e-01, -1.7490e+00, -1.8956e+00, -2.2135e-01, -7.2772e-01 },
    { -9.6164e+03, 1.6193e+00, -2.6708e+00, -4.9944e+00, -7.8159e-01,
            -2.8441e-01, -1.3909e+00, -1.6574e+00, 7.8411e-02, -5.0595e-02 },
    { -2.0944e+04, 1.5980e+00, -2.1094e+00, -4.1681e+00, -5.8143e-01,
            -1.2105e-01, -1.0055e+00, -1.3879e+00, 3.4591e-01, 5.6464e-01 },
    { -3.3843e+04, 1.5241e+00, -1.5331e+00, -3.2977e+00, -4.1342e-01,
            -8.0772e-03, -6.5026e-01, -1.1226e+00, 5.0244e-01, 9.8522e-01 },
    { -4.6678e+04, 1.3796e+00, -9.8223e-01, -2.4507e+00, -3.0368e-01,
            3.0609e-02, -3.5787e-01, -8.8913e-01, 4.9234e-01, 1.1530e+00 },
    { -6.0839e+04, 1.1013e+00, -4.8302e-01, -1.6712e+00, -2.7541e-01,
            -2.2814e-02, -1.4691e-01, -7.1095e-01, 2.6698e-01, 1.0338e+00 },
    { -7.8940e+04, 6.2341e-01, -6.4826e-02, -1.0034e+00, -3.5198e-01,
            -1.8517e-01, -3.7803e-02, -6.1240e-01, -2.1704e-01, 5.8353e-01 },
    { -1.0182e+05, -8.9362e-02, 2.5888e-01, -4.6429e-01, -5.5297e-01,
            -4.7752e-01, -4.9871e-02, -6.0739e-01, -1.0089e+00, -2.6587e-01 },
    { -1.2437e+05, -9.8625e-01, 4.7236e-01, -8.1256e-02, -8.8097e-01,
            -9.0979e-01, -2.0039e-01, -6.9837e-01, -2.1229e+00, -1.5424e+00 },
    { -1.3878e+05, -1.9546e+00, 5.6976e-01, 1.2361e-01, -1.3043e+00,
            -1.4534e+00, -4.7690e-01, -8.6807e-01, -3.4831e+00, -3.1393e+00 },
    { -1.3979e+05, -2.8896e+00, 5.6962e-01, 1.6577e-01, -1.7631e+00,
            -2.0456e+00, -8.3102e-01, -1.0792e+00, -4.9380e+00, -4.8430e+00 },
    { -1.2717e+05, -3.7474e+00, 5.0493e-01, 9.2444e-02, -2.1969e+00,
            -2.6201e+00, -1.2028e+00, -1.2907e+00, -6.3319e+00, -6.4416e+00 },
    { -1.0548e+05, -4.5565e+00, 4.0397e-01, -4.5775e-02, -2.5711e+00,
            -3.1354e+00, -1.5493e+00, -1.4771e+00, -7.5697e+00, -7.8170e+00 },
    { -8.0621e+04, -5.3691e+00, 2.8365e-01, -2.1252e-01, -2.8783e+00,
            -3.5784e+00, -1.8523e+00, -1.6312e+00, -8.6245e+00, -8.9480e+00 },
    { -5.6310e+04, -6.2411e+00, 1.5008e-01, -3.9022e-01, -3.1294e+00,
            -3.9597e+00, -2.1142e+00, -1.7569e+00, -9.5239e+00, -9.8785e+00 },
    { -3.4173e+04, -7.2306e+00, 3.0396e-03, -5.7242e-01, -3.3347e+00,
            -4.2928e+00, -2.3417e+00, -1.8583e+00, -1.0301e+01, -1.0652e+01 },
    { -1.5877e+04, -8.3900e+00, -1.5871e-01, -7.5362e-01, -3.4959e+00,
            -4.5816e+00, -2.5356e+00, -1.9353e+00, -1.0963e+01, -1.1284e+01 },
    { -3.3829e+03, -9.7572e+00, -3.3006e-01, -9.1554e-01, -3.5912e+00,
            -4.8035e+00, -2.6770e+00, -1.9722e+00, -1.1452e+01, -1.1714e+01 },
    { -5.6088e+02, -1.1394e+01, -5.0305e-01, -1.0261e+00, -3.5777e+00,
            -4.9138e+00, -2.7301e+00, -1.9403e+00, -1.1653e+01, -1.1829e+01 },
    { -1.4303e+04, -1.3346e+01, -6.7336e-01, -1.0564e+00, -3.4266e+00,
            -4.8757e+00, -2.6690e+00, -1.8219e+00, -1.1470e+01, -1.1561e+01 },
    { -4.9066e+04, -1.5534e+01, -8.4176e-01, -1.0079e+00, -3.1636e+00,
            -4.7028e+00, -2.5116e+00, -1.6369e+00, -1.0937e+01, -1.0995e+01 },
    { -9.9717e+04, -1.7702e+01, -1.0039e+00, -9.1443e-01, -2.8597e+00,
            -4.4595e+00, -2.3138e+00, -1.4339e+00, -1.0224e+01, -1.0331e+01 },
    { -1.5886e+05, -1.9676e+01, -1.1535e+00, -7.9762e-01, -2.5479e+00,
            -4.1805e+00, -2.1039e+00, -1.2332e+00, -9.4233e+00, -9.6530e+00 },
    { -2.2947e+05, -2.1635e+01, -1.3117e+00, -6.6325e-01, -2.2133e+00,
            -3.8587e+00, -1.8780e+00, -1.0253e+00, -8.5051e+00, -8.9416e+00 },
    { -3.1968e+05, -2.3792e+01, -1.5095e+00, -5.1672e-01, -1.8381e+00,
            -3.4770e+00, -1.6312e+00, -8.0190e-01, -7.4108e+00, -8.1836e+00 },
    { -4.3323e+05, -2.6183e+01, -1.7728e+00, -3.8390e-01, -1.4394e+00,
            -3.0487e+00, -1.3857e+00, -5.7953e-01, -6.1647e+00, -7.4521e+00 },
    { -5.6473e+05, -2.8589e+01, -2.1061e+00, -3.0168e-01, -1.0547e+00,
            -2.6054e+00, -1.1773e+00, -3.8708e-01, -4.8475e+00, -6.8476e+00 },
    { -6.9974e+05, -3.0612e+01, -2.4921e+00, -3.0913e-01, -7.2535e-01,
            -2.1849e+00, -1.0419e+00, -2.5359e-01, -3.5677e+00, -6.4479e+00 },
    { -8.0655e+05, -3.1539e+01, -2.8524e+00, -4.2185e-01, -4.8692e-01,
            -1.8260e+00, -9.9484e-01, -1.9514e-01, -2.4629e+00, -6.2373e+00 },
    { -8.5216e+05, -3.0655e+01, -3.0833e+00, -6.1881e-01, -3.3169e-01,
            -1.5249e+00, -1.0091e+00, -1.9595e-01, -1.5717e+00, -6.0513e+00 },
    { -8.2392e+05, -2.7811e+01, -3.1362e+00, -8.7526e-01, -2.3459e-01,
            -1.2631e+00, -1.0480e+00, -2.3278e-01, -8.7344e-01, -5.7341e+00 },
    { -7.3612e+05, -2.3582e+01, -3.0425e+00, -1.1744e+00, -1.7841e-01,
            -1.0351e+00, -1.0893e+00, -2.9210e-01, -3.4780e-01, -5.2495e+00 },
    { -6.1397e+05, -1.8706e+01, -2.8744e+00, -1.5195e+00, -1.5516e-01,
            -8.3816e-01, -1.1304e+00, -3.7330e-01, 3.7744e-02, -4.6424e+00 },
    { -4.8041e+05, -1.3799e+01, -2.7054e+00, -1.9262e+00, -1.6637e-01,
            -6.7558e-01, -1.1810e+00, -4.8405e-01, 3.0197e-01, -3.9898e+00 },
    { -3.4790e+05, -9.2300e+00, -2.5518e+00, -2.3683e+00, -2.0524e-01,
            -5.4582e-01, -1.2297e+00, -6.1666e-01, 4.5657e-01, -3.3063e+00 },
    { -2.2370e+05, -5.1887e+00, -2.3941e+00, -2.7911e+00, -2.5500e-01,
            -4.3560e-01, -1.2487e+00, -7.5224e-01, 5.3161e-01, -2.5570e+00 },
    { -1.1273e+05, -1.7195e+00, -2.2258e+00, -3.1794e+00, -3.0915e-01,
            -3.2974e-01, -1.2221e+00, -8.8867e-01, 5.5755e-01, -1.7017e+00 },
    { -2.8363e+04, 9.0588e-01, -2.0601e+00, -3.5233e+00, -3.7171e-01,
            -2.2162e-01, -1.1370e+00, -1.0334e+00, 5.4434e-01, -7.3209e-01 },
    { -1.2122e+03, 1.9784e+00, -1.9455e+00, -3.7971e+00, -4.5862e-01,
            -1.2081e-01, -9.8979e-01, -1.2000e+00, 4.7839e-01, 2.5783e-01 },
    { -7.1694e+04, 5.6327e-01, -2.0051e+00, -4.0345e+00, -6.1306e-01,
            -6.6602e-02, -8.2833e-01, -1.4287e+00, 3.0684e-01, 1.0022e+00 },
    { -2.6198e+05, -3.8345e+00, -2.3396e+00, -4.2798e+00, -8.6900e-01,
            -9.8822e-02, -7.1489e-01, -1.7486e+00, -1.6219e-02, 1.2358e+00 },
    { -5.5328e+05, -1.0687e+01, -2.9124e+00, -4.5058e+00, -1.2121e+00,
            -2.2259e-01, -6.7273e-01, -2.1347e+00, -4.7585e-01, 8.8080e-01 },
    { -8.9436e+05, -1.8602e+01, -3.5518e+00, -4.6037e+00, -1.5911e+00,
            -4.1140e-01, -6.7958e-01, -2.5173e+00, -1.0137e+00, 3.7886e-02 },
    { -1.2162e+06, -2.5781e+01, -4.0541e+00, -4.4848e+00, -1.9485e+00,
            -6.3137e-01, -7.0903e-01, -2.8240e+00, -1.5699e+00, -1.1063e+00 },
    { -1.4436e+06, -3.0414e+01, -4.2395e+00, -4.1197e+00, -2.2265e+00,
            -8.4654e-01, -7.3852e-01, -2.9921e+00, -2.0869e+00, -2.2970e+00 },
    { -1.5227e+06, -3.1337e+01, -3.9989e+00, -3.5197e+00, -2.3836e+00,
            -1.0315e+00, -7.4887e-01, -2.9823e+00, -2.5313e+00, -3.3017e+00 },
    { -1.4386e+06, -2.8472e+01, -3.3472e+00, -2.7563e+00, -2.4087e+00,
            -1.1801e+00, -7.3524e-01, -2.7971e+00, -2.9034e+00, -3.9803e+00 },
    { -1.2257e+06, -2.2958e+01, -2.4364e+00, -1.9521e+00, -2.3275e+00,
            -1.3043e+00, -7.0875e-01, -2.4858e+00, -3.2295e+00, -4.3252e+00 },
    { -9.4813e+05, -1.6527e+01, -1.4675e+00, -1.2121e+00, -2.1965e+00,
            -1.4367e+00, -6.9389e-01, -2.1228e+00, -3.5740e+00, -4.4844e+00 },
    { -6.6589e+05, -1.0680e+01, -6.1313e-01, -6.1440e-01, -2.0638e+00,
            -1.5984e+00, -7.0917e-01, -1.7727e+00, -3.9726e+00, -4.5979e+00 },
    { -4.1809e+05, -6.2975e+00, 3.1651e-02, -1.8731e-01, -1.9586e+00,
            -1.7982e+00, -7.6241e-01, -1.4730e+00, -4.4365e+00, -4.7645e+00 },
    { -2.2534e+05, -3.7546e+00, 4.3188e-01, 7.1872e-02, -1.8959e+00,
            -2.0366e+00, -8.5455e-01, -1.2417e+00, -4.9637e+00, -5.0424e+00 },
    { -9.4330e+04, -3.0403e+00, 5.9517e-01, 1.8422e-01, -1.8702e+00,
            -2.2952e+00, -9.7314e-01, -1.0776e+00, -5.5109e+00, -5.4155e+00 },
    { -2.1454e+04, -3.9202e+00, 5.5647e-01, 1.8381e-01, -1.8704e+00,
            -2.5578e+00, -1.1056e+00, -9.6899e-01, -6.0419e+00, -5.8579e+00 },
    { -31.4830, -6.0953, 0.3567, 0.1044, -1.8840, -2.8086, -1.2397,
            -0.9026, -6.5224, -6.3374 },
    { -2.2442e+04, -9.2735e+00, 3.4960e-02, -2.1605e-02, -1.8931e+00,
            -3.0282e+00, -1.3611e+00, -8.6066e-01, -6.9076e+00, -6.8075e+00 },
    { -8.1676e+04, -1.3138e+01, -3.6831e-01, -1.6104e-01, -1.8763e+00,
            -3.1905e+00, -1.4522e+00, -8.2511e-01, -7.1362e+00, -7.2081e+00 },
    { -1.6865e+05, -1.7287e+01, -8.0643e-01, -2.8264e-01, -1.8178e+00,
            -3.2726e+00, -1.4987e+00, -7.8144e-01, -7.1585e+00, -7.4877e+00 },
    { -2.7001e+05, -2.1213e+01, -1.2247e+00, -3.6116e-01, -1.7095e+00,
            -3.2596e+00, -1.4928e+00, -7.2002e-01, -6.9485e+00, -7.6058e+00 },
    { -3.7506e+05, -2.4628e+01, -1.5962e+00, -3.9394e-01, -1.5583e+00,
            -3.1610e+00, -1.4428e+00, -6.4101e-01, -6.5350e+00, -7.5763e+00 },
    { -4.7871e+05, -2.7455e+01, -1.9194e+00, -3.9090e-01, -1.3720e+00,
            -2.9900e+00, -1.3606e+00, -5.4763e-01, -5.9492e+00, -7.4279e+00 },
    { -5.7329e+05, -2.9501e+01, -2.1830e+00, -3.6323e-01, -1.1594e+00,
            -2.7564e+00, -1.2564e+00, -4.4501e-01, -5.2194e+00, -7.1738e+00 },
    { -6.4968e+05, -3.0560e+01, -2.3747e+00, -3.2775e-01, -9.3375e-01,
            -2.4742e+00, -1.1428e+00, -3.4141e-01, -4.3880e+00, -6.8281e+00 },
    { -6.9933e+05, -3.0501e+01, -2.4875e+00, -3.0631e-01, -7.1262e-01,
            -2.1653e+00, -1.0343e+00, -2.4789e-01, -3.5174e+00, -6.4120e+00 },
    { -7.1802e+05, -2.9350e+01, -2.5271e+00, -3.2061e-01, -5.1194e-01,
            -1.8521e+00, -9.4328e-01, -1.7450e-01, -2.6686e+00, -5.9486e+00 },
    { -7.0553e+05, -2.7236e+01, -2.5060e+00, -3.8730e-01, -3.4217e-01,
            -1.5515e+00, -8.7707e-01, -1.2819e-01, -1.8857e+00, -5.4542e+00 },
    { -6.6569e+05, -2.4393e+01, -2.4435e+00, -5.1663e-01, -2.1031e-01,
            -1.2775e+00, -8.3941e-01, -1.1339e-01, -1.2023e+00, -4.9470e+00 },
    { -6.1301e+05, -2.1269e+01, -2.3992e+00, -7.3370e-01, -1.2064e-01,
            -1.0383e+00, -8.4300e-01, -1.3855e-01, -6.1878e-01, -4.4864e+00 },
    { -5.6195e+05, -1.8233e+01, -2.4507e+00, -1.0921e+00, -8.5743e-02,
            -8.4467e-01, -9.1749e-01, -2.2378e-01, -1.3441e-01, -4.1677e+00 },
    { -5.0308e+05, -1.5078e+01, -2.5824e+00, -1.6122e+00, -1.1850e-01,
            -7.0720e-01, -1.0690e+00, -3.7916e-01, 2.0948e-01, -3.9737e+00 },
    { -4.2417e+05, -1.1613e+01, -2.7333e+00, -2.2592e+00, -2.1392e-01,
            -6.2529e-01, -1.2691e+00, -5.9232e-01, 3.8196e-01, -3.8021e+00 },
    { -3.4311e+05, -8.4490e+00, -2.9262e+00, -2.9840e+00, -3.6201e-01,
            -6.0612e-01, -1.5036e+00, -8.4657e-01, 3.8172e-01, -3.6994e+00 },
    { -2.6553e+05, -5.7959e+00, -3.0657e+00, -3.6135e+00, -5.0450e-01,
            -6.1056e-01, -1.6893e+00, -1.0726e+00, 2.9263e-01, -3.5310e+00 },
    { -1.6581e+05, -2.8806e+00, -2.9242e+00, -3.9480e+00, -5.5108e-01,
            -5.3603e-01, -1.6743e+00, -1.1832e+00, 2.8121e-01, -2.8215e+00 },
    { -6.3112e+04, -4.4355e-02, -2.4673e+00, -3.8848e+00, -4.8010e-01,
            -3.5415e-01, -1.4075e+00, -1.1547e+00, 4.0803e-01, -1.5227e+00 },
    { -5.4196e+03, 1.6750e+00, -1.9272e+00, -3.5655e+00, -3.8433e-01,
            -1.5578e-01, -1.0312e+00, -1.0745e+00, 5.4628e-01, -1.8838e-01 },
    { -7.9742e+03, 1.9542e+00, -1.5297e+00, -3.2224e+00, -3.5023e-01,
            -2.9557e-02, -7.1541e-01, -1.0340e+00, 5.7335e-01, 7.0234e-01 },
    { -4.6838e+04, 1.3383e+00, -1.2840e+00, -2.9202e+00, -3.6943e-01,
            2.1035e-02, -4.9879e-01, -1.0295e+00, 5.0388e-01, 1.1296e+00 },
    { -9.2965e+04, 5.0293e-01, -1.1033e+00, -2.6251e+00, -4.0461e-01,
            2.4992e-02, -3.5062e-01, -1.0246e+00, 3.8909e-01, 1.2595e+00 },
    { -1.3250e+05, -2.3738e-01, -9.9398e-01, -2.4136e+00, -4.4740e-01,
            6.0968e-03, -2.6559e-01, -1.0308e+00, 2.6684e-01, 1.2440e+00 },
    { -1.7149e+05, -9.7999e-01, -9.1698e-01, -2.2384e+00, -4.9912e-01,
            -2.5475e-02, -2.0762e-01, -1.0468e+00, 1.3078e-01, 1.1599e+00 },
    { -2.2091e+05, -1.9350e+00, -8.5497e-01, -2.0582e+00, -5.7508e-01,
            -7.7816e-02, -1.6138e-01, -1.0794e+00, -5.6045e-02, 9.8875e-01 },
    { -2.8140e+05, -3.1219e+00, -8.2568e-01, -1.8962e+00, -6.7862e-01,
            -1.5242e-01, -1.3554e-01, -1.1353e+00, -2.9320e-01, 7.2082e-01 },
    { -3.4167e+05, -4.3171e+00, -8.2824e-01, -1.7733e+00, -7.8907e-01,
            -2.3483e-01, -1.3167e-01, -1.2015e+00, -5.3627e-01, 4.0854e-01 },
    { -3.7868e+05, -5.0537e+00, -8.3691e-01, -1.7046e+00, -8.6035e-01,
            -2.9036e-01, -1.3690e-01, -1.2447e+00, -6.9304e-01, 1.9260e-01 },
    { -3.7429e+05, -4.9406e+00, -7.8456e-01, -1.6323e+00, -8.6203e-01,
            -3.0644e-01, -1.3159e-01, -1.2267e+00, -7.3358e-01, 1.3468e-01 },
    { -3.3293e+05, -4.0758e+00, -6.6873e-01, -1.5416e+00, -8.0032e-01,
            -2.8877e-01, -1.1346e-01, -1.1498e+00, -6.7458e-01, 2.1365e-01 },
    { -2.7541e+05, -2.9085e+00, -5.3210e-01, -1.4470e+00, -7.0706e-01,
            -2.5517e-01, -9.1435e-02, -1.0445e+00, -5.6402e-01, 3.5113e-01 },
    { -2.2010e+05, -1.8209e+00, -4.1116e-01, -1.3627e+00, -6.1220e-01,
            -2.2144e-01, -7.3319e-02, -9.4005e-01, -4.4660e-01, 4.7992e-01 },
    { -1.7809e+05, -1.0242e+00, -3.2646e-01, -1.3011e+00, -5.3612e-01,
            -1.9567e-01, -6.2291e-02, -8.5731e-01, -3.5032e-01, 5.7022e-01 },
    { -1.5426e+05, -5.8691e-01, -2.8121e-01, -1.2660e+00, -4.9111e-01,
            -1.8141e-01, -5.7387e-02, -8.0842e-01, -2.9317e-01, 6.1601e-01 },
  };

  const double loglikelihoodRef = -2734.43;

  // Test log-likelihood calculation for the whole data.
  {
    const double loglikelihood = hmm.LogLikelihood(obs);
    REQUIRE(loglikelihood == Approx(loglikelihoodRef).epsilon(1e-3));
  }

  // Test loglikelihood calculation in an incremental way.
  // It simulates the case where we have a stream of data.
  {
    double loglikelihood;
    arma::vec forwardLogProb;
    for (size_t t = 0; t<obs.n_cols; ++t)
    {
      loglikelihood = hmm.LogLikelihood(obs.col(t), loglikelihood,
                                        forwardLogProb);
    }
    REQUIRE(loglikelihood == Approx(loglikelihoodRef).epsilon(1e-3));
  }

  // Test loglikelihood calculation in an incremental way.
  // It simulates the case where we have a stream of data.
  // In this case the accumulation of the log scales factor to calculate
  // the log-likelihood value is done outside of the loop
  {
    double loglikelihood = 0;
    arma::vec forwardLogProb;
    for (size_t t = 0; t<obs.n_cols; ++t)
    {
      double logScale = hmm.LogScaleFactor(obs.col(t), forwardLogProb);
      loglikelihood += logScale;
    }
    REQUIRE(loglikelihood == Approx(loglikelihoodRef).epsilon(1e-3));
  }

  // Test loglikelihood calculation in an incremental way.
  // It simulates the case where we have emission probabilities pre-calculated.
  {
    double loglikelihood = 0;
    arma::vec forwardLogProb;
    for (size_t t = 0; t < emissionProb.size(); ++t)
    {
      loglikelihood = hmm.EmissionLogLikelihood(emissionProb.at(t),
                                                loglikelihood,
                                                forwardLogProb);
    }
    REQUIRE(loglikelihood == Approx(loglikelihoodRef).epsilon(1e-1));
  }

  arma::Row<size_t> stateSeq;
  hmm.Predict(obs, stateSeq);

  arma::Row<size_t> stateSeqRef = { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4,
      4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9,
      9, 9, 9, 9, 9, 9, 9, 9, 9, 9 };

  for (size_t i = 0; i < stateSeqRef.n_cols; ++i)
  {
    REQUIRE(stateSeqRef.at(i) == stateSeq.at(i));
  }
}

/**
 * Test that HMMs work with Gaussian mixture models.  We'll try putting in a
 * simple model by hand and making sure that prediction of observation sequences
 * works correctly.
 */
TEST_CASE("GMMHMMPredictTest", "[HMMTest]")
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

    for (size_t i = 1; i < 1000; ++i)
    {
      double randValue = Random();

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
    for (size_t i = 0; i < 1000; ++i)
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

  REQUIRE(success == true);
}

/**
 * Test that GMM-based HMMs can train on models correctly using labeled training
 * data.
 */
TEST_CASE("GMMHMMLabeledTrainingTest", "[HMMTest]")
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

    for (size_t i = 1; i < 2500; ++i)
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
  REQUIRE(hmm.Initial()[0] == Approx(1.0).epsilon(0.0001));
  REQUIRE(hmm.Initial()[1] == Approx(.0).margin(0.01));

  // Check the results.  Use absolute tolerances instead of percentages.
  REQUIRE(hmm.Transition()(0, 0) - transMat(0, 0) == Approx(.0).margin(0.03));
  REQUIRE(hmm.Transition()(0, 1) - transMat(0, 1) == Approx(.0).margin(0.03));
  REQUIRE(hmm.Transition()(1, 0) - transMat(1, 0) == Approx(.0).margin(0.03));
  REQUIRE(hmm.Transition()(1, 1) - transMat(1, 1) == Approx(.0).margin(0.03));

  // Now the emission probabilities (the GMMs).
  // We have to sort each GMM for comparison.
  arma::uvec sortedIndices = sort_index(hmm.Emission()[0].Weights());

  REQUIRE(hmm.Emission()[0].Weights()[sortedIndices[0]] -
      gmms[0].Weights()[0] == Approx(.0).margin(0.08));
  REQUIRE(hmm.Emission()[0].Weights()[sortedIndices[1]] -
      gmms[0].Weights()[1] == Approx(.0).margin(0.08));

  REQUIRE(arma::norm(
      hmm.Emission()[0].Component(sortedIndices[0]).Mean() -
      gmms[0].Component(0).Mean()) < 0.2);
  REQUIRE(arma::norm(
      hmm.Emission()[0].Component(sortedIndices[1]).Mean() -
      gmms[0].Component(1).Mean()) < 0.2);

  REQUIRE(arma::norm(
      hmm.Emission()[0].Component(sortedIndices[0]).Covariance() -
      gmms[0].Component(0).Covariance()) < 0.5);
  REQUIRE(arma::norm(
      hmm.Emission()[0].Component(sortedIndices[1]).Covariance() -
      gmms[0].Component(0).Covariance()) < 0.5);

  // Sort the GMM.
  sortedIndices = sort_index(hmm.Emission()[1].Weights());

  REQUIRE(hmm.Emission()[1].Weights()[sortedIndices[0]] -
      gmms[1].Weights()[0] == Approx(.0).margin(0.08));
  REQUIRE(hmm.Emission()[1].Weights()[sortedIndices[1]] -
      gmms[1].Weights()[1] == Approx(.0).margin(0.08));

  REQUIRE(arma::norm(
      hmm.Emission()[1].Component(sortedIndices[0]).Mean() -
      gmms[1].Component(0).Mean()) < 0.2);
  REQUIRE(arma::norm(
      hmm.Emission()[1].Component(sortedIndices[1]).Mean() -
      gmms[1].Component(1).Mean()) < 0.2);

  REQUIRE(arma::norm(
      hmm.Emission()[1].Component(sortedIndices[0]).Covariance() -
      gmms[1].Component(0).Covariance()) < 0.5);
  REQUIRE(arma::norm(
      hmm.Emission()[1].Component(sortedIndices[1]).Covariance() -
      gmms[1].Component(1).Covariance()) < 0.5);
}

/**
 * Test saving and loading of GMM HMMs
 */
TEST_CASE("GMMHMMLoadSaveTest", "[HMMTest]")
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
    cereal::XMLOutputArchive ar(ofs);
    ar(CEREAL_NVP(hmm));
  }

  // Load the HMM.
  HMM<GMM> hmm2(3, GMM(4, 3));
  {
    std::ifstream ifs("test-hmm-save.xml");
    cereal::XMLInputArchive ar(ifs);
    ar(cereal::make_nvp("hmm", hmm2));
  }

  // Remove clutter.
  remove("test-hmm-save.xml");

  for (size_t j = 0; j < hmm.Emission().size(); ++j)
  {
    REQUIRE(hmm.Emission()[j].Gaussians() ==
                        hmm2.Emission()[j].Gaussians());
    REQUIRE(hmm.Emission()[j].Dimensionality() ==
                        hmm2.Emission()[j].Dimensionality());

    for (size_t i = 0; i < hmm.Emission()[j].Dimensionality(); ++i)
      REQUIRE(hmm.Emission()[j].Weights()[i] ==
          Approx(hmm2.Emission()[j].Weights()[i]).epsilon(1e-5));

    for (size_t i = 0; i < hmm.Emission()[j].Gaussians(); ++i)
    {
      for (size_t l = 0; l < hmm.Emission()[j].Dimensionality(); ++l)
      {
        REQUIRE(hmm.Emission()[j].Component(i).Mean()[l] ==
            Approx(hmm2.Emission()[j].Component(i).Mean()[l]).epsilon(1e-5));

        for (size_t k = 0; k < hmm.Emission()[j].Dimensionality(); ++k)
        {
          REQUIRE(hmm.Emission()[j].Component(i).Covariance()(l, k) ==
              Approx(hmm2.Emission()[j].Component(i).Covariance()(l,
              k)).epsilon(1e-5));
        }
      }
    }
  }
}

/**
 * Test saving and loading of Gaussian HMMs
 */
TEST_CASE("GaussianHMMLoadSaveTest", "[HMMTest]")
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
    cereal::XMLOutputArchive ar(ofs);
    ar(cereal::make_nvp("hmm", hmm));
  }

  // Load the HMM.
  HMM<GaussianDistribution> hmm2(3, GaussianDistribution(2));
  {
    std::ifstream ifs("test-hmm-save.xml");
    cereal::XMLInputArchive ar(ifs);
    ar(cereal::make_nvp("hmm", hmm2));
  }

  // Remove clutter.
  remove("test-hmm-save.xml");

  for (size_t j = 0; j < hmm.Emission().size(); ++j)
  {
    REQUIRE(hmm.Emission()[j].Dimensionality() ==
                        hmm2.Emission()[j].Dimensionality());

    for (size_t i = 0; i < hmm.Emission()[j].Dimensionality(); ++i)
    {
        REQUIRE(hmm.Emission()[j].Mean()[i] ==
            Approx(hmm2.Emission()[j].Mean()[i]).epsilon(1e-5));

      for (size_t k = 0; k < hmm.Emission()[j].Dimensionality(); ++k)
      {
        REQUIRE(hmm.Emission()[j].Covariance()(i, k) ==
            Approx(hmm2.Emission()[j].Covariance()(i, k)).epsilon(1e-5));
      }
    }
  }
}

/**
 * Test saving and loading of Discrete HMMs
 */
TEST_CASE("DiscreteHMMLoadSaveTest", "[HMMTest]")
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
    cereal::XMLOutputArchive ar(ofs);
    ar(cereal::make_nvp("hmm", hmm));
  }

  // Load the HMM.
  HMM<DiscreteDistribution> hmm2(3, DiscreteDistribution(3));
  {
    std::ifstream ifs("test-hmm-save.xml");
    cereal::XMLInputArchive ar(ifs);
    ar(cereal::make_nvp("hmm", hmm2));
  }

  // Remove clutter.
  remove("test-hmm-save.xml");

  for (size_t j = 0; j < hmm.Emission().size(); ++j)
    for (size_t i = 0; i < hmm.Emission()[j].Probabilities().n_elem; ++i)
      REQUIRE(hmm.Emission()[j].Probabilities()[i] ==
          Approx(hmm2.Emission()[j].Probabilities()[i]).epsilon(1e-5));
}

/**
 * Test that HMM::Train() returns finite log-likelihood.
 */
TEST_CASE("HMMTrainReturnLogLikelihood", "[HMMTest]")
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

  REQUIRE(std::isfinite(loglik) == true);
}

/********************************************/
/** DiagonalGMM Hidden Markov Models Tests **/
/********************************************/

//! Make sure the prediction of DiagonalGMM HMMs is reasonable.
TEST_CASE("DiagonalGMMHMMPredictTest", "[HMMTest]")
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

    for (size_t i = 1; i < 1000; ++i)
    {
      double randValue = Random();

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
    for (size_t i = 0; i < 1000; ++i)
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

  REQUIRE(success == true);
}

/**
 * Make sure a random data sequence generation is correct when the emission
 * distribution is DiagonalGMM.
 */
TEST_CASE("DiagonalGMMHMMGenerateTest", "[HMMTest]")
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
  REQUIRE(arma::norm(hmm.Transition() - hmm2.Transition()) < 0.05);

  // Check that each Gaussian is the same.
  for (size_t dist = 0; dist < 3; dist++)
  {
    REQUIRE(arma::norm(hmm.Emission()[dist].Mean() -
        hmm2.Emission()[dist].Mean()) < 0.1);
    REQUIRE(arma::norm(hmm.Emission()[dist].Covariance() -
        hmm2.Emission()[dist].Covariance()) < 0.2);
  }
}

/**
 * Make sure the unlabeled 1-state training works reasonably given a single
 * distribution with diagonal covariance.
 */
TEST_CASE("DiagonalGMMHMMOneGaussianOneStateTrainingTest", "[HMMTest]")
{
  // Create a Gaussian distribution with diagonal covariance.
  DiagonalGaussianDistribution d("2.05 3.45", "0.89 1.05");

  // Make a sequence of observations.
  std::vector<arma::mat> observations(1, arma::mat(2, 5000));
  for (size_t obs = 0; obs < 1; obs++)
  {
    observations[obs].col(0) = d.Random();

    for (size_t i = 1; i < 5000; ++i)
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
      ColumnCovariance(observations[0],
      1 /* biased estimator */));

  // Check the model to see that it is correct.
  CheckMatrices(hmm.Emission()[0].Component(0).Mean(), actualMean);
  CheckMatrices(hmm.Emission()[0].Component(0).Covariance(), actualCovar);
}

/**
 * Make sure the unlabeled training works reasonably given a single
 * distribution with diagonal covariance.
 */
TEST_CASE("DiagonalGMMHMMOneGaussianUnlabeledTrainingTest", "[HMMTest]")
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

    for (size_t i = 1; i < 500; ++i)
    {
      double randValue = Random();

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
  REQUIRE(hmm.Initial()[0] == Approx(1.0).epsilon(0.0001));
  REQUIRE(hmm.Initial()[1] == Approx(0.0).margin(0.01));

  // Check the transition probability matrix.
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 2; ++j)
      REQUIRE(hmm.Transition()(i, j) - transProbs(i, j) ==
          Approx(0.0).margin(0.08));

  // Check the estimated weights of the each emission distribution.
  for (size_t i = 0; i < 2; ++i)
    REQUIRE(hmm.Emission()[i].Weights()[0] - gmms[i].Weights()[0] ==
        Approx(0.0).margin(0.08));

  // Check the estimated means of the each emission distribution.
  for (size_t i = 0; i < 2; ++i)
    REQUIRE(arma::norm(hmm.Emission()[i].Component(0).Mean() -
        gmms[i].Component(0).Mean()) < 0.2);

  // Check the estimated covariances of the each emission distribution.
  for (size_t i = 0; i < 2; ++i)
    REQUIRE(arma::norm(hmm.Emission()[i].Component(0).Covariance() -
        gmms[i].Component(0).Covariance()) < 0.5);
}

/**
 * Make sure the labeled training works reasonably given a single distribution
 * with diagonal covariance.
 */
TEST_CASE("DiagonalGMMHMMOneGaussianLabeledTrainingTest", "[HMMTest]")
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

    for (size_t i = 1; i < 5000; ++i)
    {
      double randValue = Random();
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
  REQUIRE(hmm.Initial()[0] == Approx(1.0).epsilon(0.0001));
  REQUIRE(hmm.Initial()[1] == Approx(0.0).margin(0.01));
  REQUIRE(hmm.Initial()[2] == Approx(0.0).margin(0.01));

  // Check the transition probability matrix.
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      REQUIRE(hmm.Transition()(i, j) - transProbs(i, j) ==
          Approx(0.0).margin(0.03));

  // Check the estimated weights of the each emission distribution.
  for (size_t i = 0; i < 3; ++i)
    REQUIRE(hmm.Emission()[i].Weights()[0] - gmms[i].Weights()[0] ==
        Approx(0.0).margin(0.08));

  // Check the estimated means of the each emission distribution.
  for (size_t i = 0; i < 3; ++i)
    REQUIRE(arma::norm(hmm.Emission()[i].Component(0).Mean() -
        gmms[i].Component(0).Mean()) < 0.2);

  // Check the estimated covariances of the each emission distribution.
  for (size_t i = 0; i < 3; ++i)
    REQUIRE(arma::norm(hmm.Emission()[i].Component(0).Covariance() -
        gmms[i].Component(0).Covariance()) < 0.5);
}

/**
 * Make sure the unlabeled training works reasonably given multiple
 * distributions with diagonal covariance.
 */
TEST_CASE("DiagonalGMMHMMMultipleGaussiansUnlabeledTrainingTest", "[HMMTest]")
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

    for (size_t i = 1; i < 1000; ++i)
    {
      double randValue = Random();

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
  REQUIRE(hmm.Initial()[0] == Approx(1.0).epsilon(0.0001));
  REQUIRE(hmm.Initial()[1] == Approx(0.0).margin(0.01));

  // Check the transition probability matrix.
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 2; ++j)
      REQUIRE(hmm.Transition()(i, j) - transProbs(i, j) ==
          Approx(0.0).margin(0.08));

  // Sort by the estimated weights of the first emission distribution.
  arma::uvec sortedIndices = sort_index(hmm.Emission()[0].Weights());

  // Check the first emission distribution.
  for (size_t i = 0; i < 2; ++i)
  {
    // Check the estimated weights using the first DiagonalGMM.
    REQUIRE(hmm.Emission()[0].Weights()[sortedIndices[i]] -
        gmms[0].Weights()[i] == Approx(0.0).margin(0.08));

    // Check the estimated means using the first DiagonalGMM.
    REQUIRE(arma::norm(
      hmm.Emission()[0].Component(sortedIndices[i]).Mean() -
      gmms[0].Component(i).Mean()) < 0.35);

    // Check the estimated covariances using the first DiagonalGMM.
    REQUIRE(arma::norm(
      hmm.Emission()[0].Component(sortedIndices[i]).Covariance() -
      gmms[0].Component(i).Covariance()) < 0.6);
  }

  // Sort by the estimated weights of the second emission distribution.
  sortedIndices = sort_index(hmm.Emission()[1].Weights());

  // Check the second emission distribution.
  for (size_t i = 0; i < 2; ++i)
  {
    // Check the estimated weights using the second DiagonalGMM.
    REQUIRE(hmm.Emission()[1].Weights()[sortedIndices[i]] -
        gmms[1].Weights()[i] == Approx(0.0).margin(0.08));

    // Check the estimated means using the second DiagonalGMM.
    REQUIRE(arma::norm(
      hmm.Emission()[1].Component(sortedIndices[i]).Mean() -
      gmms[1].Component(i).Mean()) < 0.35);

    // Check the estimated covariances using the second DiagonalGMM.
    REQUIRE(arma::norm(
      hmm.Emission()[1].Component(sortedIndices[i]).Covariance() -
      gmms[1].Component(i).Covariance()) < 0.6);
  }
}

/**
 * Make sure the labeled training works reasonably given multiple distributions
 * with diagonal covariance.
 */
TEST_CASE("DiagonalGMMHMMMultipleGaussiansLabeledTrainingTest", "[HMMTest]")
{
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

    for (size_t i = 1; i < 2500; ++i)
    {
      double randValue = Random();

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
  REQUIRE(hmm.Initial()[0] == Approx(1.0).epsilon(0.0001));
  REQUIRE(hmm.Initial()[1] == Approx(0.0).margin(0.01));

  // Check the transition probability matrix.
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 2; ++j)
      REQUIRE(hmm.Transition()(i, j) - transProbs(i, j) ==
          Approx(0.0).margin(0.03));

  // Sort by the estimated weights of the first emission distribution.
  arma::uvec sortedIndices = sort_index(hmm.Emission()[0].Weights());

  // Check the first emission distribution.
  for (size_t i = 0; i < 2; ++i)
  {
    // Check the estimated weights using the first DiagonalGMM.
    REQUIRE(hmm.Emission()[0].Weights()[sortedIndices[i]] -
        gmms[0].Weights()[i] == Approx(0.0).margin(0.08));

    // Check the estimated means using the first DiagonalGMM.
    REQUIRE(arma::norm(
      hmm.Emission()[0].Component(sortedIndices[i]).Mean() -
      gmms[0].Component(i).Mean()) < 0.2);

    // Check the estimated covariances using the first DiagonalGMM.
    REQUIRE(arma::norm(
      hmm.Emission()[0].Component(sortedIndices[i]).Covariance() -
      gmms[0].Component(i).Covariance()) < 0.5);
  }

  // Sort by the estimated weights of the second emission distribution.
  sortedIndices = sort_index(hmm.Emission()[1].Weights());

  // Check the second emission distribution.
  for (size_t i = 0; i < 2; ++i)
  {
    // Check the estimated weights using the second DiagonalGMM.
    REQUIRE(hmm.Emission()[1].Weights()[sortedIndices[i]] -
        gmms[1].Weights()[i] == Approx(0.0).margin(0.08));

    // Check the estimated means using the second DiagonalGMM.
    REQUIRE(arma::norm(
      hmm.Emission()[1].Component(sortedIndices[i]).Mean() -
      gmms[1].Component(i).Mean()) < 0.2);

    // Check the estimated covariances using the second DiagonalGMM.
    REQUIRE(arma::norm(
      hmm.Emission()[1].Component(sortedIndices[i]).Covariance() -
      gmms[1].Component(i).Covariance()) < 0.5);
  }
}

/**
 * Make sure loading and saving the model is correct.
 */
TEST_CASE("DiagonalGMMHMMLoadSaveTest", "[HMMTest]")
{
  // Create a GMM HMM, save and load it.
  HMM<DiagonalGMM> hmm(3, DiagonalGMM(4, 3));

  // Generate intial random values.
  for (size_t j = 0; j < hmm.Emission().size(); ++j)
  {
    hmm.Emission()[j].Weights().randu();
    for (size_t i = 0; i < hmm.Emission()[j].Gaussians(); ++i)
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
    cereal::XMLOutputArchive ar(ofs);
    ar(cereal::make_nvp("hmm", hmm));
  }

  // Load the HMM.
  HMM<DiagonalGMM> hmm2(3, DiagonalGMM(4, 3));
  {
    std::ifstream ifs("test-hmm-save.xml");
    cereal::XMLInputArchive ar(ifs);
    ar(cereal::make_nvp("hmm", hmm2));
  }

  // Remove clutter.
  remove("test-hmm-save.xml");

  for (size_t j = 0; j < hmm.Emission().size(); ++j)
  {
    // Check the number of Gaussians.
    REQUIRE(hmm.Emission()[j].Gaussians() == hmm2.Emission()[j].Gaussians());

    // Check the dimensionality.
    REQUIRE(hmm.Emission()[j].Dimensionality() ==
            hmm2.Emission()[j].Dimensionality());

    for (size_t i = 0; i < hmm.Emission()[j].Dimensionality(); ++i)
      // Check the weights.
      REQUIRE(hmm.Emission()[j].Weights()[i] ==
          Approx(hmm2.Emission()[j].Weights()[i]).epsilon(1e-5));

    for (size_t i = 0; i < hmm.Emission()[j].Gaussians(); ++i)
    {
      for (size_t l = 0; l < hmm.Emission()[j].Dimensionality(); l++)
      {
        // Check the means.
      REQUIRE(hmm.Emission()[j].Component(i).Mean()[l] ==
          Approx(hmm2.Emission()[j].Component(i).Mean()[l]).epsilon(1e-5));

        // Check the covariances.
      REQUIRE(hmm.Emission()[j].Component(i).Covariance()[l] ==
          Approx(hmm2.Emission()[j].Component(i).Covariance()[l]).epsilon(
          1e-5));
      }
    }
  }
}
