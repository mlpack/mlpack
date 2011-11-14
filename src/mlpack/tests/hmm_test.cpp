/**
 * @file hmm_test.cpp
 *
 * Test file for HMMs.
 */
#include <mlpack/core.h>
#include <mlpack/methods/hmm/discreteHMM.hpp>
#include <mlpack/methods/hmm/hmm.hpp>
#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::hmm;

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
  arma::mat transition("0.7 0.3; 0.3 0.7");
  arma::mat emission("0.9 0.2; 0.1 0.8");

  HMM<int> hmm(transition, emission);

  // Now let's take a sequence and find what the most likely state is.
  // We'll use the sequence [U U N U U] (U = umbrella, N = no umbrella) like on
  // p. 547.
  arma::vec observation("0 0 1 0 0");

  arma::Col<size_t> states;
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
  // Two hidden states: H (high GC content) and L (low GC content), as well as a
  // start state.
  arma::mat transition("0.0 0.0 0.0;"
                       "0.5 0.5 0.4;"
                       "0.5 0.5 0.6");
  // Four emission states: A, C, G, T.  Start state doesn't emit...
  arma::mat emission("0.25 0.20 0.30;"
                     "0.25 0.30 0.20;"
                     "0.25 0.30 0.20;"
                     "0.25 0.20 0.30");

  HMM<int> hmm(transition, emission);

  // GGCACTGAA.
  arma::vec observation("2 2 1 0 1 3 2 0 0");

  arma::Col<size_t> states;
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
  arma::vec obs("3 3 2 1 1 1 1 3 3 1");
  arma::mat transm("0.1 0.9; 0.4 0.6");
  arma::mat emis("0.85 0; 0.15 0; 0 0.5; 0 0.5");

  HMM<int> hmm(2, 4);

  hmm.Transition() = transm;
  hmm.Emission() = emis;

  // Now check we are getting the same results as MATLAB for this sequence.
  arma::mat stateProb;
  arma::mat forwardProb;
  arma::mat backwardProb;
  arma::vec scales;

  double log = hmm.Estimate(obs, stateProb, forwardProb, backwardProb, scales);

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
  // Don't yet require a useful distribution.
  HMM<int> hmm(1, 1); // 1 state, 1 emission.

  std::vector<arma::vec> observations;
  observations.push_back("0 0 0 0 0 0 0 0 0");
  observations.push_back("0 0 0 0 0 0 0");
  observations.push_back("0 0 0 0 0 0 0 0 0 0 0 0");
  observations.push_back("0 0 0 0 0 0 0 0 0 0");

  hmm.Train(observations);

  BOOST_REQUIRE_CLOSE(hmm.Emission()(0, 0), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(hmm.Transition()(0, 0), 1.0, 1e-5);
}

/**
 * A slightly more complex model to estimate.
 */
BOOST_AUTO_TEST_CASE(SimpleBaumWelchDiscreteHMM)
{
  HMM<int> hmm(1, 2); // 1 state, 2 emissions.
  // Randomize the emission matrix.
  hmm.Emission().randu();
  hmm.Emission().col(0) /= accu(hmm.Emission().col(0));

  // P(each emission) = 0.5.
  // I've been careful to make P(first emission = 0) = P(first emission = 1).
  std::vector<arma::vec> observations;
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
  observations.push_back("0 1 0 1 0 1 0 1 0 1 0 1");
  observations.push_back("0 0 0 0 0 0 1 1 1 1 1 1");
  observations.push_back("1 1 1 1 1 1 0 0 0 0 0 0");
  observations.push_back("1 1 1 0 0 0 1 1 1 0 0 0");
  observations.push_back("0 0 1 1 0 0 0 0 1 1 1 1");
  observations.push_back("1 1 1 0 0 0 1 1 1 0 0 0");

  hmm.Train(observations);

  BOOST_REQUIRE_CLOSE(hmm.Emission()(0, 0), 0.5, 1e-5);
  BOOST_REQUIRE_CLOSE(hmm.Emission()(1, 0), 0.5, 1e-5);
  BOOST_REQUIRE_CLOSE(hmm.Transition()(0, 0), 1.0, 1e-5);
}

/**
 * Increasing complexity, but still simple; 4 emissions, 2 states; the state can
 * be determined directly by the emission.
 */
BOOST_AUTO_TEST_CASE(SimpleBaumWelchDiscreteHMM_2)
{
  HMM<int> hmm(2, 4); // 3 states (first state is start state), 5 emissions.

  // A little bit of obfuscation to the solution.
  hmm.Transition() = arma::mat("0.1 0.4; 0.9 0.6");
  hmm.Emission() = arma::mat("0.85 0.0; 0.15 0.0; 0.0 0.5; 0.0 0.5");

  // True emission matrix:
  //  [[1   0   0  ]
  //   [0   0.4 0  ]
  //   [0   0.6 0  ]
  //   [0   0   0.2]
  //   [0   0   0.8]]

  // True transmission matrix:
  //  [[0   0   0  ]
  //   [0.5 0.5 0.5]
  //   [0.5 0.5 0.5]]

  // Generate observations randomly by hand.  This is kinda ugly, but it works.
  std::vector<arma::vec> observations;
  size_t obsNum = 250; // Number of observations.
  size_t obsLen = 500; // Number of elements in each observation.
  for (size_t i = 0; i < obsNum; i++)
  {
    arma::vec observation(obsLen);

    size_t state = 0;
    size_t emission = 0;

    for (size_t obs = 0; obs < obsLen; obs++)
    {
      // See if state changed.
      double r = (double) rand() / (double) RAND_MAX;

      if (r <= 0.5)
        state = 0;
      else
        state = 1;

      // Now set the observation.
      r = (double) rand() / (double) RAND_MAX;

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

      observation[obs] = emission;
    }

    observations.push_back(observation);
  }

  arma::mat out(obsLen, obsNum);
  for (size_t i = 0; i < obsNum; i++)
    out.col(i) = observations[i];
  data::Save("out.csv", out);

  hmm.Train(observations);

  // Only require 2.5% tolerance, because this is a little fuzzier.
  BOOST_REQUIRE_CLOSE(hmm.Transition()(0, 0), 0.5, 2.5);
  BOOST_REQUIRE_CLOSE(hmm.Transition()(1, 0), 0.5, 2.5);
  BOOST_REQUIRE_CLOSE(hmm.Transition()(0, 1), 0.5, 2.5);
  BOOST_REQUIRE_CLOSE(hmm.Transition()(1, 1), 0.5, 2.5);

  BOOST_REQUIRE_CLOSE(hmm.Emission()(0, 0), 0.4, 2.5);
  BOOST_REQUIRE_CLOSE(hmm.Emission()(1, 0), 0.6, 2.5);
  BOOST_REQUIRE_SMALL(hmm.Emission()(2, 0), 2.5);
  BOOST_REQUIRE_SMALL(hmm.Emission()(3, 0), 2.5);
  BOOST_REQUIRE_SMALL(hmm.Emission()(0, 1), 2.5);
  BOOST_REQUIRE_SMALL(hmm.Emission()(1, 1), 2.5);
  BOOST_REQUIRE_CLOSE(hmm.Emission()(2, 1), 0.2, 2.5);
  BOOST_REQUIRE_CLOSE(hmm.Emission()(3, 1), 0.8, 2.5);
}

BOOST_AUTO_TEST_CASE(DiscreteHMMLabeledTrainTest)
{
  // Generate a random Markov model with 3 hidden states and 6 observations.
  arma::mat transition;
  arma::mat emission;

  transition.randu(3, 3);
  emission.randu(6, 3);

  // Normalize so they are correct transition and emission matrices.
  for (size_t col = 0; col < 3; col++)
  {
    transition.col(col) /= accu(transition.col(col));
    emission.col(col) /= accu(emission.col(col));
  }

  // Now generate sequences.
  size_t obsNum = 250;
  size_t obsLen = 800;

  std::vector<arma::vec> observations(obsNum);
  std::vector<arma::Col<size_t> > states(obsNum);

  for (size_t n = 0; n < obsNum; n++)
  {
    observations[n].set_size(obsLen);
    states[n].set_size(obsLen);

    // Random starting state.
    states[n][0] = rand() % 3;

    // Random starting observation.
    double obs = (double) rand() / (double) RAND_MAX;
    double sumProb = 0;
    for (size_t em = 0; em < 6; em++)
    {
      sumProb += emission(em, states[n][0]);
      if (sumProb >= obs)
      {
        observations[n][0] = em;
        break;
      }
    }

    // Now the rest of the observations.
    for (size_t t = 1; t < obsLen; t++)
    {
      // Choose random numbers for state transition and for emission transition.
      double obs = (double) rand() / (double) RAND_MAX;
      double state = (double) rand() / (double) RAND_MAX;

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
      sumProb = 0;
      for (size_t em = 0; em < 6; em++)
      {
        sumProb += emission(em, states[n][t]);
        if (sumProb >= obs)
        {
          observations[n][t] = em;
          break;
        }
      }
    }
  }

  // Now that our data is generated, we give the HMM the labeled data to train
  // on.
  HMM<int> hmm(3, 6);

  hmm.Train(observations, states);

  // We can't use % tolerance here because percent error increases as the actual
  // value gets very small.  So, instead, we just ensure that every value is no
  // more than 0.009 away from the actual value.
  for (size_t row = 0; row < hmm.Transition().n_rows; row++)
    for (size_t col = 0; col < hmm.Transition().n_cols; col++)
      BOOST_REQUIRE_SMALL(hmm.Transition()(row, col) - transition(row, col),
          0.009);

  for (size_t row = 0; row < hmm.Emission().n_rows; row++)
    for (size_t col = 0; col < hmm.Emission().n_cols; col++)
      BOOST_REQUIRE_SMALL(hmm.Emission()(row, col) - emission(row, col), 0.009);
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
  HMM<int> hmm(2, 4);

  // Now generate a really, really long sequence.
  arma::vec dataSeq;
  arma::Col<size_t> stateSeq;

  hmm.Generate(100000, dataSeq, stateSeq);

  // Now find the empirical probabilities of each state.
  arma::vec emissionProb(4);
  for (size_t em = 0; em < 4; em++)
    emissionProb[em] = accu(dataSeq == em);

  arma::vec stateProb(2);
  for (size_t st = 0; st < 2; st++)
    stateProb[st] = accu(stateSeq == st);

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

BOOST_AUTO_TEST_SUITE_END();
