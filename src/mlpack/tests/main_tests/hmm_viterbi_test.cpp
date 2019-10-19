/**
 * @file hmm_viterbi_test.cpp
 * @author Daivik Nema
 *
 * Test mlpackMain() of hmm_viterbi_main.cpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "HMMViterbi";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/hmm/hmm_model.hpp>
#include <mlpack/methods/hmm/hmm.hpp>
#include <mlpack/methods/hmm/hmm_viterbi_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

#include "hmm_test_utils.hpp"

using namespace mlpack;

struct HMMViterbiTestFixture
{
 public:
  HMMViterbiTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~HMMViterbiTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(HMMViterbiMainTest, HMMViterbiTestFixture);

BOOST_AUTO_TEST_CASE(HMMViterbiDiscreteHMMCheckDimensionsTest)
{
  // Load data to train a discrete HMM model with.
  arma::mat inp;
  data::Load("obs1.csv", inp);
  std::vector<arma::mat> trainSeq = {inp};

  // Initialize and train a discrete HMM model.
  HMMModel* h = new HMMModel(DiscreteHMM);
  h->PerformAction<InitHMMModel, std::vector<arma::mat>>(&trainSeq);
  h->PerformAction<TrainHMMModel, std::vector<arma::mat>>(&trainSeq);

  // Now that we have a trained HMM model, we can use it to predict the state
  // sequence for a given observation sequence - using the Viterbi algorithm.
  // Load the input model to be used for inference and the sequence over which
  // inference is to be performed.
  SetInputParam("input_model", h);
  SetInputParam("input", inp);

  // Call to hmm_viterbi_main.
  mlpackMain();

  // Get the output of viterbi inference.
  arma::Mat<size_t> out = CLI::GetParam<arma::Mat<size_t> >("output");

  // Output sequence length must be the same as input sequence length and
  // there should only be one row (since states are single dimensional values).
  BOOST_REQUIRE_EQUAL(out.n_rows, 1);
  BOOST_REQUIRE_EQUAL(out.n_cols, inp.n_cols);
}

BOOST_AUTO_TEST_CASE(HMMViterbiGaussianHMMCheckDimensionsTest)
{
  // Load data to train a gaussian HMM model with.
  arma::mat inp;
  data::Load("obs1.csv", inp);
  std::vector<arma::mat> trainSeq = {inp};

  // Initialize and train a gaussian HMM model.
  HMMModel* h = new HMMModel(GaussianHMM);
  h->PerformAction<InitHMMModel, std::vector<arma::mat>>(&trainSeq);
  h->PerformAction<TrainHMMModel, std::vector<arma::mat>>(&trainSeq);

  // Now that we have a trained HMM model, we can use it to predict the state
  // sequence for a given observation sequence - using the Viterbi algorithm.
  // Load the input model to be used for inference and the sequence over which
  // inference is to be performed.
  SetInputParam("input_model", h);
  SetInputParam("input", inp);

  // Call to hmm_viterbi_main.
  mlpackMain();

  // Get the output of viterbi inference.
  arma::Mat<size_t> out = CLI::GetParam<arma::Mat<size_t> >("output");

  // Output sequence length must be the same as input sequence length and
  // there should only be one row (since states are single dimensional values).
  BOOST_REQUIRE_EQUAL(out.n_rows, 1);
  BOOST_REQUIRE_EQUAL(out.n_cols, inp.n_cols);
}

BOOST_AUTO_TEST_CASE(HMMViterbiGMMHMMCheckDimensionsTest)
{
  std::vector<GMM> gmms(2, GMM(2, 2));
  gmms[0].Weights() = arma::vec("0.3 0.7");

  gmms[0].Component(0) = GaussianDistribution("4.25 3.10",
      "1.00 0.20; 0.20 0.89");
  gmms[0].Component(1) = GaussianDistribution("7.10 5.01",
      "1.00 0.00; 0.00 1.01");
  gmms[1].Weights() = arma::vec("0.20 0.80");

  gmms[1].Component(0) = GaussianDistribution("-3.00 -6.12",
      "1.00 0.00; 0.00 1.00");
  gmms[1].Component(1) = GaussianDistribution("-4.25 -2.12",
      "1.50 0.60; 0.60 1.20");

  // Transition matrix.
  arma::mat transMat("0.40 0.60; 0.60 0.40");

  // Make some observations.
  arma::mat observations(2, 50);
  arma::Row<size_t> states(50);

  states[0] = 0;
  observations.col(0) = gmms[0].Random();

  for (size_t i = 1; i < 50; ++i)
  {
    double randValue = (double) rand() / (double) RAND_MAX;

    if (randValue <= transMat(0, states[i - 1]))
      states[i] = 0;
    else
      states[i] = 1;

    observations.col(i) = gmms[states[i]].Random();
  }

  // Initialize and train a GMM HMM model.
  HMMModel* h = new HMMModel(GaussianMixtureModelHMM);
  *(h->GMMHMM()) = HMM<GMM>(2, GMM(2, 2));

  // Manually set the components.
  h->GMMHMM()->Transition() = transMat;
  h->GMMHMM()->Emission() = gmms;

  // Now that we have a trained HMM model, we can use it to predict the state
  // sequence for a given observation sequence - using the Viterbi algorithm.
  // Load the input model to be used for inference and the sequence over which
  // inference is to be performed.
  SetInputParam("input_model", h);
  SetInputParam("input", observations);

  // Call to hmm_viterbi_main.
  mlpackMain();

  // Get the output of viterbi inference.
  arma::Mat<size_t> out = CLI::GetParam<arma::Mat<size_t> >("output");

  // Output sequence length must be the same as input sequence length and
  // there should only be one row (since states are single dimensional values).
  BOOST_REQUIRE_EQUAL(out.n_rows, 1);
  BOOST_REQUIRE_EQUAL(out.n_cols, observations.n_cols);
}

BOOST_AUTO_TEST_CASE(HMMViterbiDiagonalGMMHMMCheckDimensionsTest)
{
  std::vector<DiagonalGMM> gmms(2, DiagonalGMM(2, 2));
  gmms[0].Weights() = arma::vec("0.2 0.8");

  gmms[0].Component(0) = DiagonalGaussianDistribution("2.75 1.60",
      "0.50 0.50");
  gmms[0].Component(1) = DiagonalGaussianDistribution("6.15 2.51",
      "1.00 1.50");
  gmms[1].Weights() = arma::vec("0.4 0.6");

  gmms[1].Component(0) = DiagonalGaussianDistribution("-1.00 -3.42",
      "0.20 1.00");
  gmms[1].Component(1) = DiagonalGaussianDistribution("-3.10 -5.05",
      "1.20 0.80");

  // Transition matrix.
  arma::mat transMat("0.30 0.70; 0.70 0.30");

  // Make some observations.
  arma::mat observations(2, 50);
  arma::Row<size_t> states(50);

  states[0] = 0;
  observations.col(0) = gmms[0].Random();

  for (size_t i = 1; i < 50; ++i)
  {
    double randValue = mlpack::math::Random();

    if (randValue <= transMat(0, states[i - 1]))
      states[i] = 0;
    else
      states[i] = 1;

    observations.col(i) = gmms[states[i]].Random();
  }

  // Initialize and train a diagonal GMM HMM model.
  HMMModel* h = new HMMModel(DiagonalGaussianMixtureModelHMM);
  *(h->DiagGMMHMM()) = HMM<DiagonalGMM>(2, DiagonalGMM(2, 2));

  // Manually set the components.
  h->DiagGMMHMM()->Transition() = transMat;
  h->DiagGMMHMM()->Emission() = gmms;

  // Now that we have a trained HMM model, we can use it to predict the state
  // sequence for a given observation sequence - using the Viterbi algorithm.
  // Load the input model to be used for inference and the sequence over which
  // inference is to be performed.
  SetInputParam("input_model", h);
  SetInputParam("input", observations);

  // Call to hmm_viterbi_main.
  mlpackMain();

  // Get the output of viterbi inference.
  arma::Mat<size_t> out = CLI::GetParam<arma::Mat<size_t> >("output");

  // Output sequence length must be the same as input sequence length and
  // there should only be one row (since states are single dimensional values).
  BOOST_REQUIRE_EQUAL(out.n_rows, 1);
  BOOST_REQUIRE_EQUAL(out.n_cols, observations.n_cols);
}

BOOST_AUTO_TEST_SUITE_END();
