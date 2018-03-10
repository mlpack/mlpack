/**
 * @file hmm_generate_test.cpp
 * @author Daivik Nema
 *
 * Test mlpackMain() of hmm_generate_main.cpp
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "HMMGenerate";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/hmm/hmm_model.hpp>
#include <mlpack/methods/hmm/hmm.hpp>
#include <mlpack/methods/hmm/hmm_generate_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

#include "hmm_test_utils.hpp"

using namespace mlpack;

struct HMMGenerateTestFixture
{
 public:
  HMMGenerateTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~HMMGenerateTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(HMMGenerateMainTest, HMMGenerateTestFixture);

BOOST_AUTO_TEST_CASE(HMMGenerateDiscreteHMMCheckDimensionsTest)
{
  // Train an HMM
  HMMModel* h = new HMMModel(DiscreteHMM);
  // Load data
  arma::mat inp;
  data::Load("obs1.csv", inp);
  std::vector<arma::mat> trainSeq = {inp};
  // Init
  h->PerformAction<InitHMMModel, std::vector<arma::mat>>(&trainSeq);
  // Train
  h->PerformAction<TrainHMMModel, std::vector<arma::mat>>(&trainSeq);

  // Set the params for the hmm_generate invocation
  int length = 3;
  SetInputParam("model", h);
  SetInputParam("length", length);

  mlpackMain();

  arma::mat obsSeq = CLI::GetParam<arma::mat>("output");
  BOOST_REQUIRE_EQUAL(obsSeq.n_cols, (size_t)length);
  BOOST_REQUIRE_EQUAL(obsSeq.n_rows, (size_t)1);
  BOOST_REQUIRE_EQUAL(obsSeq.n_elem, (size_t)length);

  arma::Mat<size_t> stateSeq = CLI::GetParam<arma::Mat<size_t>>("state");
  BOOST_REQUIRE_EQUAL(stateSeq.n_cols, (size_t)length);
  BOOST_REQUIRE_EQUAL(stateSeq.n_rows, (size_t)1);
  BOOST_REQUIRE_EQUAL(stateSeq.n_elem, (size_t)length);
}

BOOST_AUTO_TEST_CASE(HMMGenerateGaussianHMMCheckDimensionsTest)
{
  // Train an HMM
  HMMModel* h = new HMMModel(GaussianHMM);
  // Load data
  arma::mat inp;
  data::Load("obs1.csv", inp);
  std::vector<arma::mat> trainSeq = {inp};
  // Init
  h->PerformAction<InitHMMModel, std::vector<arma::mat>>(&trainSeq);
  // Train
  h->PerformAction<TrainHMMModel, std::vector<arma::mat>>(&trainSeq);

  // Set the params for the hmm_generate invocation
  int length = 3;
  SetInputParam("model", h);
  SetInputParam("length", length);

  mlpackMain();

  arma::mat obsSeq = CLI::GetParam<arma::mat>("output");
  BOOST_REQUIRE_EQUAL(obsSeq.n_cols, (size_t)length);
  BOOST_REQUIRE_EQUAL(obsSeq.n_rows, (size_t)1);
  BOOST_REQUIRE_EQUAL(obsSeq.n_elem, (size_t)length);

  arma::Mat<size_t> stateSeq = CLI::GetParam<arma::Mat<size_t>>("state");
  BOOST_REQUIRE_EQUAL(stateSeq.n_cols, (size_t)length);
  BOOST_REQUIRE_EQUAL(stateSeq.n_rows, (size_t)1);
  BOOST_REQUIRE_EQUAL(stateSeq.n_elem, (size_t)length);
}

BOOST_AUTO_TEST_CASE(HMMGenerateGMMHMMCheckDimensionsTest)
{
  // Train an HMM
  HMMModel* h = new HMMModel(GaussianMixtureModelHMM);
  // Load data
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
  std::vector<arma::mat> observations(5, arma::mat(2, 50));
  std::vector<arma::Row<size_t> > states(5, arma::Row<size_t>(50));
  for (size_t obs = 0; obs < 5; obs++)
  {
    states[obs][0] = 0;
    observations[obs].col(0) = gmms[0].Random();

    for (size_t i = 1; i < 50; i++)
    {
      double randValue = (double) rand() / (double) RAND_MAX;

      if (randValue <= transMat(0, states[obs][i - 1]))
        states[obs][i] = 0;
      else
        states[obs][i] = 1;

      observations[obs].col(i) = gmms[states[obs][i]].Random();
    }
  }
  // Init
  h->PerformAction<InitHMMModel, std::vector<arma::mat>>(&observations);
  // Train
  h->PerformAction<TrainHMMModel, std::vector<arma::mat>>(&observations);

  // Set the params for the hmm_generate invocation
  int length = 3;
  SetInputParam("model", h);
  SetInputParam("length", length);

  mlpackMain();

  arma::mat obsSeq = CLI::GetParam<arma::mat>("output");
  BOOST_REQUIRE_EQUAL(obsSeq.n_cols, (size_t)length);
  BOOST_REQUIRE_EQUAL(obsSeq.n_rows, (size_t)2);
  BOOST_REQUIRE_EQUAL(obsSeq.n_elem, (size_t)(length*2));

  arma::Mat<size_t> stateSeq = CLI::GetParam<arma::Mat<size_t>>("state");
  BOOST_REQUIRE_EQUAL(stateSeq.n_cols, (size_t)length);
  BOOST_REQUIRE_EQUAL(stateSeq.n_rows, (size_t)1);
  BOOST_REQUIRE_EQUAL(stateSeq.n_elem, (size_t)length);
}

BOOST_AUTO_TEST_CASE(HMMGenerateLengthPositiveTest)
{
  // Train an HMM
  HMMModel* h = new HMMModel(DiscreteHMM);
  // Load data
  arma::mat inp;
  data::Load("obs1.csv", inp);
  std::vector<arma::mat> trainSeq = {inp};
  // Init
  h->PerformAction<InitHMMModel, std::vector<arma::mat>>(&trainSeq);
  // Train
  h->PerformAction<TrainHMMModel, std::vector<arma::mat>>(&trainSeq);

  // Set the params for the hmm_generate invocation
  int length = -3; // Invalid
  SetInputParam("model", h);
  SetInputParam("length", length);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_CASE(HMMGenerateValidStartStateTest)
{
  // Train an HMM
  HMMModel* h = new HMMModel(DiscreteHMM);
  // Load data
  arma::mat inp;
  data::Load("obs1.csv", inp);
  std::vector<arma::mat> trainSeq = {inp};
  // Init
  h->PerformAction<InitHMMModel, std::vector<arma::mat>>(&trainSeq);
  // Train
  h->PerformAction<TrainHMMModel, std::vector<arma::mat>>(&trainSeq);

  int length = 3;
  int startState = 2; // Invalid
  SetInputParam("model", h);
  SetInputParam("length", length);
  SetInputParam("start_state", startState);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();
