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
  HMMModel * h = new HMMModel(DiscreteHMM);
  // Load data
  arma::mat inp;
  data::Load("obs1.csv", inp);
  std::vector<arma::mat> trainSeq = {inp};
  // Init
  h->PerformAction<Init, std::vector<arma::mat>>(&trainSeq);
  // Train
  h->PerformAction<Train, std::vector<arma::mat>>(&trainSeq);

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
  HMMModel * h = new HMMModel(GaussianHMM);
  // Load data
  arma::mat inp;
  data::Load("obs1.csv", inp);
  std::vector<arma::mat> trainSeq = {inp};
  // Init
  h->PerformAction<Init, std::vector<arma::mat>>(&trainSeq);
  // Train
  h->PerformAction<Train, std::vector<arma::mat>>(&trainSeq);

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
  HMMModel * h = new HMMModel(GaussianMixtureModelHMM);
  // Load data
  arma::mat inp;
  data::Load("obs1.csv", inp);
  std::vector<arma::mat> trainSeq = {inp};
  // Init
  h->PerformAction<Init, std::vector<arma::mat>>(&trainSeq);
  // Train
  h->PerformAction<Train, std::vector<arma::mat>>(&trainSeq);

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

BOOST_AUTO_TEST_CASE(HMMGenerateLengthPositiveTest)
{
  // Train an HMM
  HMMModel * h = new HMMModel(DiscreteHMM);
  // Load data
  arma::mat inp;
  data::Load("obs1.csv", inp);
  std::vector<arma::mat> trainSeq = {inp};
  // Init
  h->PerformAction<Init, std::vector<arma::mat>>(&trainSeq);
  // Train
  h->PerformAction<Train, std::vector<arma::mat>>(&trainSeq);

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
  HMMModel * h = new HMMModel(DiscreteHMM);
  // Load data
  arma::mat inp;
  data::Load("obs1.csv", inp);
  std::vector<arma::mat> trainSeq = {inp};
  // Init
  h->PerformAction<Init, std::vector<arma::mat>>(&trainSeq);
  // Train
  h->PerformAction<Train, std::vector<arma::mat>>(&trainSeq);

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
