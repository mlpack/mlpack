/**
 * @file hmm_viterbi_test.cpp
 * @author Daivik Nema
 *
 * Test mlpackMain() of hmm_viterbi_main.cpp
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

  // Set the params for the hmm_viterbi invocation
  SetInputParam("input_model", h);
  SetInputParam("input", inp);

  mlpackMain();

  arma::Mat<size_t> out = CLI::GetParam<arma::Mat<size_t> >("output");

  BOOST_REQUIRE_EQUAL(out.n_rows, 1);
  BOOST_REQUIRE_EQUAL(out.n_cols, inp.n_cols);
}

BOOST_AUTO_TEST_CASE(HMMViterbiGaussianHMMCheckDimensionsTest)
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

  // Set the params for the hmm_viterbi invocation
  SetInputParam("input_model", h);
  SetInputParam("input", inp);

  mlpackMain();

  arma::Mat<size_t> out = CLI::GetParam<arma::Mat<size_t> >("output");

  BOOST_REQUIRE_EQUAL(out.n_rows, 1);
  BOOST_REQUIRE_EQUAL(out.n_cols, inp.n_cols);
}

BOOST_AUTO_TEST_CASE(HMMViterbiGMMHMMCheckDimensionsTest)
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

  // Set the params for the hmm_viterbi invocation
  SetInputParam("input_model", h);
  SetInputParam("input", inp);

  mlpackMain();

  arma::Mat<size_t> out = CLI::GetParam<arma::Mat<size_t> >("output");

  BOOST_REQUIRE_EQUAL(out.n_rows, 1);
  BOOST_REQUIRE_EQUAL(out.n_cols, inp.n_cols);
}

BOOST_AUTO_TEST_SUITE_END();
