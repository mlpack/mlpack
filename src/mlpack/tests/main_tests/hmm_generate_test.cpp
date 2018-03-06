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

BOOST_AUTO_TEST_CASE(HMMGenerateCheckDimensionsTest)
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
