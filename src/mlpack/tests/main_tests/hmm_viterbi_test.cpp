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
                     double tolerance=1e-05)
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
                     double tolerance=1e-05)
  {
    // Not implemented
  }

  static void Create(HMM<GMM>& hmm,
                     vector<mat>& trainSeq,
                     size_t states,
                     double tolerance=1e-05)
  {
    // Not implemented
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
  }

  static void RandomInitialize(vector<GMM>& e)
  {
    // Not implemented
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

BOOST_AUTO_TEST_CASE(HMMViterbiCheckDimensionsTest)
{
  // Train an HMM
  HMMModel * h = new HMMModel(DiscreteHMM);
  // Load data
  arma::mat inp;
  data::Load("obs1.csv", inp);
  std::cout << "Loaded data:" << std::endl << inp << std::endl;
  std::vector<arma::mat> trainSeq = {inp};
  // Init
  h->PerformAction<Init, std::vector<arma::mat>>(&trainSeq);
  // Train
  h->PerformAction<Train, std::vector<arma::mat>>(&trainSeq);

  std::cout << __func__ << ": Training complete!" << std::endl;

  // Set the params for the hmm_viterbi invocation
  SetInputParam("input_model", h);
  SetInputParam("input", inp);
  std::cout << __func__ << ": Set input params!" << std::endl;

  mlpackMain();
  std::cout << __func__ << ": HMMViterbiMain() complete!" << std::endl;

  arma::Mat<size_t> out = CLI::GetParam<arma::Mat<size_t> >("output");
  BOOST_REQUIRE(out.n_rows == inp.n_rows);
  BOOST_REQUIRE(out.n_cols == inp.n_cols);
  BOOST_REQUIRE(out.n_elem == inp.n_elem);
}

BOOST_AUTO_TEST_SUITE_END();
