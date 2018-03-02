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

struct Train
{
  template<typename HMMType>
  static void Apply(HMMType& hmm, vector<arma::mat>* trainSeq)
  {
    // For now, perform unsupervised (Baum-Welch) training
    hmm.Train(*trainSeq);
  }
};

BOOST_AUTO_TEST_CASE(HMMViterbiCheckDimenstionsTest)
{
  // Train an HMM
  HMMModel h;
  arma::mat inp;
  data::Load("obs-1.csv", inp);
  std::cout << "Loaded data:" << std::endl << inp << std::endl;
  std::vector<arma::mat> trainSeq = {inp};

  // Train the HMM
  h.PerformAction<Train, std::vector<arma::mat> >(&trainSeq);
  std::cout << __func__ << ": Training complete!" << std::endl;

  // Set the params for the hmm_viterbi invocation
  SetInputParam("input_model", &h);
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
