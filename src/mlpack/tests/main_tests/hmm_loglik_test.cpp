/**
 * @file hmm_loglik_test.cpp
 * @author Daivik Nema
 *
 * Test mlpackMain() of hmm_loglik_main.cpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "HMMLoglik";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/hmm/hmm_model.hpp>
#include <mlpack/methods/hmm/hmm.hpp>
#include <mlpack/methods/hmm/hmm_loglik_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

#include "hmm_test_utils.hpp"

using namespace mlpack;

struct HMMLoglikTestFixture
{
 public:
  HMMLoglikTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~HMMLoglikTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(HMMLoglikMainTest, HMMLoglikTestFixture);

BOOST_AUTO_TEST_CASE(HMMLoglikOutputNegativeTest)
{
  // Load data to train a discrete HMM model with.
  arma::mat inp;
  data::Load("obs1.csv", inp);
  std::vector<arma::mat> trainSeq = {inp};

  // Initialize and train an HMM model.
  HMMModel* h = new HMMModel(DiscreteHMM);
  h->PerformAction<InitHMMModel, std::vector<arma::mat>>(&trainSeq);
  h->PerformAction<TrainHMMModel, std::vector<arma::mat>>(&trainSeq);


  // Set the params for the hmm_loglik invocation
  SetInputParam("input_model", h);
  SetInputParam("input", inp);

  mlpackMain();

  double loglik = CLI::GetParam<double>("log_likelihood");

  // Since the log of a probability <= 0 ...
  BOOST_REQUIRE(loglik <= 0);
}

BOOST_AUTO_TEST_SUITE_END();
