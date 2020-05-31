#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "K-RankApproximateNearestNeighborsSearch;

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/neighbor_search/krann_main.hpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct  KRANNTestFixture
{
  KRANNTestFixture()
  {
    CLI::RestoreSettings(testName);
  }

  ~KRANNTestFixture()
  {
    bindings::test::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(KRANNMainTest, KRANNTestFixture);

//not both ref data and model
BOOST_AUTO_TEST_CASE(KRANNRefModelTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100);

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);

  mlpackMain();

  SetInputParam("input_model",
      std::move(CLI::GetParam<KRANNModel*>("output_model")));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

//alpha in range
BOOST_AUTO_TEST_CASE(KRANNInvalidAlphaTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100);

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);

  mlpackMain();

  SetInputParam("alpha", (double) 1.2);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

//tau in range
BOOST_AUTO_TEST_CASE(KRANNInvalidTauTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100);
  
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);

  mlpackMain();

  SetInputParam("tau", (double) -1);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

//query dimension check
BOOST_AUTO_TEST_CASE(KRANNQueriesDimCheck)
{
  arma::mat referenceData;
  referenceData.randu(3, 100);

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  
  arma::mat queryData;
  queryData.randu(2, 10);

  SetInputParam("query", std::move(queryData));


  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

//k validity
BOOST_AUTO_TEST_CASE(KRANNValidKTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100);

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 101);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["k"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) -1);

  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}


