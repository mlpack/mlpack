/**
 * @file hmm_training_tests.cpp
 * @author Daivik Nema
 *
 * Test mlpackMain() of hmm_train_main.cpp.
 */
#include <string>
#include <fstream>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "HMMTrain";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/hmm/hmm_train_main.cpp>
#include <mlpack/methods/hmm/hmm_model.hpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

using namespace mlpack;

struct HMMTrainMainTestFixture
{
 public:
  HMMTrainMainTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~HMMTrainMainTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(HMMTrainMainTest, HMMTrainMainTestFixture);

inline void fileExists(std::string fileName)
{
  ifstream ifp(fileName);
  if (!ifp.good())
    BOOST_FAIL("Bad stream " + fileName);
  ifp.close();
}

// Make sure that the number of states cannot be negative
BOOST_AUTO_TEST_CASE(HMMTrainStatesTest)
{
  std::string inputFileName = "hmm_train_obs.csv";
  int states = -3;  // Invalid!
  std::string hmmType = "discrete";

  fileExists(inputFileName);
  SetInputParam("input_file", std::move(inputFileName));
  SetInputParam("states", states);
  SetInputParam("type", std::move(hmmType));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

// Make sure that tolerance is non negative
BOOST_AUTO_TEST_CASE(HMMTrainToleranceNonNegative)
{
  std::string inputFileName = "hmm_train_obs.csv";
  int states = 3;
  std::string hmmType = "gaussian";
  double tol = - 100;  // Invalid

  fileExists(inputFileName);
  SetInputParam("input_file", std::move(inputFileName));
  SetInputParam("states", states);
  SetInputParam("type", std::move(hmmType));
  SetInputParam("tolerance", tol);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

// Make sure an error is thrown if type is something other than
// "discrete", "gaussian" or "gmm"
BOOST_AUTO_TEST_CASE(HMMTrainTypeTest)
{
  std::string inputFileName = "hmm_train_obs.csv";
  int states = 3;
  std::string hmmType = "some-not-supported-possibly-non-type";

  fileExists(inputFileName);
  SetInputParam("input_file", std::move(inputFileName));
  SetInputParam("states", states);
  SetInputParam("type", std::move(hmmType));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

// Make sure that the number of gaussians cannot be less than 0
BOOST_AUTO_TEST_CASE(HMMTrainGaussianTest)
{
  std::string inputFileName = "hmm_train_obs.csv";
  int states = 3;
  std::string hmmType = "gmm";
  int gaussians = -2;

  fileExists(inputFileName);
  SetInputParam("input_file", std::move(inputFileName));
  SetInputParam("states", states);
  SetInputParam("type", std::move(hmmType));
  SetInputParam("gaussians", gaussians);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

// Make sure that model reuse is possible and work properly
BOOST_AUTO_TEST_CASE(HMMTrainReuseModelTest)
{
  std::string inputObsFileName = "hmm_train_obs.csv";
  std::string inputLabFileName = "hmm_train_lab.csv";
  std::string hmmType = "discrete";
  int states = 3;

  fileExists(inputObsFileName);
  fileExists(inputLabFileName);
  // Make sure that the size of the 
  // training seq, and training labels is same
  arma::mat trainObs, trainLab;
  data::Load(inputObsFileName, trainObs);
  data::Load(inputLabFileName, trainLab);
  BOOST_REQUIRE_EQUAL(trainObs.n_rows, trainLab.n_rows);
  
  SetInputParam("input_file", std::move(inputObsFileName));
  SetInputParam("labels_file", std::move(inputLabFileName));
  SetInputParam("type", std::move(hmmType));
  SetInputParam("states", states);

  mlpackMain();

  HMMModel * ph1 = CLI::GetParam<HMMModel*>("output_model");
  HMMModel h1 = *(CLI::GetParam<HMMModel*>("output_model"));

  SetInputParam("input_model", std::move(ph1));

  CLI::GetSingleton().Parameters()["type"].wasPassed = false;
  CLI::GetSingleton().Parameters()["states"].wasPassed = false;

  mlpackMain();

  HMMModel h2 = *(CLI::GetParam<HMMModel*>("output_model"));

  BOOST_REQUIRE(h1.ApproximatelyEqual(h2, 1));
  BOOST_REQUIRE(h1.ApproximatelyEqual(h2, 1e-01));
  BOOST_REQUIRE(h1.ApproximatelyEqual(h2, 1e-02));
  BOOST_REQUIRE(h1.ApproximatelyEqual(h2, 1e-03));
  BOOST_REQUIRE(h1.ApproximatelyEqual(h2, 1e-04));
  BOOST_REQUIRE(h1.ApproximatelyEqual(h2, 1e-05));
}

// Test batch mode
BOOST_AUTO_TEST_CASE(HMMTrainBatchModeTest)
{
  std::string observationsFileName = "observations.txt";
  std::string labelsFileName = "labels.txt";
  std::string hmmType = "discrete";
  int states = 2;

  SetInputParam("input_file", std::move(observationsFileName));
  SetInputParam("labels_file", std::move(labelsFileName));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  SetInputParam("states", states);
  SetInputParam("type", std::move(hmmType));
  SetInputParam("batch", (bool) true);

  mlpackMain();

  // Now pass an observations file with extra non-existent filenames
  observationsFileName = "corrupt-observations-1.txt";
  SetInputParam("input_file", std::move(observationsFileName));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Now a mismatch between #observation files and #label files
  observationsFileName = "corrupt-observations-2.txt";
  SetInputParam("input_file", std::move(observationsFileName));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();
