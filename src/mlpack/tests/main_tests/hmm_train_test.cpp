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

inline void FileExists(std::string fileName)
{
  ifstream ifp(fileName);
  if (!ifp.good())
    BOOST_FAIL("Bad stream " + fileName);
  ifp.close();
}

inline bool ApproximatelyEqual(HMMModel& h1,
                               HMMModel& h2,
                               double tolerance)
{
  if (h1.Type() != h2.Type())
    return false;
  HMMType  hmmType = h1.Type();
  bool transitionEqual = false;
  bool emissionEqual = false;
  bool initialEqual = false;
  if (hmmType ==  DiscreteHMM)
  {
    transitionEqual = approx_equal(
        h1.DiscreteHMM()->Transition(),
        h2.DiscreteHMM()->Transition(),
        "absdiff",
        tolerance
        );
    initialEqual = approx_equal(
        h1.DiscreteHMM()->Transition(),
        h2.DiscreteHMM()->Transition(),
        "absdiff",
        tolerance
        );
    // TODO
    emissionEqual = true;
  }
  else if (hmmType == GaussianHMM)
  {
    transitionEqual = approx_equal(
        h1.GaussianHMM()->Transition(),
        h2.GaussianHMM()->Transition(),
        "absdiff",
        tolerance
        );
    initialEqual = approx_equal(
        h1.GaussianHMM()->Initial(),
        h2.GaussianHMM()->Initial(),
        "absdiff",
        tolerance
        );
    // TODO
    emissionEqual = true;
  }
  else if (hmmType == GaussianMixtureModelHMM)
  {
    transitionEqual = approx_equal(
        h1.GMMHMM()->Transition(),
        h2.GMMHMM()->Transition(),
        "absdiff",
        tolerance
        );
    initialEqual = approx_equal(
        h1.GMMHMM()->Initial(),
        h2.GMMHMM()->Initial(),
        "absdiff",
        tolerance
        );
    // TODO
    emissionEqual = true;
  }
  return emissionEqual && transitionEqual && initialEqual;
}

// Make sure that the number of states cannot be negative
BOOST_AUTO_TEST_CASE(HMMTrainStatesTest)
{
  std::string inputFileName = "hmm_train_obs.csv";
  int states = -3;  // Invalid!
  std::string hmmType = "discrete";

  FileExists(inputFileName);
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

  FileExists(inputFileName);
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

  FileExists(inputFileName);
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

  FileExists(inputFileName);
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

  FileExists(inputObsFileName);
  FileExists(inputLabFileName);
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

  HMMModel h1 = *(CLI::GetParam<HMMModel*>("output_model"));

  SetInputParam("input_model", CLI::GetParam<HMMModel*>("output_model"));

  CLI::GetSingleton().Parameters()["type"].wasPassed = false;
  CLI::GetSingleton().Parameters()["states"].wasPassed = false;

  mlpackMain();

  HMMModel h2 = *(CLI::GetParam<HMMModel*>("output_model"));

  BOOST_REQUIRE(ApproximatelyEqual(h1, h2, 1e-01));
  BOOST_REQUIRE(ApproximatelyEqual(h1, h2, 1e-02));
  BOOST_REQUIRE(ApproximatelyEqual(h1, h2, 1e-03));
  BOOST_REQUIRE(ApproximatelyEqual(h1, h2, 1e-04));
  BOOST_REQUIRE(ApproximatelyEqual(h1, h2, 1e-05));
}

BOOST_AUTO_TEST_CASE(HMMTrainNoLabelsReuseModelTest)
{
  std::string inputObsFileName = "hmm_train_obs.csv";
  std::string hmmType = "discrete";
  int states = 3;

  FileExists(inputObsFileName);
  SetInputParam("input_file", std::move(inputObsFileName));
  SetInputParam("states", states);
  SetInputParam("type", std::move(hmmType));

  // This call will train HMM using Baum-Welch training
  mlpackMain();

  HMMModel h1 = *(CLI::GetParam<HMMModel*>("output_model"));

  SetInputParam("input_model", CLI::GetParam<HMMModel*>("output_model"));

  CLI::GetSingleton().Parameters()["type"].wasPassed = false;
  CLI::GetSingleton().Parameters()["states"].wasPassed = false;

  // Train again using Baum Welch
  mlpackMain();

  HMMModel h2 = *(CLI::GetParam<HMMModel*>("output_model"));

  BOOST_REQUIRE(ApproximatelyEqual(h1, h2, 1e-01));
  BOOST_REQUIRE(ApproximatelyEqual(h1, h2, 1e-02));
  BOOST_REQUIRE(ApproximatelyEqual(h1, h2, 1e-03));
  BOOST_REQUIRE(ApproximatelyEqual(h1, h2, 1e-04));
  BOOST_REQUIRE(ApproximatelyEqual(h1, h2, 1e-05));
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

BOOST_AUTO_TEST_CASE(HMMTrainRetrainTest1)
{
  std::string inputObsFile1 = "obs1.csv";
  std::string type = "discrete";
  int states = 2;

  FileExists(inputObsFile1);
  SetInputParam("input_file", std::move(inputObsFile1));
  SetInputParam("type", std::move(type));
  SetInputParam("states", states);

  mlpackMain();

  HMMModel h1 = *(CLI::GetParam<HMMModel*>("output_model"));

  std::string inputObsFile2 = "obs3.csv";

  CLI::GetSingleton().Parameters()["input_file"].wasPassed = false;
  CLI::GetSingleton().Parameters()["type"].wasPassed = false;
  CLI::GetSingleton().Parameters()["states"].wasPassed = false;

  FileExists(inputObsFile2);
  SetInputParam("input_file", std::move(inputObsFile2));
  SetInputParam("input_model", CLI::GetParam<HMMModel*>("output_model"));

  mlpackMain();

  HMMModel h2 = *(CLI::GetParam<HMMModel*>("output_model"));

  BOOST_REQUIRE(!ApproximatelyEqual(h1, h2, 1e-04));
}

// Attempt to retrain but increase states the second time round
BOOST_AUTO_TEST_CASE(HMMTrainRetrainTest2)
{
  // Provide no labels file
  std::string inputObsFile1 = "obs1.csv";
  std::string type = "discrete";
  int states = 2;

  SetInputParam("input_file", std::move(inputObsFile1));
  SetInputParam("type", std::move(type));
  SetInputParam("states", states);

  mlpackMain();

  HMMModel h1 = *(CLI::GetParam<HMMModel*>("output_model"));

  std::string inputObsFile2 = "obs3.csv";
  std::string inputLabFile2 = "lab1_corrupt.csv";

  SetInputParam("input_file", std::move(inputObsFile2));
  // Provide a labels file with more states than initially specified
  SetInputParam("labels_file", std::move(inputLabFile2));
  SetInputParam("input_model", CLI::GetParam<HMMModel*>("output_model"));

  CLI::GetSingleton().Parameters()["type"].wasPassed = false;
  CLI::GetSingleton().Parameters()["states"].wasPassed = false;

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

// Attempt to retrain but change the emission distribution type
BOOST_AUTO_TEST_CASE(HMMTrainRetrainTest3)
{
  // Provide no labels file
  std::string inputObsFile1 = "obs1.csv";
  std::string type = "discrete";
  int states = 2;

  SetInputParam("input_file", std::move(inputObsFile1));
  SetInputParam("type", std::move(type));
  SetInputParam("states", states);

  mlpackMain();

  HMMModel h1 = *(CLI::GetParam<HMMModel*>("output_model"));

  std::string inputObsFile2 = "obs2.csv";
  std::string inputLabFile2 = "lab2.csv";
  type = "gaussian";

  SetInputParam("input_file", std::move(inputObsFile2));
  // Provide a labels file with more states than initially specified
  SetInputParam("labels_file", std::move(inputLabFile2));
  SetInputParam("type", std::move(type));
  SetInputParam("input_model", CLI::GetParam<HMMModel*>("output_model"));

  CLI::GetSingleton().Parameters()["type"].wasPassed = false;
  CLI::GetSingleton().Parameters()["states"].wasPassed = false;

  HMMModel h2 = *(CLI::GetParam<HMMModel*>("output_model"));

  BOOST_REQUIRE(h1.Type() == DiscreteHMM);
  BOOST_REQUIRE(h2.Type() == DiscreteHMM);
  BOOST_REQUIRE(h2.Type() != GaussianHMM);
}

BOOST_AUTO_TEST_SUITE_END();
