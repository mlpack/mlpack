/**
 * @file tests/main_tests/hmm_train_test.cpp
 * @author Daivik Nema
 *
 * Test RUN_BINDING() of hmm_train_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/hmm/hmm_model.hpp>
#include <mlpack/methods/hmm/hmm_train_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(HMMTrainMainTestFixture);

inline void FileExists(std::string fileName)
{
  ifstream ifp(fileName);
  if (!ifp.good())
    FAIL("Bad stream " + fileName);
  ifp.close();
}

inline void CheckMatricesDiffer(arma::mat& a, arma::mat& b, double tolerance)
{
  bool dimsEqual = (a.n_rows == b.n_rows)
    && (a.n_cols == b.n_cols)
    && (a.n_elem == b.n_elem);
  bool valsEqual = true;
  if (dimsEqual)
  {
    for (size_t i = 0; i < a.n_elem; ++i)
    {
      if (std::abs(a[i]) < tolerance / 2)
        valsEqual = valsEqual && (std::abs(b[i]) < tolerance / 2);
      else
        valsEqual = valsEqual && (std::abs(a[i] - b[i]) < tolerance);
    }
  }
  REQUIRE(!(dimsEqual && valsEqual));
}

inline void ApproximatelyEqual(HMMModel& h1,
                               HMMModel& h2,
                               double tolerance = 1.0)
{
  REQUIRE(h1.Type() == h2.Type());
  HMMType  hmmType = h1.Type();
  if (hmmType ==  DiscreteHMM)
  {
    CheckMatrices(
        h1.DiscreteHMM()->Transition()*100,
        h2.DiscreteHMM()->Transition()*100,
        tolerance);
    CheckMatrices(
        h1.DiscreteHMM()->Transition()*100,
        h2.DiscreteHMM()->Transition()*100,
        tolerance);

    // Check if emission dists are equal
    std::vector<DiscreteDistribution> d1 = h1.DiscreteHMM()->Emission();
    std::vector<DiscreteDistribution> d2 = h2.DiscreteHMM()->Emission();

    REQUIRE(d1.size() == d2.size());

    size_t states = d1.size();
    for (size_t i = 0; i < states; ++i)
      for (size_t j = 0; j < d1[i].Dimensionality(); ++j)
        CheckMatrices(d1[i].Probabilities(j)*100,
            d2[i].Probabilities(j)*100,
            tolerance);
  }
  else if (hmmType == GaussianHMM)
  {
    CheckMatrices(
        h1.GaussianHMM()->Transition()*100,
        h2.GaussianHMM()->Transition()*100,
        tolerance);
    CheckMatrices(
        h1.GaussianHMM()->Initial()*100,
        h2.GaussianHMM()->Initial()*100,
        tolerance);
    // Check if emission dists are equal by comparing the mean and coviariance
    std::vector<GaussianDistribution> d1 = h1.GaussianHMM()->Emission();
    std::vector<GaussianDistribution> d2 = h2.GaussianHMM()->Emission();

    REQUIRE(d1.size() == d2.size());

    size_t states = d1.size();
    for (size_t i=0; i < states; ++i)
    {
      CheckMatrices(d1[i].Mean()*100, d2[i].Mean()*100, tolerance);
      CheckMatrices(d1[i].Covariance()*100, d2[i].Covariance()*100, tolerance);
    }
  }
  else if (hmmType == GaussianMixtureModelHMM)
  {
    CheckMatrices(
        h1.GMMHMM()->Transition()*100,
        h2.GMMHMM()->Transition()*100,
        tolerance);
    CheckMatrices(
        h1.GMMHMM()->Initial()*100,
        h2.GMMHMM()->Initial()*100,
        tolerance);
    // Check if emission dists are equal
    std::vector<GMM> d1 = h1.GMMHMM()->Emission();
    std::vector<GMM> d2 = h2.GMMHMM()->Emission();

    REQUIRE(d1.size() == d2.size());

    size_t states = d1.size();
    for (size_t i=0; i < states; ++i)
    {
      REQUIRE(d1[i].Gaussians() == d2[i].Gaussians());
      size_t gaussians = d1[i].Gaussians();
      for (size_t j=0; j<gaussians; ++j)
      {
        CheckMatrices(d1[i].Component(j).Mean()*100,
            d2[i].Component(j).Mean()*100,
            tolerance);
        CheckMatrices(d1[i].Component(j).Covariance()*100,
            d2[i].Component(j).Covariance()*100,
            tolerance);
      }
      CheckMatrices(d1[i].Weights()*100, d2[i].Weights()*100, tolerance);
    }
  }
  else if (hmmType == DiagonalGaussianMixtureModelHMM)
  {
    CheckMatrices(
        h1.DiagGMMHMM()->Transition()*100,
        h2.DiagGMMHMM()->Transition()*100,
        tolerance);
    CheckMatrices(
        h1.DiagGMMHMM()->Initial()*100,
        h2.DiagGMMHMM()->Initial()*100,
        tolerance);
    // Check if emission dists are equal.
    std::vector<DiagonalGMM> d1 = h1.DiagGMMHMM()->Emission();
    std::vector<DiagonalGMM> d2 = h2.DiagGMMHMM()->Emission();

    REQUIRE(d1.size() == d2.size());

    // Check if gaussian, mean, covariance and weights are equal.
    size_t states = d1.size();
    for (size_t i = 0; i < states; ++i)
    {
      REQUIRE(d1[i].Gaussians() == d2[i].Gaussians());
      size_t gaussians = d1[i].Gaussians();
      for (size_t j = 0; j < gaussians; ++j)
      {
        CheckMatrices(d1[i].Component(j).Mean()*100,
            d2[i].Component(j).Mean()*100,
            tolerance);
        CheckMatrices(d1[i].Component(j).Covariance()*100,
            d2[i].Component(j).Covariance()*100,
            tolerance);
      }
      CheckMatrices(d1[i].Weights()*100, d2[i].Weights()*100, tolerance);
    }
  }
}

// Make sure that the number of states cannot be negative
TEST_CASE_METHOD(HMMTrainMainTestFixture, "HMMTrainStatesTest",
                 "[HMMTrainMainTest][BindingTests]")
{
  std::string inputFileName = "hmm_train_obs.csv";
  int states = -3;  // Invalid!
  std::string hmmType = "discrete";

  FileExists(inputFileName);
  SetInputParam("input_file", std::move(inputFileName));
  SetInputParam("states", states);
  SetInputParam("type", std::move(hmmType));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

// Make sure that tolerance is non negative
TEST_CASE_METHOD(HMMTrainMainTestFixture, "HMMTrainToleranceNonNegative",
                 "[HMMTrainMainTest][BindingTests]")
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

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

// Make sure an error is thrown if type is something other than
// "discrete", "gaussian" or "gmm"
TEST_CASE_METHOD(HMMTrainMainTestFixture, "HMMTrainTypeTest",
                 "[HMMTrainMainTest][BindingTests]")
{
  std::string inputFileName = "hmm_train_obs.csv";
  int states = 3;
  std::string hmmType = "some-not-supported-possibly-non-type";

  FileExists(inputFileName);
  SetInputParam("input_file", std::move(inputFileName));
  SetInputParam("states", states);
  SetInputParam("type", std::move(hmmType));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

// Make sure that the number of gaussians cannot be less than 0
TEST_CASE_METHOD(HMMTrainMainTestFixture, "HMMTrainGaussianTest",
                 "[HMMTrainMainTest][BindingTests]")
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

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

// Make sure that the number of Gaussians cannot be less than 0.
TEST_CASE_METHOD(HMMTrainMainTestFixture, "HMMTrainDiagonalGaussianTest",
                 "[HMMTrainMainTest][BindingTests]")
{
  std::string inputFileName = "hmm_train_obs.csv";
  int states = 3;
  std::string hmmType = "diag_gmm";
  int gaussians = -2;

  FileExists(inputFileName);
  SetInputParam("input_file", std::move(inputFileName));
  SetInputParam("states", states);
  SetInputParam("type", std::move(hmmType));
  SetInputParam("gaussians", gaussians);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

// Make sure that model reuse is possible and work properly
TEST_CASE_METHOD(HMMTrainMainTestFixture, "HMMTrainReuseDiscreteModelTest",
                 "[HMMTrainMainTest][BindingTests]")
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
  REQUIRE(trainObs.n_rows == trainLab.n_rows);

  SetInputParam("input_file", inputObsFileName);
  SetInputParam("labels_file", inputLabFileName);
  SetInputParam("type", hmmType);
  SetInputParam("states", states);

  RUN_BINDING();

  HMMModel h1 = *(params.Get<HMMModel*>("output_model"));
  params.Get<HMMModel*>("output_model") = NULL;

  CleanMemory();
  ResetSettings();

  SetInputParam("input_model", &h1);
  SetInputParam("input_file", std::move(inputObsFileName));
  SetInputParam("labels_file", std::move(inputLabFileName));

  RUN_BINDING();

  HMMModel h2 = *(params.Get<HMMModel*>("output_model"));

  ResetSettings();

  ApproximatelyEqual(h1, h2);
}

// Make sure that model reuse is possible and work properly
TEST_CASE_METHOD(HMMTrainMainTestFixture, "HMMTrainReuseGaussianModelTest",
                 "[HMMTrainMainTest][BindingTests]")
{
  std::string inputObsFileName = "hmm_train_obs.csv";
  std::string hmmType = "gaussian";
  int states = 3;

  FileExists(inputObsFileName);
  // Make sure that the size of the
  // training seq, and training labels is same
  arma::mat trainObs;
  data::Load(inputObsFileName, trainObs);

  SetInputParam("input_file", inputObsFileName);
  SetInputParam("type", std::move(hmmType));
  SetInputParam("states", states);

  RUN_BINDING();

  HMMModel h1 = *(params.Get<HMMModel*>("output_model"));

  ResetSettings();

  SetInputParam("input_model", &h1);
  SetInputParam("input_file", std::move(inputObsFileName));
  SetInputParam("tolerance", 1e10);

  RUN_BINDING();

  HMMModel h2 = *(params.Get<HMMModel*>("output_model"));

  ResetSettings();

  ApproximatelyEqual(h1, h2);
}

TEST_CASE_METHOD(HMMTrainMainTestFixture, "HMMTrainNoLabelsReuseModelTest",
                 "[HMMTrainMainTest][BindingTests]")
{
  std::string inputObsFileName = "hmm_train_obs.csv";
  std::string hmmType = "discrete";
  int states = 3;
  int seed = 0;

  FileExists(inputObsFileName);
  SetInputParam("input_file", inputObsFileName);
  SetInputParam("states", states);
  SetInputParam("type", std::move(hmmType));
  SetInputParam("seed", seed);

  // This call will train HMM using Baum-Welch training
  RUN_BINDING();

  HMMModel h1 = *(params.Get<HMMModel*>("output_model"));

  ResetSettings();

  SetInputParam("input_model", &h1);
  SetInputParam("input_file", std::move(inputObsFileName));

  // Train again using Baum Welch
  RUN_BINDING();

  HMMModel h2 = *(params.Get<HMMModel*>("output_model"));

  ResetSettings();

  ApproximatelyEqual(h1, h2);
}

// Test batch mode
TEST_CASE_METHOD(HMMTrainMainTestFixture, "HMMTrainBatchModeTest",
                 "[HMMTrainMainTest][BindingTests]")
{
  std::string observationsFileName = "observations.txt";
  std::string labelsFileName = "labels.txt";
  std::string hmmType = "discrete";
  int states = 2;

  SetInputParam("input_file", std::move(observationsFileName));
  SetInputParam("labels_file", std::move(labelsFileName));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  SetInputParam("states", states);
  SetInputParam("type", std::move(hmmType));
  SetInputParam("batch", (bool) true);

  RUN_BINDING();

  // Now pass an observations file with extra non-existent filenames
  observationsFileName = "corrupt-observations-1.txt";
  SetInputParam("input_file", std::move(observationsFileName));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  // Now a mismatch between #observation files and #label files
  observationsFileName = "corrupt-observations-2.txt";
  SetInputParam("input_file", std::move(observationsFileName));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

TEST_CASE_METHOD(HMMTrainMainTestFixture, "HMMTrainRetrainTest1",
                 "[HMMTrainMainTest][BindingTests]")
{
  std::string inputObsFile1 = "obs1.csv";
  std::string type = "discrete";
  int states = 2;
  int seed = 0;

  FileExists(inputObsFile1);
  SetInputParam("input_file", std::move(inputObsFile1));
  SetInputParam("type", std::move(type));
  SetInputParam("states", states);
  SetInputParam("seed", seed);

  RUN_BINDING();

  HMMModel h1 = *(params.Get<HMMModel*>("output_model"));
  arma::mat h1Transition = h1.DiscreteHMM()->Transition();

  std::string inputObsFile2 = "obs4.csv";
  FileExists(inputObsFile2);

  ResetSettings();

  SetInputParam("input_file", std::move(inputObsFile2));
  SetInputParam("input_model", &h1);

  RUN_BINDING();

  HMMModel h2 = *(params.Get<HMMModel*>("output_model"));

  ResetSettings();

  REQUIRE(h1.Type() == h2.Type());
  // Since we know that type of HMMs is discrete
  CheckMatricesDiffer(h1Transition, h2.DiscreteHMM()->Transition(), 1e-50);
}

// Attempt to retrain but increase states the second time round
TEST_CASE_METHOD(HMMTrainMainTestFixture, "HMMTrainRetrainTest2",
                 "[HMMTrainMainTest][BindingTests]")
{
  // Provide no labels file
  std::string inputObsFile1 = "obs1.csv";
  std::string type = "discrete";
  int states = 2;

  SetInputParam("input_file", std::move(inputObsFile1));
  SetInputParam("type", std::move(type));
  SetInputParam("states", states);

  RUN_BINDING();

  HMMModel h1 = *(params.Get<HMMModel*>("output_model"));

  std::string inputObsFile2 = "obs3.csv";
  std::string inputLabFile2 = "lab1_corrupt.csv";

  ResetSettings();

  SetInputParam("input_file", std::move(inputObsFile2));
  // Provide a labels file with more states than initially specified
  SetInputParam("labels_file", std::move(inputLabFile2));
  SetInputParam("input_model", &h1);

  ResetSettings();

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

// Attempt to retrain but change the emission distribution type
TEST_CASE_METHOD(HMMTrainMainTestFixture, "HMMTrainRetrainTest3",
                 "[HMMTrainMainTest][BindingTests]")
{
  // Provide no labels file
  std::string inputObsFile1 = "obs1.csv";
  std::string type = "discrete";
  int states = 2;

  SetInputParam("input_file", std::move(inputObsFile1));
  SetInputParam("type", std::move(type));
  SetInputParam("states", states);

  RUN_BINDING();

  HMMModel h1 = *(params.Get<HMMModel*>("output_model"));

  std::string inputObsFile2 = "obs2.csv";
  std::string inputLabFile2 = "lab2.csv";
  type = "gaussian";

  ResetSettings();

  SetInputParam("input_file", std::move(inputObsFile2));
  SetInputParam("labels_file", std::move(inputLabFile2));
  SetInputParam("type", std::move(type));
  SetInputParam("input_model", &h1);

  RUN_BINDING();
  // Note that when emission type is changed -- like in this test, a warning
  // is printed stating that the new type is being ignored (no error is raised)

  HMMModel h2 = *(params.Get<HMMModel*>("output_model"));

  ResetSettings();

  REQUIRE(h1.Type() == DiscreteHMM);
  REQUIRE(h2.Type() == DiscreteHMM);
  REQUIRE(h2.Type() != GaussianHMM);
}
