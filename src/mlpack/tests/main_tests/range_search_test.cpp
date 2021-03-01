/**
 * @file tests/main_tests/range_search_test.cpp
 * @author Niteya Shah
 *
 * Test mlpackMain() of range_search_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "RangeSearchMain";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/range_search/range_search_main.cpp>
#include "range_search_utils.hpp"
#include "../catch.hpp"

using namespace mlpack;

struct RangeSearchTestFixture
{
 public:
  RangeSearchTestFixture()
  {
    // Cache in the options for this program.
    IO::RestoreSettings(testName);
  }
  ~RangeSearchTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    IO::ClearSettings();
  }
};

/**
 * Check that we have to specify a reference set or input model.
 */
TEST_CASE_METHOD(RangeSearchTestFixture, "RangeSearchNoReference",
                 "[RangeSearchMainTest][BindingTests]")
{
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that we cannot pass an incorrect parameter.
 */
TEST_CASE_METHOD(RangeSearchTestFixture, "RangeSearchWrongParameter",
                 "[RangeSearchMainTest][BindingTests]")
{
  string wrongString = "abc";

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(SetInputParam("RST", wrongString), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that we have to specify a query if an input model is specified.
 */
TEST_CASE_METHOD(RangeSearchTestFixture, "RangeSearchInputModelNoQuery",
                 "[RangeSearchMainTest][BindingTests]")
{
  arma::mat inputData;
  double minVal = 0, maxVal = 3;
  string distanceFile = "distances.csv";
  string neighborsFile = "neighbors.csv";

  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");

  SetInputParam("reference", move(inputData));
  SetInputParam("min", minVal);
  SetInputParam("max", maxVal);
  SetInputParam("distances_file", distanceFile);
  SetInputParam("neighbors_file", neighborsFile);

  mlpackMain();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  SetInputParam("input_model", move(IO::GetParam<RSModel*>("output_model")));

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  remove(neighborsFile.c_str());
  remove(distanceFile.c_str());
}

/**
 * Check that we cannot specify a tree type which is not available or wrong.
 */
TEST_CASE_METHOD(RangeSearchTestFixture, "RangeSearchDifferentTree",
                 "[RangeSearchMainTest][BindingTests]")
{
  arma::mat inputData;
  double minVal = 0, maxVal = 3;
  string distanceFile = "distances.csv";
  string neighborsFile = "neighbors.csv";
  string wrongTreeType = "RST";
  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");

  SetInputParam("reference", move(inputData));
  SetInputParam("min", minVal);
  SetInputParam("max", maxVal);
  SetInputParam("distances_file", distanceFile);
  SetInputParam("neighbors_file", neighborsFile);
  SetInputParam("tree_type", wrongTreeType);

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  remove(neighborsFile.c_str());
  remove(distanceFile.c_str());
}

/**
 * Check that we cannot specify both a reference set and input model.
 */
TEST_CASE_METHOD(RangeSearchTestFixture, "RangeSearchBothReferenceAndModel",
                 "[RangeSearchMainTest][BindingTests]")
{
  arma::mat inputData, queryData;
  double minVal = 0, maxVal = 3;
  string distanceFile = "distances.csv";
  string neighborsFile = "neighbors.csv";

  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");
  if (!data::Load("iris_test.csv", queryData))
    FAIL("Unable to load dataset iris_test.csv!");

  SetInputParam("reference", move(inputData));
  SetInputParam("min", minVal);
  SetInputParam("max", maxVal);
  SetInputParam("distances_file", distanceFile);
  SetInputParam("neighbors_file", neighborsFile);
  SetInputParam("query", queryData);

  mlpackMain();

  SetInputParam("input_model", move(IO::GetParam<RSModel*>("output_model")));
  SetInputParam("query", move(queryData));

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  remove(neighborsFile.c_str());
  remove(distanceFile.c_str());
}

/**
 * Check that the correct output is returned for a small synthetic input case,
 * by comparing with pre-calculated neighbor and distance values, when no query
 * set is specified.
 */
TEST_CASE_METHOD(RangeSearchTestFixture, "RangeSearchTest",
                 "[RangeSearchMainTest][BindingTests]")
{
  arma::mat x = {{0, 3, 3, 4, 3, 1},
                 {4, 4, 4, 5, 5, 2},
                 {0, 1, 2, 2, 3, 3}};

  string distanceFile = "distances.csv";
  string neighborsFile = "neighbors.csv";
  double minVal = 0, maxVal = 3;
  vector<vector<size_t>> neighborVal = {{},
                                        {2, 3, 4},
                                        {1, 3, 4, 5},
                                        {1, 2, 4},
                                        {1, 2, 3},
                                        {2}};
  vector<vector<double>> distanceVal = {{},
                                        {1, 1.73205, 2.23607},
                                        {1, 1.41421, 1.41421, 3},
                                        {1.73205, 1.41421, 1.41421},
                                        {2.23607, 1.41421, 1.41421},
                                        {3}};

  vector<vector<size_t>> neighbors;
  vector<vector<double>> distances;

  SetInputParam("reference", move(x));
  SetInputParam("min", minVal);
  SetInputParam("max", maxVal);
  SetInputParam("distances_file", distanceFile);
  SetInputParam("neighbors_file", neighborsFile);

  mlpackMain();

  neighbors = ReadData<size_t>(neighborsFile);
  distances = ReadData<double>(distanceFile);

  CheckMatrices(neighbors, neighborVal);
  CheckMatrices(distances, distanceVal);

  remove(neighborsFile.c_str());
  remove(distanceFile.c_str());
}

/**
 * Check that the correct output is returned for a small synthetic input case,
 * when a query set is provided.
 */
TEST_CASE_METHOD(RangeSearchTestFixture, "RangeSeachTestwithQuery",
                 "[RangeSearchMainTest][BindingTests]")
{
  arma::mat queryData = {{5, 3, 1}, {4, 2, 4}, {3, 1, 7}};
  arma::mat x = {{0, 3, 3, 4, 3, 1},
                 {4, 4, 4, 5, 5, 2},
                 {0, 1, 2, 2, 3, 3}};

  vector<vector<double>> distanceVal = {
                {2.82843, 2.23607, 1.73205, 2.23607, 4.47214},
                {3.74166, 2, 2.23607, 3.31662, 3.60555, 2.82843},
                {4.58258, 4.47214}};
  vector<vector<size_t>> neighborVal = {{1, 2, 3, 4, 5},
                                        {0, 1, 2, 3, 4, 5},
                                        {4, 5}};

  vector<vector<size_t>> neighbors;
  vector<vector<double>> distances;
  string distanceFile = "distances.csv";
  string neighborsFile = "neighbors.csv";
  double minVal = 0, maxVal = 5;

  SetInputParam("query", queryData);
  SetInputParam("reference", move(x));
  SetInputParam("min", minVal);
  SetInputParam("max", maxVal);
  SetInputParam("distances_file", distanceFile);
  SetInputParam("neighbors_file", neighborsFile);

  mlpackMain();

  neighbors = ReadData<size_t>(neighborsFile);
  distances = ReadData<double>(distanceFile);

  CheckMatrices(neighbors, neighborVal);
  CheckMatrices(distances, distanceVal);

  remove(neighborsFile.c_str());
  remove(distanceFile.c_str());
}

/**
 * Train a model using a synthetic dataset and then output the model, and ensure
 * it can be used again.
 */
TEST_CASE_METHOD(RangeSearchTestFixture, "ModelCheck",
                 "[RangeSearchMainTest][BindingTests]")
{
  arma::mat inputData, queryData;
  double minVal = 0, maxVal = 3;
  string distanceFile = "distances.csv";
  string neighborsFile = "neighbors.csv";
  vector<vector<size_t>> neighbors, neighborsTemp;
  vector<vector<double>> distances, distancetemp;

  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");
  if (!data::Load("iris_test.csv", queryData))
    FAIL("Unable to load dataset iris_test.csv!");

  SetInputParam("reference", move(inputData));
  SetInputParam("min", minVal);
  SetInputParam("max", maxVal);
  SetInputParam("distances_file", distanceFile);
  SetInputParam("neighbors_file", neighborsFile);
  SetInputParam("query", queryData);

  mlpackMain();

  neighbors = ReadData<size_t>(neighborsFile);
  distances = ReadData<double>(distanceFile);

  RSModel* outputModel = IO::GetParam<RSModel*>("output_model");
  IO::GetSingleton().Parameters()["reference"].wasPassed = false;

  SetInputParam("input_model", outputModel);
  SetInputParam("query", move(queryData));

  mlpackMain();

  neighborsTemp = ReadData<size_t>(neighborsFile);
  distancetemp = ReadData<double>(distanceFile);

  CheckMatrices(neighbors, neighborsTemp);
  CheckMatrices(distances, distancetemp);

  REQUIRE(ModelToString(outputModel) ==
          ModelToString(IO::GetParam<RSModel*>("output_model")));

  remove(neighborsFile.c_str());
  remove(distanceFile.c_str());
}

/**
 * Check that the models are different but the results are the same for three
 * different leaf size parameters.
 */
TEST_CASE_METHOD(RangeSearchTestFixture, "LeafValueTesting",
                 "[RangeSearchMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");

  string distanceFile = "distances.csv";
  string neighborsFile = "neighbors.csv";
  double minVal = 0, maxVal = 3;

  vector<vector<size_t>> neighbors, neighborsTemp;
  vector<vector<double>> distances, distancestemp;

  vector<int> leafSizes {20, 15, 25};

  SetInputParam("reference", inputData);
  SetInputParam("min", minVal);
  SetInputParam("max", maxVal);
  SetInputParam("distances_file", distanceFile);
  SetInputParam("neighbors_file", neighborsFile);
  SetInputParam("leaf_size", leafSizes[0]);
  // The default leaf size is 20.

  mlpackMain();

  RSModel* outputModel1 = IO::GetParam<RSModel*>("output_model");
  neighbors = ReadData<size_t>(neighborsFile);
  distances = ReadData<double>(distanceFile);

  for (size_t i = 1; i < leafSizes.size(); ++i)
  {
    SetInputParam("leaf_size", leafSizes[i]);
    SetInputParam("reference", inputData);
    SetInputParam("min", minVal);
    SetInputParam("max", maxVal);
    SetInputParam("distances_file", distanceFile);
    SetInputParam("neighbors_file", neighborsFile);

    mlpackMain();

    neighborsTemp = ReadData<size_t>(neighborsFile);
    distancestemp = ReadData<double>(distanceFile);

    CheckMatrices(neighbors, neighborsTemp);
    CheckMatrices(distances, distancestemp);

    REQUIRE(ModelToString(outputModel1) !=
            ModelToString(IO::GetParam<RSModel*>("output_model")));

    if (i != leafSizes.size() - 1)
      delete IO::GetParam<RSModel*>("output_model");
  }

  delete outputModel1;

  remove(neighborsFile.c_str());
  remove(distanceFile.c_str());
}

/**
 * Make sure that the models are different but the results are the same for
 * different tree types.  We use the default kd-tree as the base model to
 * compare against.
 */
TEST_CASE_METHOD(RangeSearchTestFixture, "TreeTypeTesting",
                 "[RangeSearchMainTest][BindingTests]")
{
  string distanceFile = "distances.csv";
  string neighborsFile = "neighbors.csv";

  double minVal = 0, maxVal = 3;
  arma::mat queryData, inputData;
  vector<vector<size_t>> neighbors, neighborsTemp;
  vector<vector<double>> distances, distancestemp;
  vector<string> trees = {"kd", "cover", "r", "r-star", "ball", "x",
                          "hilbert-r", "r-plus", "r-plus-plus", "vp", "rp",
                          "max-rp", "ub", "oct"};

  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");
  if (!data::Load("iris_test.csv", queryData))
    FAIL("Unable to load dataset iris_test.csv!");

  // Define base parameters with the kd-tree.
  SetInputParam("tree_type", trees[0]);
  SetInputParam("min", minVal);
  SetInputParam("max", maxVal);
  SetInputParam("distances_file", distanceFile);
  SetInputParam("neighbors_file", neighborsFile);
  SetInputParam("reference", inputData);
  SetInputParam("query", queryData);

  mlpackMain();

  neighbors = ReadData<size_t>(neighborsFile);
  distances = ReadData<double>(distanceFile);
  RSModel* outputModel1 = IO::GetParam<RSModel*>("output_model");

  for (size_t i = 1; i < trees.size(); ++i)
  {
    if (!data::Load("iris.csv", inputData))
      FAIL("Unable to load dataset iris.csv!");
    if (!data::Load("iris_test.csv", queryData))
      FAIL("Unable to load dataset iris_test.csv!");

    SetInputParam("min", minVal);
    SetInputParam("max", maxVal);
    SetInputParam("distances_file", distanceFile);
    SetInputParam("neighbors_file", neighborsFile);
    SetInputParam("query", queryData);
    SetInputParam("reference", inputData);
    SetInputParam("tree_type", trees[i]);

    mlpackMain();

    neighborsTemp = ReadData<size_t>(neighborsFile);
    distancestemp = ReadData<double>(distanceFile);

    CheckMatrices(neighbors, neighborsTemp);
    CheckMatrices(distances, distancestemp);
    REQUIRE(ModelToString(outputModel1) !=
            ModelToString(IO::GetParam<RSModel*>("output_model")));

    if (i != trees.size() - 1)
      delete IO::GetParam<RSModel*>("output_model");
  }

  delete outputModel1;

  remove(neighborsFile.c_str());
  remove(distanceFile.c_str());
}

/**
 * Project the data onto a random basis and ensure that this gives identical
 * results to non-projected data but different models.
 */
TEST_CASE_METHOD(RangeSearchTestFixture, "RandomBasisTesting",
                 "[RangeSearchMainTest][BindingTests]")
{
  string distanceFile = "distances.csv";
  string neighborsFile = "neighbors.csv";
  double minVal = 0, maxVal = 3;

  arma::mat queryData, inputData;
  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");
  if (!data::Load("iris_test.csv", queryData))
    FAIL("Unable to load dataset iris_test.csv!");

  SetInputParam("min", minVal);
  SetInputParam("max", maxVal);
  SetInputParam("distances_file", distanceFile);
  SetInputParam("neighbors_file", neighborsFile);
  SetInputParam("reference", inputData);

  mlpackMain();

  RSModel* outputModel = move(IO::GetParam<RSModel*>("output_model"));

  SetInputParam("min", minVal);
  SetInputParam("max", maxVal);
  SetInputParam("distances_file", distanceFile);
  SetInputParam("neighbors_file", neighborsFile);
  SetInputParam("reference", inputData);
  SetInputParam("random_basis", true);

  mlpackMain();

  REQUIRE(ModelToString(outputModel) !=
          ModelToString(IO::GetParam<RSModel*>("output_model")));

  delete outputModel;

  remove(neighborsFile.c_str());
  remove(distanceFile.c_str());
}

/**
 * Ensure that naive mode gives the same result, but different models.
 */
TEST_CASE_METHOD(RangeSearchTestFixture, "NaiveModeTest",
                 "[RangeSearchMainTest][BindingTests]")
{
  string distanceFile = "distances.csv";
  string neighborsFile = "neighbors.csv";
  double minVal = 0, maxVal = 3;

  arma::mat queryData, inputData;
  vector<vector<size_t>> neighbors, neighborsTemp;
  vector<vector<double>> distances, distancestemp;

  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");
  if (!data::Load("iris_test.csv", queryData))
    FAIL("Unable to load dataset iris_test.csv!");

  SetInputParam("min", minVal);
  SetInputParam("max", maxVal);
  SetInputParam("distances_file", distanceFile);
  SetInputParam("neighbors_file", neighborsFile);
  SetInputParam("reference", inputData);

  mlpackMain();

  neighbors = ReadData<size_t>(neighborsFile);
  distances = ReadData<double>(distanceFile);
  RSModel* outputModel = move(IO::GetParam<RSModel*>("output_model"));

  SetInputParam("min", minVal);
  SetInputParam("max", maxVal);
  SetInputParam("distances_file", distanceFile);
  SetInputParam("neighbors_file", neighborsFile);
  SetInputParam("reference", inputData);
  SetInputParam("naive", true);

  mlpackMain();

  neighborsTemp = ReadData<size_t>(neighborsFile);
  distancestemp = ReadData<double>(distanceFile);

  CheckMatrices(neighbors, neighborsTemp);
  CheckMatrices(distances, distancestemp);

  REQUIRE(ModelToString(outputModel) !=
          ModelToString(IO::GetParam<RSModel*>("output_model")));

  delete outputModel;

  remove(neighborsFile.c_str());
  remove(distanceFile.c_str());
}

/**
 * Ensure that single-tree mode gives the same result but different models.
 */
TEST_CASE_METHOD(RangeSearchTestFixture, "SingleModeTest",
                 "[RangeSearchMainTest][BindingTests]")
{
  string distanceFile = "distances.csv";
  string neighborsFile = "neighbors.csv";
  double minVal = 0, maxVal = 3;

  arma::mat queryData, inputData;
  vector<vector<size_t>> neighbors, neighborsTemp;
  vector<vector<double>> distances, distancestemp;

  if (!data::Load("iris.csv", inputData))
    FAIL("Unable to load dataset iris.csv!");
  if (!data::Load("iris_test.csv", queryData))
    FAIL("Unable to load dataset iris_test.csv!");

  SetInputParam("min", minVal);
  SetInputParam("max", maxVal);
  SetInputParam("distances_file", distanceFile);
  SetInputParam("neighbors_file", neighborsFile);
  SetInputParam("reference", inputData);

  mlpackMain();

  neighbors = ReadData<size_t>(neighborsFile);
  distances = ReadData<double>(distanceFile);
  RSModel* outputModel = move(IO::GetParam<RSModel*>("output_model"));

  SetInputParam("min", minVal);
  SetInputParam("max", maxVal);
  SetInputParam("distances_file", distanceFile);
  SetInputParam("neighbors_file", neighborsFile);
  SetInputParam("reference", inputData);
  SetInputParam("single_mode", true);

  mlpackMain();

  neighborsTemp = ReadData<size_t>(neighborsFile);
  distancestemp = ReadData<double>(distanceFile);

  CheckMatrices(neighbors, neighborsTemp);
  CheckMatrices(distances, distancestemp);
  REQUIRE(ModelToString(outputModel) !=
          ModelToString(IO::GetParam<RSModel*>("output_model")));

  delete outputModel;

  remove(neighborsFile.c_str());
  remove(distanceFile.c_str());
}
