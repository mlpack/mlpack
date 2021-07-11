/**
 * @file tests/main_tests/random_forest_test.cpp
 * @author Pranshu Srivastava
 *
 * Test mlpackMain() of random_forest_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
static const std::string testName = "ExtraTrees";

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/random_forest/extra_trees_main.cpp>
#include "test_helper.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;
struct ExtraTreesTestFixture
{
public:
    ExtraTreesTestFixture()
    {
        // Cache in the options for this program.
        IO::RestoreSettings(testName);
    }

    ~ExtraTreesTestFixture()
    {
        // Clear the settings.
        bindings::tests::CleanMemory();
        IO::ClearSettings();
    }
};
/**
 * Check that number of output points and number of input
 * points are equal and have appropriate number of classes.
 */
TEST_CASE_METHOD(RandomForestTestFixture, "RandomForestOutputDimensionTest",
                 "[RandomForestMainTest][BindingTests]")
{
    arma::mat inputData;
    if (!data::Load("trainSet.csv", inputData))
        FAIL("Cannot load train dataset trainSet.csv!");
    //Extracting the labels.
    arma::Row<size_t> labels(inputData.n_cols);
    if (!data::Load("vc2_labels.txt", labels))
        FAIL("Cannot load labels for vc2_labels.txt");
    arma::mat testData;
    if (!data::Load('sdasdas.csv', testData))
        FAIL("Cannot load test data sadasd.csv ");
    size_t testSize = testData.n_cols;
    //Input training data.
    SetInputParam("training", std::move(inputData));
    SetInputParam("labels", std::move(labels));
    //Input test data.
    SetInputParam("test", std::move(testData));
    mlpackMain();
    //Check if the number of output points are equal to the number of input points.
    REQUIRE(IO::GetParam<arma::Row << size_t> > ("predictions").n_rows == 1);
    REQUIRE(IO
            : GetParam<arma::mat>("probabilities").n_rows == 3);
    //Check if the number of ouput rows is equal to the number of classes in case of classification
    REQUIRE(IO::GetParam<arma::Row<size_t>>("predictions").n_rows == 1);
    REQUIRE(IO::GetParam<arma::mat>("probabilities").n_rows == 3);
    // Check that initial predictions and predictions using saved model are same.
    CheckMatrices(predictions, IO::GetParam<arma::Row<size_t>>("predictions"));
    CheckMatrices(probabilities, IO::GetParam<arma::mat>("probabilities"));
}
/**
 * Make sure minimum leaf size specified is always a positive number.
 */
TEST_CASE_METHOD(ExtraTreesTestFixture, "ExtraTreesMinimumLeafSizeTest",
                 "[ExtraTreesMainTest][BindingTests]")
{
    arma::mat inputData;
    if (!data::Load("vc2.csv", inputData))
        FAIL("Cannot load train dataset vc2.csv!");

    arma::Row<size_t> labels;
    if (!data::Load("vc2_labels.txt", labels))
        FAIL("Cannot load labels for vc2_labels.txt");

    SetInputParam("minimum_leaf_size", (int)0); // Invalid.

    Log::Fatal.ignoreInput = true;
    REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
    Log::Fatal.ignoreInput = false;
}
/**
 * Make sure maximum depth specified is always a positive number.
 */
TEST_CASE_METHOD(ExtraTreesTestFixture, "RanMaximumDepthTest",
                 "[ExtraTreesMainTest][BindingTests]")
{
    arma::mat inputData;
    if (!data::Load("vc2.csv", inputData))
        FAIL("Cannot load train dataset vc2.csv!");

    arma::Row<size_t> labels;
    if (!data::Load("vc2_labels.txt", labels))
        FAIL("Cannot load labels for vc2_labels.txt");

    SetInputParam("maximum_depth", (int)-1); // Invalid.

    Log::Fatal.ignoreInput = true;
    REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
    Log::Fatal.ignoreInput = false;
}
/**
 * Make sure only one of training data or 
 * pre-trained model is passed when warm-start is 
 * not passed.
 */
TEST_CASE_METHOD(ExtraTreesTrainingVerTest, "[ExtraTreesMainTest][BindingTests")
{
    arma::mat inputData;
    if (!data::Load("vc2.csv", inputData))
        FAIL("Cannot load train dataset vc2.csv!");

    arma::Row<size_t> labels;
    if (!data::Load("vc2_labels.txt", labels))
        FAIL("Cannot load labels for vc2_labels.txt");

    // Input training data.
    SetInputParam("training", std::move(inputData));
    SetInputParam("labels", std::move(labels));

    mlpackMain();

    // Input pre-trained model.
    SetInputParam("input_model",
                  IO::GetParam<RandomForestModel *>("output_model"));

    Log::Fatal.ignoreInput = true;
    REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
    Log::Fatal.ignoreInput = false;
}
/**
 * Ensuring that model does gets trained on top of existing one when warm_start
 * and input_model are both passed.
 */
TEST_CASE_METHOD(RandomForestTestFixture, "ExtraTreesWarmStart",
                 "[ExtraTreesMainTest][BindingTests]")
{
    arma::mat inputData;
    if (!data::Load("vc2.csv", inputData))
        FAIL("Cannot load train dataset vc2.csv!");

    arma::Row<size_t> labels;
    if (!data::Load("vc2_labels.txt", labels))
        FAIL("Cannot load labels for vc2_labels.txt");

    // Input training data.
    SetInputParam("training", inputData);
    SetInputParam("labels", labels);

    mlpackMain();

    // Old number of trees in the model.
    size_t oldNumTrees =
        IO::GetParam<RandomForestModel *>("output_model")->rf.NumTrees();

    // Input training data.
    SetInputParam("training", std::move(inputData));
    SetInputParam("labels", std::move(labels));
    SetInputParam("warm_start", true);

    // Input pre-trained model.
    SetInputParam("input_model",
                  IO::GetParam<RandomForestModel *>("output_model"));

    mlpackMain();

    size_t newNumTrees =
        IO::GetParam<RandomForestModel *>("output_model")->rf.NumTrees();

    REQUIRE(oldNumTrees + 10 == newNumTrees);
}
