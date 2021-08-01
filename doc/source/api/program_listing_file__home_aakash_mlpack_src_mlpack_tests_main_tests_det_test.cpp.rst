
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_main_tests_det_test.cpp:

Program Listing for File det_test.cpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_main_tests_det_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/main_tests/det_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <string>
   
   #define BINDING_TYPE BINDING_TYPE_TEST
   
   static const std::string testName = "DET";
   
   #include <mlpack/core.hpp>
   #include <mlpack/core/util/mlpack_main.hpp>
   #include "test_helper.hpp"
   #include <mlpack/methods/det/det_main.cpp>
   
   #include "../test_catch_tools.hpp"
   #include "../catch.hpp"
   
   using namespace mlpack;
   
   struct DETTestFixture
   {
    public:
     DETTestFixture()
     {
       // Cache in the options for this program.
       IO::RestoreSettings(testName);
     }
   
     ~DETTestFixture()
     {
       // Clear the settings.
       bindings::tests::CleanMemory();
       IO::ClearSettings();
     }
   };
   
   TEST_CASE_METHOD(DETTestFixture, "DETOutputDimensionTest",
                   "[DETMainTest][BindingTests]")
   {
     arma::mat trainingData;
     if (!data::Load("iris.csv", trainingData))
       FAIL("Unable to load dataset iris.csv!");
   
     arma::mat testData;
     if (!data::Load("iris_test.csv", testData))
       FAIL("Unable to load dataset iris_test.csv!");
   
     // Input data.
     SetInputParam("training", trainingData);
     SetInputParam("test", testData);
   
     mlpackMain();
   
     // Check the training_set_estimates has 100 points.
     REQUIRE(IO::GetParam<arma::mat>("training_set_estimates").n_rows == 1);
     REQUIRE(IO::GetParam<arma::mat>("training_set_estimates").n_cols ==
             trainingData.n_cols);
   
     // Check the test_set_estimates has 40 points.
     REQUIRE(IO::GetParam<arma::mat>("test_set_estimates").n_rows == 1);
     REQUIRE(IO::GetParam<arma::mat>("test_set_estimates").n_cols ==
             testData.n_cols);
   }
   
   TEST_CASE_METHOD(DETTestFixture, "DETParamBoundTest",
                   "[DETMainTest][BindingTests]")
   {
     arma::mat trainingData;
     if (!data::Load("iris.csv", trainingData))
       FAIL("Unable to load dataset iris.csv!");
   
     // Test for max_leaf_size.
   
     SetInputParam("training", trainingData);
     SetInputParam("max_leaf_size", (int) 0);
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   
     bindings::tests::CleanMemory();
   
     // Test for min_leaf_size.
   
     SetInputParam("training", trainingData);
     SetInputParam("min_leaf_size", (int) 0);
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   
     bindings::tests::CleanMemory();
   
     // Test for folds.
   
     SetInputParam("training", move(trainingData));
     SetInputParam("folds", (int) -1);
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(DETTestFixture, "DETModelReuseTest",
                   "[DETMainTest][BindingTests]")
   {
     arma::mat trainingData;
     if (!data::Load("iris.csv", trainingData))
       FAIL("Unable to load dataset iris.csv!");
   
     arma::mat testData;
     if (!data::Load("iris_test.csv", testData))
       FAIL("Unable to load dataset iris_test.csv!");
   
     // Input data.
     SetInputParam("training", std::move(trainingData));
     SetInputParam("test", testData);
   
     mlpackMain();
   
     arma::mat trainingSetEstimates =
         IO::GetParam<arma::mat>("training_set_estimates");
     arma::mat testSetEstimates = IO::GetParam<arma::mat>("test_set_estimates");
   
     IO::GetSingleton().Parameters()["training"].wasPassed = false;
   
     SetInputParam("input_model", IO::GetParam<DTree<>*>("output_model"));
     SetInputParam("test", std::move(testData));
   
     mlpackMain();
   
     // Check that initial estimates and final estimate using saved model are same.
     CheckMatrices(trainingSetEstimates,
                   IO::GetParam<arma::mat>("training_set_estimates"));
     CheckMatrices(testSetEstimates,
                   IO::GetParam<arma::mat>("test_set_estimates"));
   }
   
   TEST_CASE_METHOD(DETTestFixture, "DETViDimensionTest",
                   "[DETMainTest][BindingTests]")
   {
     arma::mat trainingData;
     if (!data::Load("iris.csv", trainingData))
       FAIL("Unable to load dataset iris.csv!");
   
     arma::mat testData;
     if (!data::Load("iris_test.csv", testData))
       FAIL("Unable to load dataset iris_test.csv!");
   
     size_t testRows = testData.n_rows;
   
     // Input data.
     SetInputParam("training", std::move(trainingData));
     SetInputParam("test", std::move(testData));
   
     mlpackMain();
   
     // Check the number of output points equals number of input features.
     REQUIRE(IO::GetParam<arma::mat>("vi").n_rows == 1);
     REQUIRE(IO::GetParam<arma::mat>("vi").n_cols == testRows);
   }
   
   TEST_CASE_METHOD(DETTestFixture, "DETModelValidityTest",
                   "[DETMainTest][BindingTests]")
   {
     arma::mat trainingData;
     if (!data::Load("iris.csv", trainingData))
       FAIL("Unable to load dataset iris.csv!");
   
     SetInputParam("training", std::move(trainingData));
   
     mlpackMain();
   
     SetInputParam("input_model", IO::GetParam<DTree<>*>("output_model"));
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(DETTestFixture, "DETDiffMinLeafTest",
                   "[DETMainTest][BindingTests]")
   {
     arma::mat trainingData;
     if (!data::Load("iris.csv", trainingData))
       FAIL("Unable to load dataset iris.csv!");
   
     arma::mat testData;
     if (!data::Load("iris_test.csv", testData))
       FAIL("Unable to load dataset iris_test.csv!");
   
     // Input data.
     SetInputParam("training", trainingData);
     SetInputParam("test", testData);
   
     mlpackMain();
   
     arma::mat trainingSetEstimates =
         IO::GetParam<arma::mat>("training_set_estimates");
     arma::mat testSetEstimates = IO::GetParam<arma::mat>("test_set_estimates");
   
     bindings::tests::CleanMemory();
   
     // Train model using min_leaf_size equals to 10.
   
     SetInputParam("training", std::move(trainingData));
     SetInputParam("test", std::move(testData));
     SetInputParam("min_leaf_size", (int) 10);
   
     mlpackMain();
   
     // Check that initial estimates and final estimates using two models are
     // different.
     REQUIRE(arma::accu(trainingSetEstimates ==
         IO::GetParam<arma::mat>("training_set_estimates")) <
         trainingSetEstimates.n_elem);
   
     REQUIRE(arma::accu(testSetEstimates ==
         IO::GetParam<arma::mat>("test_set_estimates")) <
         testSetEstimates.n_elem);
   }
   
   TEST_CASE_METHOD(DETTestFixture, "DETDiffMaxLeafTest",
                   "[DETMainTest][BindingTests]")
   {
     arma::mat trainingData;
     if (!data::Load("iris.csv", trainingData))
       FAIL("Unable to load dataset iris.csv!");
   
     arma::mat testData;
     if (!data::Load("iris_test.csv", testData))
       FAIL("Unable to load dataset iris_test.csv!");
   
     // Input data.
     SetInputParam("training", trainingData);
     SetInputParam("test", testData);
   
     mlpackMain();
   
     arma::mat trainingSetEstimates =
         IO::GetParam<arma::mat>("training_set_estimates");
     arma::mat testSetEstimates = IO::GetParam<arma::mat>("test_set_estimates");
   
     bindings::tests::CleanMemory();
   
     // Train model using max_leaf_size equals to 40.
   
     SetInputParam("training", std::move(trainingData));
     SetInputParam("test", std::move(testData));
     SetInputParam("max_leaf_size", (int) 40);
   
     mlpackMain();
   
     // Check that initial estimates and final estimates using two models are
     // different.
     REQUIRE(arma::accu(trainingSetEstimates ==
         IO::GetParam<arma::mat>("training_set_estimates")) <
         trainingSetEstimates.n_elem);
   
     REQUIRE(arma::accu(testSetEstimates ==
         IO::GetParam<arma::mat>("test_set_estimates")) <
         testSetEstimates.n_elem);
   }
   
   TEST_CASE_METHOD(DETTestFixture, "DETDiffFoldsTest",
                   "[DETMainTest][BindingTests]")
   {
     arma::mat trainingData;
     if (!data::Load("iris.csv", trainingData))
       FAIL("Unable to load dataset iris.csv!");
   
     arma::mat testData;
     if (!data::Load("iris_test.csv", testData))
       FAIL("Unable to load dataset iris_test.csv!");
   
     // Input data.
     SetInputParam("training", trainingData);
     SetInputParam("test", testData);
   
     mlpackMain();
   
     arma::mat trainingSetEstimates =
         IO::GetParam<arma::mat>("training_set_estimates");
     arma::mat testSetEstimates = IO::GetParam<arma::mat>("test_set_estimates");
   
     bindings::tests::CleanMemory();
   
     // Train model using folds equals to 20.
   
     SetInputParam("training", std::move(trainingData));
     SetInputParam("test", std::move(testData));
     SetInputParam("folds", (int) 20);
   
     mlpackMain();
   
     // Check that initial estimates and final estimates using two models are
     // different.
     REQUIRE(arma::accu(trainingSetEstimates ==
         IO::GetParam<arma::mat>("training_set_estimates")) <
         trainingSetEstimates.n_elem);
   
     REQUIRE(arma::accu(testSetEstimates ==
         IO::GetParam<arma::mat>("test_set_estimates")) <
         testSetEstimates.n_elem);
   }
   
   TEST_CASE_METHOD(DETTestFixture, "DETSkipPruningTest",
                   "[DETMainTest][BindingTests]")
   {
     arma::mat trainingData;
     if (!data::Load("iris.csv", trainingData))
       FAIL("Unable to load dataset iris.csv!");
   
     arma::mat testData;
     if (!data::Load("iris_test.csv", testData))
       FAIL("Unable to load dataset iris_test.csv!");
   
     // Input data.
     SetInputParam("training", trainingData);
     SetInputParam("test", testData);
   
     mlpackMain();
   
     arma::mat trainingSetEstimates =
         IO::GetParam<arma::mat>("training_set_estimates");
     arma::mat testSetEstimates = IO::GetParam<arma::mat>("test_set_estimates");
   
     bindings::tests::CleanMemory();
   
     // Train model by bypassing pruning process.
   
     SetInputParam("training", std::move(trainingData));
     SetInputParam("test", std::move(testData));
     SetInputParam("skip_pruning", (bool) true);
   
     mlpackMain();
   
     // Check that initial estimates and final estimates using two models are
     // different.
     REQUIRE(arma::accu(trainingSetEstimates ==
         IO::GetParam<arma::mat>("training_set_estimates")) <
         trainingSetEstimates.n_elem);
   
     REQUIRE(arma::accu(testSetEstimates ==
         IO::GetParam<arma::mat>("test_set_estimates")) <
         testSetEstimates.n_elem);
   }
