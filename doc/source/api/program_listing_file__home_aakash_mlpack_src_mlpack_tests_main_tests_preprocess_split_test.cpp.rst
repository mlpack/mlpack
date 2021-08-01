
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_main_tests_preprocess_split_test.cpp:

Program Listing for File preprocess_split_test.cpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_main_tests_preprocess_split_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/main_tests/preprocess_split_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #define BINDING_TYPE BINDING_TYPE_TEST
   
   #include <mlpack/core.hpp>
   static const std::string testName = "PreprocessSplit";
   
   #include <mlpack/core/util/mlpack_main.hpp>
   #include <mlpack/methods/preprocess/preprocess_split_main.cpp>
   
   #include "test_helper.hpp"
   #include "../test_catch_tools.hpp"
   #include "../catch.hpp"
   
   #include <cmath>
   
   using namespace mlpack;
   
   struct PreprocessSplitTestFixture
   {
    public:
     PreprocessSplitTestFixture()
     {
       // Cache in the options for this program.
       IO::RestoreSettings(testName);
     }
   
     ~PreprocessSplitTestFixture()
     {
       // Clear the settings.
       bindings::tests::CleanMemory();
       IO::ClearSettings();
     }
   };
   
   TEST_CASE_METHOD(PreprocessSplitTestFixture, "PreprocessSplitDimensionTest",
                    "[PreprocessSplitMainTest][BindingTests]")
   {
     // Load custom dataset.
     arma::mat inputData;
     arma::Mat<size_t> labels;
     if (!data::Load("vc2.csv", inputData))
       FAIL("Cannot load train dataset vc2.csv!");
     if (!data::Load("vc2_labels.txt", labels))
       FAIL("Unable to load label dataset vc2_labels.txt!");
   
     // Store size of input dataset.
     int inputSize  = inputData.n_cols;
     int labelSize  = labels.n_cols;
   
     // Input custom data points and labels.
     SetInputParam("input", std::move(inputData));
     SetInputParam("input_labels", std::move(labels));
   
     // Input test_ratio.
     SetInputParam("test_ratio", (double) 0.1);
   
     mlpackMain();
   
     // Now check that the output has desired dimensions.
     REQUIRE(IO::GetParam<arma::mat>("training").n_cols ==
         std::ceil(0.9 * inputSize));
     REQUIRE(IO::GetParam<arma::mat>("test").n_cols ==
         std::floor(0.1 * inputSize));
   
     REQUIRE(
         IO::GetParam<arma::Mat<size_t>>("training_labels").n_cols ==
         std::ceil(0.9 * labelSize));
     REQUIRE(IO::GetParam<arma::Mat<size_t>>("test_labels").n_cols ==
         std::floor(0.1 * labelSize));
   }
   
   TEST_CASE_METHOD(
       PreprocessSplitTestFixture,
       "PreprocessSplitLabelLessDimensionTest",
       "[PreprocessSplitMainTest][BindingTests]")
   {
     // Load custom dataset.
     arma::mat inputData;
     if (!data::Load("vc2.csv", inputData))
       FAIL("Cannot load train dataset vc2.csv!");
   
     // Store size of input dataset.
     int inputSize  = inputData.n_cols;
   
     // Input custom data points and labels.
     SetInputParam("input", std::move(inputData));
   
     // Input test_ratio.
     SetInputParam("test_ratio", (double) 0.1);
   
     mlpackMain();
   
     // Now check that the output has desired dimensions.
     REQUIRE(IO::GetParam<arma::mat>("training").n_cols ==
         std::ceil(0.9 * inputSize));
     REQUIRE(IO::GetParam<arma::mat>("test").n_cols ==
         std::floor(0.1 * inputSize));
   }
   
   TEST_CASE_METHOD(PreprocessSplitTestFixture, "PreprocessSplitTestRatioTest",
                    "[PreprocessSplitMainTest][BindingTests]")
   {
     // Load custom dataset.
     arma::mat inputData;
     arma::Mat<size_t> labels;
     if (!data::Load("vc2.csv", inputData))
       FAIL("Cannot load train dataset vc2.csv!");
     if (!data::Load("vc2_labels.txt", labels))
       FAIL("Unable to load label dataset vc2_labels.txt!");
   
     // Input custom data points and labels.
     SetInputParam("input", std::move(inputData));
     SetInputParam("input_labels", std::move(labels));
   
     SetInputParam("test_ratio", (double) -0.2);
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(
       PreprocessSplitTestFixture, "PreprocessSplitZeroTestRatioTest",
       "[PreprocessSplitMainTest][BindingTests]")
   {
     // Load custom dataset.
     arma::mat inputData;
     arma::Mat<size_t> labels;
     if (!data::Load("vc2.csv", inputData))
       FAIL("Cannot load train dataset vc2.csv!");
     if (!data::Load("vc2_labels.txt", labels))
       FAIL("Unable to load label dataset vc2_labels.txt!");
   
     // Store size of input dataset.
     int inputSize = inputData.n_cols;
     int labelSize = labels.n_cols;
   
     // Input custom data points and labels.
     SetInputParam("input", std::move(inputData));
     SetInputParam("input_labels", std::move(labels));
   
     SetInputParam("test_ratio", (double) 0.0);
   
     mlpackMain();
   
     // Now check that the output has desired dimensions.
     REQUIRE(IO::GetParam<arma::mat>("training").n_cols ==
         (arma::uword) inputSize);
     REQUIRE(IO::GetParam<arma::mat>("test").n_cols == 0);
   
     REQUIRE(IO::GetParam<arma::Mat<size_t>>("training_labels").n_cols ==
         (arma::uword) labelSize);
     REQUIRE(IO::GetParam<arma::Mat<size_t>>("test_labels").n_cols == 0);
   }
   
   TEST_CASE_METHOD(
       PreprocessSplitTestFixture, "PreprocessSplitUnityTestRatioTest",
       "[PreprocessSplitMainTest][BindingTests]")
   {
     // Load custom dataset.
     arma::mat inputData;
     arma::Mat<size_t> labels;
     if (!data::Load("vc2.csv", inputData))
       FAIL("Cannot load train dataset vc2.csv!");
     if (!data::Load("vc2_labels.txt", labels))
       FAIL("Unable to load label dataset vc2_labels.txt!");
   
     // Store size of input dataset.
     int inputSize = inputData.n_cols;
     int labelSize = labels.n_cols;
   
     // Input custom data points and labels.
     SetInputParam("input", std::move(inputData));
     SetInputParam("input_labels", std::move(labels));
   
     SetInputParam("test_ratio", (double) 1.0);
   
     mlpackMain();
   
     // Now check that the output has desired dimensions.
     REQUIRE(IO::GetParam<arma::mat>("training").n_cols == 0);
     REQUIRE(IO::GetParam<arma::mat>("test").n_cols == (arma::uword) inputSize);
   
     REQUIRE(IO::GetParam<arma::Mat<size_t>>("training_labels").n_cols == 0);
     REQUIRE(IO::GetParam<arma::Mat<size_t>>("test_labels").n_cols ==
         (arma::uword) labelSize);
   }
   
   TEST_CASE_METHOD(
       PreprocessSplitTestFixture, "PreprocessSplitLabelShuffleDataTest",
       "[PreprocessSplitMainTest][BindingTests]")
   {
     // Load custom dataset.
     arma::mat inputData;
     if (!data::Load("vc2.csv", inputData))
       FAIL("Cannot load train dataset vc2.csv!");
   
     // Store size of input dataset.
     int inputSize  = inputData.n_cols;
   
     // Input custom data points and labels.
     SetInputParam("input", inputData);
   
     // Input test_ratio.
     SetInputParam("test_ratio", (double) 0.1);
     SetInputParam("no_shuffle", true);
     mlpackMain();
   
     // Now check that the output has desired dimensions.
     REQUIRE(IO::GetParam<arma::mat>("training").n_cols ==
         std::ceil(0.9 * inputSize));
     REQUIRE(IO::GetParam<arma::mat>("test").n_cols ==
         std::floor(0.1 * inputSize));
   
     arma::mat concat = arma::join_rows(IO::GetParam<arma::mat>("training"),
         IO::GetParam<arma::mat>("test"));
     CheckMatrices(inputData, concat);
   }
   
   TEST_CASE_METHOD(
       PreprocessSplitTestFixture, "PreprocessStratifiedSplitZeroTestRatioTest",
       "[PreprocessSplitMainTest][BindingTests]")
   {
     // Load custom dataset.
     arma::mat inputData;
     arma::Mat<size_t> labels;
     if (!data::Load("vc2.csv", inputData))
       FAIL("Cannot load train dataset vc2.csv!");
     if (!data::Load("vc2_labels.txt", labels))
       FAIL("Unable to load label dataset vc2_labels.txt!");
   
     // Store size of input dataset.
     int inputSize = inputData.n_cols;
     int labelSize = labels.n_cols;
   
     // Input custom data points and labels.
     SetInputParam("input", std::move(inputData));
     SetInputParam("input_labels", std::move(labels));
   
     SetInputParam("test_ratio", (double) 0.0);
     SetInputParam("stratify_data", true);
   
     mlpackMain();
   
     // Now check that the output has desired dimensions.
     REQUIRE(IO::GetParam<arma::mat>("training").n_cols ==
         (arma::uword) inputSize);
     REQUIRE(IO::GetParam<arma::mat>("test").n_cols == 0);
   
     REQUIRE(IO::GetParam<arma::Mat<size_t>>("training_labels").n_cols ==
         (arma::uword) labelSize);
     REQUIRE(IO::GetParam<arma::Mat<size_t>>("test_labels").n_cols == 0);
   }
   
   TEST_CASE_METHOD(
       PreprocessSplitTestFixture, "PreprocessStratifiedSplitUnityTestRatioTest",
       "[PreprocessSplitMainTest][BindingTests]")
   {
     // Load custom dataset.
     arma::mat inputData;
     arma::Mat<size_t> labels;
     if (!data::Load("vc2.csv", inputData))
       FAIL("Cannot load train dataset vc2.csv!");
     if (!data::Load("vc2_labels.txt", labels))
       FAIL("Unable to load label dataset vc2_labels.txt!");
   
     // Store size of input dataset.
     int inputSize = inputData.n_cols;
     int labelSize = labels.n_cols;
   
     // Input custom data points and labels.
     SetInputParam("input", std::move(inputData));
     SetInputParam("input_labels", std::move(labels));
   
     SetInputParam("test_ratio", (double) 1.0);
     SetInputParam("stratify_data", true);
   
     mlpackMain();
   
     // Now check that the output has desired dimensions.
     REQUIRE(IO::GetParam<arma::mat>("training").n_cols == 0);
     REQUIRE(IO::GetParam<arma::mat>("test").n_cols == (arma::uword) inputSize);
   
     REQUIRE(IO::GetParam<arma::Mat<size_t>>("training_labels").n_cols == 0);
     REQUIRE(IO::GetParam<arma::Mat<size_t>>("test_labels").n_cols ==
         (arma::uword) labelSize);
   }
   
   TEST_CASE_METHOD(
       PreprocessSplitTestFixture, "PreprocessStratifiedSplitTest",
       "[PreprocessSplitMainTest][BindingTests]")
   {
     // Load custom dataset.
     arma::mat inputData;
     arma::Mat<size_t> labels;
     if (!data::Load("vc2.csv", inputData))
       FAIL("Cannot load train dataset vc2.csv!");
     if (!data::Load("vc2_labels.txt", labels))
       FAIL("Unable to load label dataset vc2_labels.txt!");
   
     // Input custom data points and labels.
     SetInputParam("input", std::move(inputData));
     SetInputParam("input_labels", std::move(labels));
   
     SetInputParam("test_ratio", (double) 0.3);
     SetInputParam("stratify_data", true);
   
     mlpackMain();
   
     // Now check that the output has desired dimensions.
     REQUIRE(IO::GetParam<arma::mat>("training").n_cols == 145);
     REQUIRE(IO::GetParam<arma::mat>("test").n_cols == 62);
   
     // Checking for specific label counts in the output.
     REQUIRE(static_cast<uvec>(find(
         IO::GetParam<arma::Mat<size_t>>("training_labels") == 0)).n_rows == 28);
     REQUIRE(static_cast<uvec>(find(
         IO::GetParam<arma::Mat<size_t>>("training_labels") == 1)).n_rows == 70);
     REQUIRE(static_cast<uvec>(find(
         IO::GetParam<arma::Mat<size_t>>("training_labels") == 2)).n_rows == 47);
   
     REQUIRE(static_cast<uvec>(find(
         IO::GetParam<arma::Mat<size_t>>("test_labels") == 0)).n_rows == 12);
     REQUIRE(static_cast<uvec>(find(
         IO::GetParam<arma::Mat<size_t>>("test_labels") == 1)).n_rows == 30);
     REQUIRE(static_cast<uvec>(find(
         IO::GetParam<arma::Mat<size_t>>("test_labels") == 2)).n_rows == 20);
   }
