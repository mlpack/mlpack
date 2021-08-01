
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_main_tests_preprocess_binarize_test.cpp:

Program Listing for File preprocess_binarize_test.cpp
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_main_tests_preprocess_binarize_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/main_tests/preprocess_binarize_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #define BINDING_TYPE BINDING_TYPE_TEST
   
   #include <mlpack/core.hpp>
   static const std::string testName = "PreprocessBinarize";
   
   #include <mlpack/core/util/mlpack_main.hpp>
   #include <mlpack/methods/preprocess/preprocess_binarize_main.cpp>
   
   #include "test_helper.hpp"
   #include "../test_catch_tools.hpp"
   #include "../catch.hpp"
   
   using namespace mlpack;
   
   struct PreprocessBinarizeTestFixture
   {
    public:
     PreprocessBinarizeTestFixture()
     {
       // Cache in the options for this program.
       IO::RestoreSettings(testName);
     }
   
     ~PreprocessBinarizeTestFixture()
     {
       // Clear the settings.
       bindings::tests::CleanMemory();
       IO::ClearSettings();
     }
   };
   
   TEST_CASE_METHOD(
       PreprocessBinarizeTestFixture, "PreprocessBinarizeDimensionTest",
       "[PreprocessBinarizeMainTest][BindingTests]")
   {
     // Create a synthetic dataset.
     arma::mat inputData = arma::randu<arma::mat>(2, 5);
   
     // Store size of input dataset.
     size_t inputSize  = inputData.n_cols;
   
     // Input custom data and labels.
     SetInputParam("input", std::move(inputData));
     SetInputParam("threshold", (double) 0.5);
     SetInputParam("dimension", (int) 1);
   
     mlpackMain();
   
     // Now check that the output has desired dimensions.
     REQUIRE(IO::GetParam<arma::mat>("output").n_rows == 2);
     REQUIRE(IO::GetParam<arma::mat>("output").n_cols == inputSize);
   }
   
   TEST_CASE_METHOD(
       PreprocessBinarizeTestFixture, "PreprocessBinarizeNegativeDimensionTest",
       "[PreprocessBinarizeMainTest][BindingTests]")
   {
     arma::mat inputData = arma::randu<arma::mat>(2, 2);
   
     SetInputParam("input", std::move(inputData));
     SetInputParam("threshold", (double) 0.5);
     SetInputParam("dimension", (int) -2); // Invalid.
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(
       PreprocessBinarizeTestFixture, "PreprocessBinarizelargerDimensionTest",
       "[PreprocessBinarizeMainTest][BindingTests]")
   {
     arma::mat inputData = arma::randu<arma::mat>(2, 2);
   
     SetInputParam("input", std::move(inputData));
     SetInputParam("threshold", (double) 0.5);
     SetInputParam("dimension", (int) 6); // Invalid.
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(
       PreprocessBinarizeTestFixture, "PreprocessBinarizeVerificationTest",
       "[PreprocessBinarizeMainTest][BindingTests]")
   {
     arma::mat inputData({{7.0, 4.0, 5.0}, {2.0, 5.0, 9.0}, {7.0, 3.0, 8.0}});
   
     SetInputParam("input", std::move(inputData));
     SetInputParam("threshold", (double) 5.0);
     SetInputParam("dimension", (int) 1);
   
     mlpackMain();
   
     arma::mat output;
     output = std::move(IO::GetParam<arma::mat>("output"));
   
     // All values dimension should remain unchanged.
     REQUIRE(output(0, 0) == Approx(7.0).epsilon(1e-7));
     REQUIRE(output(0, 1) == Approx(4.0).epsilon(1e-7));
     REQUIRE(output(0, 2) == Approx(5.0).epsilon(1e-7));
   
     // All values should be binarized according to the threshold.
     REQUIRE(output(1, 0) == Approx(0.0).margin(1e-5));
     REQUIRE(output(1, 1) == Approx(0.0).margin(1e-5));
     REQUIRE(output(1, 2) == Approx(1.0).epsilon(1e-7));
   
     // All values dimension should remain unchanged.
     REQUIRE(output(2, 0) == Approx(7.0).epsilon(1e-7));
     REQUIRE(output(2, 1) == Approx(3.0).epsilon(1e-7));
     REQUIRE(output(2, 2) == Approx(8.0).epsilon(1e-7));
   }
   
   TEST_CASE_METHOD(
       PreprocessBinarizeTestFixture, "PreprocessBinarizeDimensionLessVerTest",
       "[PreprocessBinarizeMainTest][BindingTests]")
   {
     arma::mat inputData({{7.0, 4.0, 5.0}, {2.0, 5.0, 9.0}, {7.0, 3.0, 8.0}});
   
     SetInputParam("input", std::move(inputData));
     SetInputParam("threshold", (double) 5.0);
   
     mlpackMain();
   
     arma::mat output;
     output = std::move(IO::GetParam<arma::mat>("output"));
   
     // All values should be binarized according to the threshold.
     REQUIRE(output(0, 0) == Approx(1.0).epsilon(1e-7));
     REQUIRE(output(0, 1) == Approx(0.0).margin(1e-5));
     REQUIRE(output(0, 2) == Approx(0.0).margin(1e-5));
     REQUIRE(output(1, 0) == Approx(0.0).margin(1e-5));
     REQUIRE(output(1, 1) == Approx(0.0).margin(1e-5));
     REQUIRE(output(1, 2) == Approx(1.0).epsilon(1e-7));
     REQUIRE(output(2, 0) == Approx(1.0).epsilon(1e-7));
     REQUIRE(output(2, 1) == Approx(0.0).margin(1e-5));
     REQUIRE(output(2, 2) == Approx(1.0).epsilon(1e-7));
   }
