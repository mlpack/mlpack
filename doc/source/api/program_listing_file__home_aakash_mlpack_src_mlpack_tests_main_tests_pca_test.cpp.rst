
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_main_tests_pca_test.cpp:

Program Listing for File pca_test.cpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_main_tests_pca_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/main_tests/pca_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <string>
   
   #define BINDING_TYPE BINDING_TYPE_TEST
   static const std::string testName = "PrincipalComponentAnalysis";
   
   #include <mlpack/core.hpp>
   #include <mlpack/core/util/mlpack_main.hpp>
   #include "test_helper.hpp"
   #include <mlpack/methods/pca/pca_main.cpp>
   
   #include "../catch.hpp"
   
   using namespace mlpack;
   
   struct PCATestFixture
   {
    public:
     PCATestFixture()
     {
       // Cache in the options for this program.
       IO::RestoreSettings(testName);
     }
   
     ~PCATestFixture()
     {
       // Clear the settings.
       bindings::tests::CleanMemory();
       IO::ClearSettings();
     }
   };
   
   TEST_CASE_METHOD(PCATestFixture, "PCADimensionTest",
                    "[PCAMainTest][BindingTests]")
   {
     arma::mat x = arma::randu<arma::mat>(5, 5);
   
     // Random input, new dimensionality of 3.
     SetInputParam("input", std::move(x));
     SetInputParam("new_dimensionality", (int) 3);
   
     mlpackMain();
   
     // Now check that the output has 3 dimensions.
     REQUIRE(IO::GetParam<arma::mat>("output").n_rows == 3);
     REQUIRE(IO::GetParam<arma::mat>("output").n_cols == 5);
   }
   
   TEST_CASE_METHOD(PCATestFixture, "PCAVarRetainTest",
                    "[PCAMainTest][BindingTests]")
   {
     arma::mat x = arma::randu<arma::mat>(4, 5);
   
     SetInputParam("input", std::move(x));
     SetInputParam("var_to_retain", (double) 1.0);
     SetInputParam("scale", true);
     SetInputParam("new_dimensionality", (int) 3); // Should be ignored.
   
     mlpackMain();
   
     // Check that the output has 5 dimensions.
     REQUIRE(IO::GetParam<arma::mat>("output").n_rows == 4);
     REQUIRE(IO::GetParam<arma::mat>("output").n_cols == 5);
   }
   
   TEST_CASE_METHOD(PCATestFixture, "PCANoVarRetainTest",
                    "[PCAMainTest][BindingTests]")
   {
     arma::mat x = arma::randu<arma::mat>(5, 5);
   
     SetInputParam("input", std::move(x));
     SetInputParam("var_to_retain", (double) 0.01);
     SetInputParam("scale", true);
     SetInputParam("new_dimensionality", (int) 3); // Should be ignored.
   
     mlpackMain();
   
     // Check that the output has 1 dimensions.
     REQUIRE(IO::GetParam<arma::mat>("output").n_rows == 1);
     REQUIRE(IO::GetParam<arma::mat>("output").n_cols == 5);
   }
   
   TEST_CASE_METHOD(PCATestFixture, "PCATooHighNewDimensionalityTest",
                    "[PCAMainTest][BindingTests]")
   {
     arma::mat x = arma::randu<arma::mat>(5, 5);
   
     SetInputParam("input", std::move(x));
     SetInputParam("new_dimensionality", (int) 7); // Invalid.
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
