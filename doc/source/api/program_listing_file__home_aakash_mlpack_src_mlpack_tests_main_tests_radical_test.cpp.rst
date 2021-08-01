
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_main_tests_radical_test.cpp:

Program Listing for File radical_test.cpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_main_tests_radical_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/main_tests/radical_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <string>
   
   #define BINDING_TYPE BINDING_TYPE_TEST
   static const std::string testName = "Radical";
   
   #include <mlpack/core.hpp>
   #include <mlpack/core/util/mlpack_main.hpp>
   #include "test_helper.hpp"
   #include <mlpack/methods/radical/radical_main.cpp>
   
   #include "../catch.hpp"
   
   using namespace mlpack;
   
   struct RadicalTestFixture
   {
    public:
     RadicalTestFixture()
     {
       // Cache in the options for this program.
       IO::RestoreSettings(testName);
     }
   
     ~RadicalTestFixture()
     {
       // Clear the settings.
       bindings::tests::CleanMemory();
       IO::ClearSettings();
     }
   };
   
   TEST_CASE_METHOD(RadicalTestFixture, "RadicalOutputDimensionTest",
                   "[RadicalMainTest][BindingTests]")
   {
     arma::mat input = arma::randu<arma::mat>(5, 3);
   
     SetInputParam("input", std::move(input));
   
     mlpackMain();
   
     // Check dimension of Y matrix.
     REQUIRE(IO::GetParam<arma::mat>("output_ic").n_rows == 5);
     REQUIRE(IO::GetParam<arma::mat>("output_ic").n_cols == 3);
   
     // Check dimension of W matrix.
     REQUIRE(IO::GetParam<arma::mat>("output_unmixing").n_rows == 5);
     REQUIRE(IO::GetParam<arma::mat>("output_unmixing").n_cols == 5);
   }
   
   TEST_CASE_METHOD(RadicalTestFixture, "RadicalBoundsTest",
                   "[RadicalMainTest][BindingTests]")
   {
     arma::mat input = arma::randu<arma::mat>(5, 3);
   
     // Test for replicates.
   
     SetInputParam("input", input);
     SetInputParam("replicates", (int) 0);
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   
     bindings::tests::CleanMemory();
   
     // Test for noise_std_dev.
   
     SetInputParam("input", input);
     SetInputParam("noise_std_dev", (double) -1.0);
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   
     bindings::tests::CleanMemory();
   
     // Test for angles.
   
     SetInputParam("input", input);
     SetInputParam("angles", (int) 0);
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   
     bindings::tests::CleanMemory();
   
     // Test for sweeps.
   
     SetInputParam("input", input);
     SetInputParam("sweeps", (int) -2);
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(RadicalTestFixture, "RadicalDiffNoiseStdDevTest",
                   "[RadicalMainTest][BindingTests]")
   {
     arma::mat input("0.497369 0.891621 0.565789;"
                     "0.33821 0.494571 0.491079;"
                     "0.424898 0.297599 0.475061;"
                     "0.285009 0.152635 0.878107;"
                     "0.321474 0.997979 0.42137");
   
     SetInputParam("input", input);
   
     mlpack::math::FixedRandomSeed();
     mlpackMain();
   
     arma::mat Y = IO::GetParam<arma::mat>("output_ic");
   
     bindings::tests::CleanMemory();
   
     SetInputParam("input", std::move(input));
     SetInputParam("noise_std_dev", (double) 0.01);
   
     mlpack::math::FixedRandomSeed();
     mlpackMain();
   
     // Check that initial output and final output using two models are different.
     REQUIRE(arma::accu(Y == IO::GetParam<arma::mat>("output_ic")) < Y.n_elem);
   }
   
   TEST_CASE_METHOD(RadicalTestFixture, "RadicalDiffReplicatesTest",
                   "[RadicalMainTest][BindingTests]")
   {
     arma::mat input("0.497369 0.891621 0.565789;"
                     "0.33821 0.494571 0.491079;"
                     "0.424898 0.297599 0.475061;"
                     "0.285009 0.152635 0.878107;"
                     "0.321474 0.997979 0.42137");
   
     SetInputParam("input", input);
   
     mlpack::math::FixedRandomSeed();
     mlpackMain();
   
     arma::mat Y = IO::GetParam<arma::mat>("output_ic");
   
     bindings::tests::CleanMemory();
   
     SetInputParam("input", std::move(input));
     SetInputParam("replicates", (int) 10);
   
     mlpack::math::FixedRandomSeed();
     mlpackMain();
   
     // Check that initial output and final output using two models are different.
     REQUIRE(arma::accu(Y == IO::GetParam<arma::mat>("output_ic")) < Y.n_elem);
   }
   
   TEST_CASE_METHOD(RadicalTestFixture, "RadicalDiffAnglesTest",
                   "[RadicalMainTest][BindingTests]")
   {
     arma::mat input("0.497369 0.891621 0.565789;"
                     "0.33821 0.494571 0.491079;"
                     "0.424898 0.297599 0.475061;"
                     "0.285009 0.152635 0.878107;"
                     "0.321474 0.997979 0.42137");
   
     SetInputParam("input", input);
   
     mlpack::math::FixedRandomSeed();
     mlpackMain();
   
     arma::mat Y = IO::GetParam<arma::mat>("output_ic");
   
     bindings::tests::CleanMemory();
   
     SetInputParam("input", std::move(input));
     SetInputParam("angles", (int) 20);
   
     mlpack::math::FixedRandomSeed();
     mlpackMain();
   
     // Check that initial output and final output using two models are different.
     REQUIRE(arma::accu(Y == IO::GetParam<arma::mat>("output_ic")) < Y.n_elem);
   }
   
   TEST_CASE_METHOD(RadicalTestFixture, "RadicalDiffSweepsTest",
                   "[RadicalMainTest][BindingTests]")
   {
     arma::mat input("0.497369 0.891621 0.565789;"
                     "0.33821 0.494571 0.491079;"
                     "0.424898 0.297599 0.475061;"
                     "0.285009 0.152635 0.878107;"
                     "0.321474 0.997979 0.42137");
   
     SetInputParam("input", input);
   
     mlpack::math::FixedRandomSeed();
     mlpackMain();
   
     arma::mat Y = IO::GetParam<arma::mat>("output_ic");
   
     bindings::tests::CleanMemory();
   
     SetInputParam("input", std::move(input));
     SetInputParam("sweeps", (int) 2);
   
     mlpack::math::FixedRandomSeed();
     mlpackMain();
   
     // Check that initial output and final output using two models are different.
     REQUIRE(arma::accu(Y == IO::GetParam<arma::mat>("output_ic")) < Y.n_elem);
   }
