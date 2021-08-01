
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_main_tests_emst_test.cpp:

Program Listing for File emst_test.cpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_main_tests_emst_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/main_tests/emst_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <string>
   
   #define BINDING_TYPE BINDING_TYPE_TEST
   static const std::string testName = "EMST";
   
   #include <mlpack/core.hpp>
   #include <mlpack/core/util/mlpack_main.hpp>
   #include <mlpack/methods/emst/emst_main.cpp>
   #include "test_helper.hpp"
   
   #include "../catch.hpp"
   
   #include <boost/math/special_functions/round.hpp>
   
   using namespace mlpack;
   
   struct EMSTTestFixture
   {
    public:
     EMSTTestFixture()
     {
       // Cache in the options for this program.
       IO::RestoreSettings(testName);
     }
   
     ~EMSTTestFixture()
     {
       // Clear the settings.
       bindings::tests::CleanMemory();
       IO::ClearSettings();
     }
   };
   
   TEST_CASE_METHOD(EMSTTestFixture, "EMSTOutputDimensionTest",
                    "[EMSTMainTest][BindingTests]")
   {
     arma::mat x;
     if (!data::Load("test_data_3_1000.csv", x))
       FAIL("Cannot load test dataset test_data_3_1000.csv!");
   
     // Input random data points.
     SetInputParam("input", std::move(x));
     SetInputParam("leaf_size", (int) 2);
   
     mlpackMain();
   
     // Now check that the output has 3 dimensions.
     REQUIRE(IO::GetParam<arma::mat>("output").n_rows == 3);
     // Check number of output points.
     REQUIRE(IO::GetParam<arma::mat>("output").n_cols == 999);
   }
   
   TEST_CASE_METHOD(EMSTTestFixture, "EMSTNaiveOutputDimensionTest",
                    "[EMSTMainTest][BindingTests]")
   {
     arma::mat x;
     if (!data::Load("test_data_3_1000.csv", x))
       FAIL("Cannot load test dataset test_data_3_1000.csv!");
   
     // Input random data points.
     SetInputParam("input", std::move(x));
     SetInputParam("naive", true);
   
     mlpackMain();
   
     // Now check that the output has 3 dimensions.
     REQUIRE(IO::GetParam<arma::mat>("output").n_rows == 3);
     // Check number of output points.
     REQUIRE(IO::GetParam<arma::mat>("output").n_cols == 999);
   }
   
   TEST_CASE_METHOD(EMSTTestFixture, "EMSTInvalidLeafSizeTest",
                    "[EMSTMainTest][BindingTests]")
   {
     arma::mat x;
     if (!data::Load("test_data_3_1000.csv", x))
       FAIL("Cannot load test dataset test_data_3_1000.csv!");
   
     // Input random data points.
     SetInputParam("input", std::move(x));
     SetInputParam("leaf_size", (int) -1); // Invalid leaf size.
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(EMSTTestFixture, "EMSTFirstTwoOutputRowsIntegerTest",
                    "[EMSTMainTest][BindingTests]")
   {
     arma::mat x;
     if (!data::Load("test_data_3_1000.csv", x))
       FAIL("Cannot load test dataset test_data_3_1000.csv!");
   
     // Input random data points.
     SetInputParam("input", std::move(x));
     SetInputParam("leaf_size", (int) 2);
   
     for (size_t i = 0; i < IO::GetParam<arma::mat>("output").n_cols; ++i)
     {
       REQUIRE(IO::GetParam<arma::mat>("output")(0, i) ==
           Approx(boost::math::iround(IO::GetParam<arma::mat>("output")(0, i))).
           epsilon(1e-7));
       REQUIRE(IO::GetParam<arma::mat>("output")(1, i) ==
           Approx(boost::math::iround(IO::GetParam<arma::mat>("output")(1, i))).
           epsilon(1e-7));
     }
   }
