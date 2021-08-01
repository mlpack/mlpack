
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_size_checks_test.cpp:

Program Listing for File size_checks_test.cpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_size_checks_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/size_checks_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/prereqs.hpp>
   #include "catch.hpp"
   
   using namespace mlpack;
   using namespace mlpack::util;
   
   TEST_CASE("CheckSizeTest", "[SizeCheckTest]")
   {
     arma::mat data = arma::randu<arma::mat>(20, 30);
     arma::colvec firstLabels = arma::randu<arma::colvec>(20);
     arma::colvec secondLabels = arma::randu<arma::colvec>(30);
     arma::mat thirdLabels = arma::randu<arma::mat>(40, 30);
   
     REQUIRE_THROWS_AS(CheckSameSizes(data, firstLabels, "TestChecking"),
         std::invalid_argument);
     REQUIRE_THROWS_AS(CheckSameSizes(data, (size_t) 20, "TestChecking"),
         std::invalid_argument);
   
     REQUIRE_NOTHROW(CheckSameSizes(data, secondLabels, "TestChecking"));
     REQUIRE_NOTHROW(CheckSameSizes(data, (size_t) 30, "TestChecking"));
     REQUIRE_NOTHROW(CheckSameSizes(data, (size_t) thirdLabels.n_cols,
         "TestChecking"));
   }
   
   TEST_CASE("CheckDimensionality", "[SizeCheckTest]")
   {
     arma::mat dataset = arma::randu<arma::mat>(20, 30);
     arma::colvec refSet = arma::randu<arma::colvec>(20);
     arma::colvec refSet2 = arma::randu<arma::colvec>(40);
   
     REQUIRE_NOTHROW(CheckSameDimensionality(dataset, (size_t) 20,
         "TestingDim"));
     REQUIRE_THROWS_AS(CheckSameDimensionality(dataset, (size_t) 100,
         "TestingDim"), std::invalid_argument);
   
     REQUIRE_THROWS_AS(CheckSameDimensionality(dataset, refSet2, "TestingDim"),
         std::invalid_argument);
     REQUIRE_NOTHROW(CheckSameDimensionality(dataset, refSet,
         "TestingDim"));
   }
