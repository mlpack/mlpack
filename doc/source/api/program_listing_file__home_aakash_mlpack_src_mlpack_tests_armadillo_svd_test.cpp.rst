
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_armadillo_svd_test.cpp:

Program Listing for File armadillo_svd_test.cpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_armadillo_svd_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/armadillo_svd_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   #include <mlpack/methods/cf/svd_wrapper.hpp>
   
   #include "catch.hpp"
   
   using namespace std;
   using namespace mlpack;
   using namespace mlpack::cf;
   using namespace arma;
   
   TEST_CASE("ArmadilloSVDNormalFactorizationTest", "[ArmadilloSVDTest]")
   {
     mat test = randu<mat>(20, 20);
   
     SVDWrapper<> svd;
     arma::mat W, H, sigma;
     double result = svd.Apply(test, W, sigma, H);
   
     REQUIRE(result < 0.01);
   
     test = randu<mat>(50, 50);
     result = svd.Apply(test, W, sigma, H);
   
     REQUIRE(result < 0.01);
   }
   
   TEST_CASE("ArmadilloSVDLowRankFactorizationTest", "[ArmadilloSVDTest]")
   {
     mat W_t = randu<mat>(30, 3);
     mat H_t = randu<mat>(3, 40);
   
     // create a row-rank matrix
     mat test = W_t * H_t;
   
     SVDWrapper<> svd;
     arma::mat W, H;
     double result = svd.Apply(test, 3, W, H);
   
     REQUIRE(result < 0.01);
   }
