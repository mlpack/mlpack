
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_facilities_test.cpp:

Program Listing for File facilities_test.cpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_facilities_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/facilities_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   #include <mlpack/core/cv/metrics/facilities.hpp>
   #include <mlpack/core/metrics/lmetric.hpp>
   #include <mlpack/core/data/load.hpp>
   
   #include "catch.hpp"
   
   using namespace mlpack;
   using namespace mlpack::cv;
   
   TEST_CASE("PairwiseDistanceTest", "[FacilitiesTest]")
   {
     arma::mat X;
     X = { { 0, 1, 1, 0, 0 },
           { 0, 1, 2, 0, 0 },
           { 1, 1, 3, 2, 0 } };
     metric::EuclideanDistance metric;
     arma::mat dist = PairwiseDistances(X, metric);
     REQUIRE(dist(0, 0) == 0);
     REQUIRE(dist(1, 0) == Approx(1.41421).epsilon(1e-5));
     REQUIRE(dist(2, 0) == 3);
   }
