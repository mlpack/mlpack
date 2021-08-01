
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_sumtree_test.cpp:

Program Listing for File sumtree_test.cpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_sumtree_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/sumtree_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   
   #include <mlpack/methods/reinforcement_learning/replay/sumtree.hpp>
   
   #include "catch.hpp"
   #include "test_catch_tools.hpp"
   
   using namespace mlpack;
   using namespace mlpack::rl;
   
   TEST_CASE("SetElement", "[SumTreeTest]")
   {
     SumTree<double> sumtree(4);
     sumtree.Set(0, 1.0);
     sumtree.Set(1, 0.8);
     sumtree.Set(2, 0.6);
     sumtree.Set(3, 0.4);
   
     CHECK(sumtree.Sum() == Approx(2.8).epsilon(1e-10));
     CHECK(sumtree.Sum(0, 1) == Approx(1.0).epsilon(1e-10));
     CHECK(sumtree.Sum(0, 3) == Approx(2.4).epsilon(1e-10));
     CHECK(sumtree.Sum(1, 4) == Approx(1.8).epsilon(1e-10));
   }
   
   TEST_CASE("GetElement", "[SumTreeTest]")
   {
     SumTree<double> sumtree(4);
     sumtree.Set(0, 1.0);
     sumtree.Set(1, 0.8);
     sumtree.Set(2, 0.6);
     sumtree.Set(3, 0.4);
   
     CHECK(sumtree.Get(0) == Approx(1.0).epsilon(1e-10));
     CHECK(sumtree.Get(1) == Approx(0.8).epsilon(1e-10));
     CHECK(sumtree.Get(2) == Approx(0.6).epsilon(1e-10));
     CHECK(sumtree.Get(3) == Approx(0.4).epsilon(1e-10));
   }
   
   TEST_CASE("FindPrefixSum", "[SumTreeTest]")
   {
     SumTree<double> sumtree(4);
     sumtree.Set(0, 1.0);
     sumtree.Set(1, 0.8);
     sumtree.Set(2, 0.6);
     sumtree.Set(3, 0.4);
   
     CHECK(sumtree.FindPrefixSum(0) <= 0.0);
     CHECK(sumtree.FindPrefixSum(1) <= 1.0);
     CHECK(sumtree.FindPrefixSum(2.8) <= 3.0);
     CHECK(sumtree.FindPrefixSum(3.0) <= 3.0);
   }
   
   TEST_CASE("BatchUpdate", "[SumTreeTest]")
   {
     SumTree<double> sumtree(4);
     arma::ucolvec indices = {0, 1, 2, 3};
     arma::colvec data = {1.0, 0.8, 0.6, 0.4};
   
     sumtree.BatchUpdate(indices, data);
   
     CHECK(sumtree.FindPrefixSum(0) <= 0);
     CHECK(sumtree.FindPrefixSum(1) <= 1);
     CHECK(sumtree.FindPrefixSum(2.8) <= 3);
     CHECK(sumtree.FindPrefixSum(3.0) <= 3);
   }
