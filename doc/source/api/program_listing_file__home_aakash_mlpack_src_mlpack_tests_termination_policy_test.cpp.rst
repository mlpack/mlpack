
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_termination_policy_test.cpp:

Program Listing for File termination_policy_test.cpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_termination_policy_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/termination_policy_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   #include <mlpack/methods/amf/amf.hpp>
   #include <mlpack/methods/amf/termination_policies/max_iteration_termination.hpp>
   #include <mlpack/methods/amf/update_rules/nmf_mult_div.hpp>
   
   #include "catch.hpp"
   
   using namespace std;
   using namespace arma;
   using namespace mlpack;
   using namespace mlpack::amf;
   
   TEST_CASE("MaxIterationTerminationTest", "[TerminationPolicyTest]")
   {
     MaxIterationTermination mit(500);
   
     arma::mat x; // Just an argument to pass.
     for (size_t i = 0; i < 499; ++i)
       REQUIRE(mit.IsConverged(x, x) == false);
   
     // Should keep returning true once maximum iterations are reached.
     REQUIRE(mit.IsConverged(x, x) == true);
     REQUIRE(mit.Iteration() == 500);
     REQUIRE(mit.IsConverged(x, x) == true);
     REQUIRE(mit.IsConverged(x, x) == true);
   }
   
   TEST_CASE("AMFMaxIterationTerminationTest", "[TerminationPolicyTest]")
   {
     mat w = randu<mat>(20, 12);
     mat h = randu<mat>(12, 20);
     mat v = w * h;
     size_t r = 12;
   
     // Custom tighter tolerance.
     MaxIterationTermination mit(10); // Only 10 iterations.
     AMF<MaxIterationTermination,
         RandomInitialization,
         NMFMultiplicativeDivergenceUpdate> nmf(mit);
     nmf.Apply(v, r, w, h);
   
     REQUIRE(nmf.TerminationPolicy().Iteration() == 10);
   }
