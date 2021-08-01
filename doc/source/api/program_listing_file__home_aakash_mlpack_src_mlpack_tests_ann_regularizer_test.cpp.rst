
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_ann_regularizer_test.cpp:

Program Listing for File ann_regularizer_test.cpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_ann_regularizer_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/ann_regularizer_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   
   #include <mlpack/methods/ann/layer/layer.hpp>
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   #include <mlpack/methods/ann/init_rules/random_init.hpp>
   #include <mlpack/methods/ann/regularizer/regularizer.hpp>
   
   #include "catch.hpp"
   #include "ann_test_tools.hpp"
   #include "serialization.hpp"
   
   using namespace mlpack;
   using namespace mlpack::ann;
   
   TEST_CASE("GradientL1RegularizerTest", "[ANNRegularizerTest]")
   {
     // Add function gradient instantiation.
     struct GradientFunction
     {
       GradientFunction() :
         factor(0.6),
         reg(factor)
       {
         // Nothing to do here.
       }
   
       double Output(const arma::mat& weight, size_t i, size_t j)
       {
         return std::abs(weight(i, j)) * factor;
       }
   
       void Gradient(arma::mat& weight, arma::mat& gradient)
       {
         reg.Evaluate(weight, gradient);
       }
   
       double factor;
       L1Regularizer reg;
     } function;
   
     REQUIRE(CheckRegularizerGradient(function) <= 1e-4);
   }
   
   TEST_CASE("GradientL2RegularizerTest", "[ANNRegularizerTest]")
   {
     // Add function gradient instantiation.
     struct GradientFunction
     {
       GradientFunction() :
           factor(0.6),
           reg(factor)
       {
         // Nothing to do here.
       }
   
       double Output(const arma::mat& weight, size_t i, size_t j)
       {
         return weight(i, j) * weight(i, j) * factor;
       }
   
       void Gradient(arma::mat& weight, arma::mat& gradient)
       {
         reg.Evaluate(weight, gradient);
       }
   
       double factor;
       L2Regularizer reg;
     } function;
   
     REQUIRE(CheckRegularizerGradient(function) <= 1e-4);
   }
   
   TEST_CASE("GradientOrthogonalRegularizerTest", "[ANNRegularizerTest]")
   {
     // Add function gradient instantiation.
     struct GradientFunction
     {
       GradientFunction() :
           factor(0.6),
           reg(factor)
       {
         // Nothing to do here.
       }
   
       double Output(const arma::mat& weight, size_t /* i */, size_t /* j */)
       {
         arma::mat x = arma::abs(weight * weight.t() -
             arma::eye<arma::mat>(weight.n_rows, weight.n_cols)) * factor;
         return arma::accu(x);
       }
   
       void Gradient(arma::mat& weight, arma::mat& gradient)
       {
         reg.Evaluate(weight, gradient);
       }
   
       double factor;
       OrthogonalRegularizer reg;
     } function;
   
     REQUIRE(CheckRegularizerGradient(function) <= 1e-4);
   }
