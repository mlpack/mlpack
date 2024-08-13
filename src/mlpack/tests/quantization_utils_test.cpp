/**
 * @file quantization_utils_test.cpp
 * @author Your Name
 *
 * Tests for the quantization utility functions.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/quantization/quantization_utils.hpp>

#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::ann;

TEST_CASE("LinearQuantizationTest", "[QuantizationUtils]")
{
  arma::mat input = {{-1.5, 0.0, 1.5},
                     {-0.5, 0.5, 2.5}};
  
  SECTION("8-bit quantization")
  {
    arma::mat quantized = LinearQuantize(input, -2.0, 2.0, 8);
    
    // Check if the output is within the specified range
    REQUIRE(quantized.min() >= -2.0);
    REQUIRE(quantized.max() <= 2.0);
    
    // Check if the relative order of values is preserved
    REQUIRE(quantized(0, 0) < quantized(0, 1));
    REQUIRE(quantized(0, 1) < quantized(0, 2));
    REQUIRE(quantized(1, 0) < quantized(1, 1));
    REQUIRE(quantized(1, 1) < quantized(1, 2));
  }

  SECTION("4-bit quantization")
  {
    arma::mat quantized = LinearQuantize(input, -2.0, 2.0, 4);
    
    // Check if the output is within the specified range
    REQUIRE(quantized.min() >= -2.0);
    REQUIRE(quantized.max() <= 2.0);
    
    // Check if the number of unique values is less than or equal to 2^4
    REQUIRE(arma::unique(quantized).n_elem <= 16);
  }
}

TEST_CASE("ScalingFactorCalculationTest", "[QuantizationUtils]")
{
  arma::mat input = {{-1.5, 0.0, 1.5},
                     {-0.5, 0.5, 2.5}};
  
  SECTION("8-bit scaling factor")
  {
    double scale = CalculateScalingFactor(input, 8);
    REQUIRE(scale == Approx(2.5 / 127.0).epsilon(1e-5));
  }

  SECTION("4-bit scaling factor")
  {
    double scale = CalculateScalingFactor(input, 4);
    REQUIRE(scale == Approx(2.5 / 7.0).epsilon(1e-5));
  }
}

TEST_CASE("ScaleQuantizationTest", "[QuantizationUtils]")
{
  arma::mat input = {{-1.5, 0.0, 1.5},
                     {-0.5, 0.5, 2.5}};
  
  SECTION("8-bit scale quantization")
  {
    arma::mat quantized = ScaleQuantize(input, 8);
    
    // Check if the output has the same shape as the input
    REQUIRE(quantized.n_rows == input.n_rows);
    REQUIRE(quantized.n_cols == input.n_cols);
    
    // Check if the relative order of values is preserved
    REQUIRE(quantized(0, 0) < quantized(0, 1));
    REQUIRE(quantized(0, 1) < quantized(0, 2));
    REQUIRE(quantized(1, 0) < quantized(1, 1));
    REQUIRE(quantized(1, 1) < quantized(1, 2));
    
    // Check if the number of unique values is less than or equal to 2^8
    REQUIRE(arma::unique(quantized).n_elem <= 256);
  }

  SECTION("4-bit scale quantization")
  {
    arma::mat quantized = ScaleQuantize(input, 4);
    
    REQUIRE(quantized.n_rows == input.n_rows);
    REQUIRE(quantized.n_cols == input.n_cols);
    
    // Check if the number of unique values is less than or equal to 2^4
    REQUIRE(arma::unique(quantized).n_elem <= 16);
  }
}