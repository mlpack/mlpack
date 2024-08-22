/**
 * @file quantization_test.cpp
 * @author Mark Fischinger
 *
 * Tests for the quantization strategies.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/quantization/quantization_strategy.hpp>
#include <limits>
#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::ann;

const double TOLERANCE = 1; 
const short MAX_SHORT = std::numeric_limits<short>::max();
const short MIN_SHORT = std::numeric_limits<short>::min();

TEST_CASE("SimpleQuantizationTest", "[QuantizationTest]")
{
  arma::mat sourceWeights = {{-1.5, 0.5, 1.0},
                             {-0.5, 2.0, -2.5}};

  arma::Mat<short> targetWeights;

  // Create and apply the simple quantization strategy
  SimpleQuantization<arma::mat, arma::Mat<short>> quantizer;
  quantizer.QuantizeWeights(sourceWeights, targetWeights);

  REQUIRE(targetWeights.n_rows == sourceWeights.n_rows);
  REQUIRE(targetWeights.n_cols == sourceWeights.n_cols);

  double scaleFactor = quantizer.GetScaleFactor();
  double offset = quantizer.GetOffset();

  auto check_quantization = [scaleFactor, offset](double source, short target) {
    double expected = (source - offset) * scaleFactor;
    expected = std::max(std::min(expected, static_cast<double>(MAX_SHORT)), static_cast<double>(MIN_SHORT));
    return std::abs(expected - target) <= TOLERANCE; 
  };

  // Check that values are properly quantized
  for (size_t i = 0; i < sourceWeights.n_elem; ++i)
  {
    INFO("Quantization mismatch at index " << i);
    INFO("Source: " << sourceWeights(i) << ", Target: " << targetWeights(i));
    INFO("Expected: " << ((sourceWeights(i) - offset) * scaleFactor));
    REQUIRE(check_quantization(sourceWeights(i), targetWeights(i)));
  }

  // Check that the range of values is correct
  REQUIRE(targetWeights.min() >= MIN_SHORT);
  REQUIRE(targetWeights.max() <= MAX_SHORT);
}

TEST_CASE("QuantizationStrategyInterfaceTest", "[QuantizationTest]")
{
  std::unique_ptr<QuantizationStrategy<arma::mat, arma::Mat<short>>> quantizer =
      std::make_unique<SimpleQuantization<arma::mat, arma::Mat<short>>>();

  arma::mat sourceWeights = {{-1.0, 1.0},
                             {0.5, -0.5}};

  FFN<..., arma::imat> quantizedNetwork = quantizer->Quantize(network);

  REQUIRE(targetWeights.n_rows == sourceWeights.n_rows);
  REQUIRE(targetWeights.n_cols == sourceWeights.n_cols);

  // Check that the absolute maximum is within the short range
  REQUIRE(arma::abs(targetWeights).max() <= MAX_SHORT);
}
