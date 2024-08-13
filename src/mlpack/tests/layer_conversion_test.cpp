/**
 * @file quantization_test.cpp
 * @author Mark Fischinger
 * @brief Tests for the quantization strategies using Boost.
 */

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/test/unit_test.hpp>
#include <limits>
#include <cmath>
#include <algorithm>

using namespace boost::numeric::ublas;

const double TOLERANCE = 1;
const short MAX_SHORT = std::numeric_limits<short>::max();
const short MIN_SHORT = std::numeric_limits<short>::min();

BOOST_AUTO_TEST_SUITE(QuantizationTest)

BOOST_AUTO_TEST_CASE(SimpleQuantizationTest)
{
  matrix<double> sourceWeights(2, 3);
  sourceWeights(0, 0) = -1.5; sourceWeights(0, 1) = 0.5; sourceWeights(0, 2) = 1.0;
  sourceWeights(1, 0) = -0.5; sourceWeights(1, 1) = 2.0; sourceWeights(1, 2) = -2.5;

  matrix<short> targetWeights(2, 3);

  // Calculate the scale factor and offset (assuming a simple linear quantization approach)
  double minVal = sourceWeights.data().begin() != sourceWeights.data().end() 
                  ? *std::min_element(sourceWeights.data().begin(), sourceWeights.data().end())
                  : 0;
  double maxVal = sourceWeights.data().begin() != sourceWeights.data().end() 
                  ? *std::max_element(sourceWeights.data().begin(), sourceWeights.data().end())
                  : 0;

  double scaleFactor = (maxVal - minVal) / (MAX_SHORT - MIN_SHORT);
  double offset = minVal;

  // Apply the quantization
  for (unsigned i = 0; i < sourceWeights.size1(); ++i)
  {
    for (unsigned j = 0; j < sourceWeights.size2(); ++j)
    {
      double scaled = (sourceWeights(i, j) - offset) / scaleFactor;
      scaled = std::max(std::min(scaled, static_cast<double>(MAX_SHORT)), static_cast<double>(MIN_SHORT));
      targetWeights(i, j) = static_cast<short>(scaled);
    }
  }

  // Check the dimensions
  BOOST_REQUIRE_EQUAL(targetWeights.size1(), sourceWeights.size1());
  BOOST_REQUIRE_EQUAL(targetWeights.size2(), sourceWeights.size2());

  // Function to check if the quantization is correct within a small tolerance
  auto check_quantization = [scaleFactor, offset](double source, short target) {
    double expected = (source - offset) / scaleFactor;
    expected = std::max(std::min(expected, static_cast<double>(MAX_SHORT)), static_cast<double>(MIN_SHORT));
    return std::abs(expected - target) <= TOLERANCE;
  };

  // Check that values are properly quantized
  for (unsigned i = 0; i < sourceWeights.size1(); ++i)
  {
    for (unsigned j = 0; j < sourceWeights.size2(); ++j)
    {
      BOOST_TEST_MESSAGE("Quantization mismatch at (" << i << ", " << j << ")");
      BOOST_TEST_MESSAGE("Source: " << sourceWeights(i, j) << ", Target: " << targetWeights(i, j));
      BOOST_TEST_MESSAGE("Expected: " << ((sourceWeights(i, j) - offset) / scaleFactor));
      BOOST_REQUIRE(check_quantization(sourceWeights(i, j), targetWeights(i, j)));
    }
  }

  // Check that the range of values is correct
  BOOST_REQUIRE(targetWeights.data().begin() != targetWeights.data().end() 
                ? *std::min_element(targetWeights.data().begin(), targetWeights.data().end()) >= MIN_SHORT
                : true);
  BOOST_REQUIRE(targetWeights.data().begin() != targetWeights.data().end() 
                ? *std::max_element(targetWeights.data().begin(), targetWeights.data().end()) <= MAX_SHORT
                : true);
}

BOOST_AUTO_TEST_SUITE_END()