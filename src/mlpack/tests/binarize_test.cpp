/**
 * @file tests/binarize_test.cpp
 * @author Keon Kim
 *
 * Test the Binarize method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/binarize.hpp>
#include <mlpack/core/math/random.hpp>

#include "test_catch_tools.hpp"
#include "catch.hpp"

using namespace mlpack;
using namespace arma;
using namespace mlpack::data;

TEST_CASE("BinarizeOneDimension", "[BinarizeTest]")
{
  mat input;
  input << 1 << 2 << 3 << endr
        << 4 << 5 << 6 << endr // this row will be tested
        << 7 << 8 << 9;

  mat output;
  const double threshold = 5.0;
  const size_t dimension = 1;
  Binarize<double>(input, output, threshold, dimension);

  REQUIRE(output(0, 0) == Approx(1.0).epsilon(1e-7)); // 1
  REQUIRE(output(0, 1) == Approx(2.0).epsilon(1e-7)); // 2
  REQUIRE(output(0, 2) == Approx(3.0).epsilon(1e-7)); // 3
  REQUIRE(output(1, 0) == Approx(0.0).margin(1e-5)); // 4 target
  REQUIRE(output(1, 1) == Approx(0.0).margin(1e-5)); // 5 target
  REQUIRE(output(1, 2) == Approx(1.0).epsilon(1e-7)); // 6 target
  REQUIRE(output(2, 0) == Approx(7.0).epsilon(1e-7)); // 7
  REQUIRE(output(2, 1) == Approx(8.0).epsilon(1e-7)); // 8
  REQUIRE(output(2, 2) == Approx(9.0).epsilon(1e-7)); // 9
}

TEST_CASE("BinerizeAll", "[BinarizeTest]")
{
  mat input;
  input << 1 << 2 << 3 << endr
        << 4 << 5 << 6 << endr // this row will be tested
        << 7 << 8 << 9;

  mat output;
  const double threshold = 5.0;

  Binarize<double>(input, output, threshold);

  REQUIRE(output(0, 0) == Approx(0.0).margin(1e-5)); // 1
  REQUIRE(output(0, 1) == Approx(0.0).margin(1e-5)); // 2
  REQUIRE(output(0, 2) == Approx(0.0).margin(1e-5)); // 3
  REQUIRE(output(1, 0) == Approx(0.0).margin(1e-5)); // 4
  REQUIRE(output(1, 1) == Approx(0.0).margin(1e-5)); // 5
  REQUIRE(output(1, 2) == Approx(1.0).epsilon(1e-7)); // 6
  REQUIRE(output(2, 0) == Approx(1.0).epsilon(1e-7)); // 7
  REQUIRE(output(2, 1) == Approx(1.0).epsilon(1e-7)); // 8
  REQUIRE(output(2, 2) == Approx(1.0).epsilon(1e-7)); // 9
}
