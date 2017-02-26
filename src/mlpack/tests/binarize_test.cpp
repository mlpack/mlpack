/**
 * @file binarize_test.cpp
 * @author Keon Kim
 *
 * Test the Binarzie method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/binarize.hpp>
#include <mlpack/core/math/random.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace arma;
using namespace mlpack::data;

BOOST_AUTO_TEST_SUITE(BinarizeTest);

BOOST_AUTO_TEST_CASE(BinerizeOneDimension)
{
  mat input;
  input << 1 << 2 << 3 << endr
        << 4 << 5 << 6 << endr // this row will be tested
        << 7 << 8 << 9;

  mat output;
  const double threshold = 5.0;
  const size_t dimension = 1;
  Binarize<double>(input, output, threshold, dimension);

  BOOST_REQUIRE_CLOSE(output(0, 0), 1, 1e-5); // 1
  BOOST_REQUIRE_CLOSE(output(0, 1), 2, 1e-5); // 2
  BOOST_REQUIRE_CLOSE(output(0, 2), 3, 1e-5); // 3
  BOOST_REQUIRE_SMALL(output(1, 0), 1e-5); // 4 target
  BOOST_REQUIRE_SMALL(output(1, 1), 1e-5); // 5 target
  BOOST_REQUIRE_CLOSE(output(1, 2), 1, 1e-5); // 6 target
  BOOST_REQUIRE_CLOSE(output(2, 0), 7, 1e-5); // 7
  BOOST_REQUIRE_CLOSE(output(2, 1), 8, 1e-5); // 8
  BOOST_REQUIRE_CLOSE(output(2, 2), 9, 1e-5); // 9
}

BOOST_AUTO_TEST_CASE(BinerizeAll)
{
  mat input;
  input << 1 << 2 << 3 << endr
        << 4 << 5 << 6 << endr // this row will be tested
        << 7 << 8 << 9;

  mat output;
  const double threshold = 5.0;

  Binarize<double>(input, output, threshold);

  BOOST_REQUIRE_SMALL(output(0, 0), 1e-5); // 1
  BOOST_REQUIRE_SMALL(output(0, 1), 1e-5); // 2
  BOOST_REQUIRE_SMALL(output(0, 2), 1e-5); // 3
  BOOST_REQUIRE_SMALL(output(1, 0), 1e-5); // 4
  BOOST_REQUIRE_SMALL(output(1, 1), 1e-5); // 5
  BOOST_REQUIRE_CLOSE(output(1, 2), 1.0, 1e-5); // 6
  BOOST_REQUIRE_CLOSE(output(2, 0), 1.0, 1e-5); // 7
  BOOST_REQUIRE_CLOSE(output(2, 1), 1.0, 1e-5); // 8
  BOOST_REQUIRE_CLOSE(output(2, 2), 1.0, 1e-5); // 9
}

BOOST_AUTO_TEST_SUITE_END();
