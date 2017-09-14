/**
 * @file line_search_test.cpp
 * @author Chenzhe Diao
 *
 * Test file for line search optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */


#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/line_search/line_search.hpp>
#include <mlpack/core/optimizers/fw/test_func_fw.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(LineSearchTest);

/**
 * Simple test of Line Search with TestFuncFW function.
 */
BOOST_AUTO_TEST_CASE(FuncFWTest)
{
  vec x1 = zeros<vec>(3);
  vec x2;
  x2 << 0.2 << 0.4 << 0.6;

  TestFuncFW f;
  LineSearch s;

  double result = s.Optimize(f, x1, x2);

  BOOST_REQUIRE_SMALL(result, 1e-10);
  BOOST_REQUIRE_SMALL(x2[0] - 0.1, 1e-10);
  BOOST_REQUIRE_SMALL(x2[1] - 0.2, 1e-10);
  BOOST_REQUIRE_SMALL(x2[2] - 0.3, 1e-10);
}


BOOST_AUTO_TEST_SUITE_END();
