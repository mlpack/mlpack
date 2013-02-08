/**
 * @file nmf_test.cpp
 * @author Mohan Rajendran
 *
 * Test file for NMF class.
 *
 * This file is part of MLPACK 1.0.4.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/nmf/nmf.hpp>
#include <mlpack/methods/nmf/random_acol_init.hpp>
#include <mlpack/methods/nmf/mult_div_update_rules.hpp>
#include <mlpack/methods/nmf/als_update_rules.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

BOOST_AUTO_TEST_SUITE(NMFTest);

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::nmf;

/**
 * Check the if the product of the calculated factorization is close to the
 * input matrix. Default case
 */
BOOST_AUTO_TEST_CASE(NMFDefaultTest)
{
  mat w = randu<mat>(20, 16);
  mat h = randu<mat>(16, 20);
  mat v = w * h;
  size_t r = 16;

  NMF<> nmf;
  nmf.Apply(v, r, w, h);

  mat wh = w * h;

  for (size_t row = 0; row < 5; row++)
    for (size_t col = 0; col < 5; col++)
      BOOST_REQUIRE_CLOSE(v(row, col), wh(row, col), 10.0);
}

/**
 * Check the if the product of the calculated factorization is close to the
 * input matrix. Random Acol Initialization Distance Minimization Update
 */
BOOST_AUTO_TEST_CASE(NMFAcolDistTest)
{
  mat w = randu<mat>(20, 16);
  mat h = randu<mat>(16, 20);
  mat v = w * h;
  size_t r = 16;

  NMF<RandomAcolInitialization<> > nmf;
  nmf.Apply(v, r, w, h);

  mat wh = w * h;

  for (size_t row = 0; row < 5; row++)
    for (size_t col = 0; col < 5; col++)
      BOOST_REQUIRE_CLOSE(v(row, col), wh(row, col), 10.0);
}

/**
 * Check the if the product of the calculated factorization is close to the
 * input matrix. Random Initialization Divergence Minimization Update
 */
BOOST_AUTO_TEST_CASE(NMFRandomDivTest)
{
  mat w = randu<mat>(20, 16);
  mat h = randu<mat>(16, 20);
  mat v = w * h;
  size_t r = 16;

  NMF<RandomInitialization,
      WMultiplicativeDivergenceRule,
      HMultiplicativeDivergenceRule> nmf;
  nmf.Apply(v, r, w, h);

  mat wh = w * h;

  for (size_t row = 0; row < 5; row++)
    for (size_t col = 0; col < 5; col++)
      BOOST_REQUIRE_CLOSE(v(row, col), wh(row, col), 10.0);
}

/**
 * Check the if the product of the calculated factorization is close to the
 * input matrix. Random Initialization Alternating Least Squares Update
 */
BOOST_AUTO_TEST_CASE(NMFALSTest)
{
  mat w = randu<mat>(20, 16);
  mat h = randu<mat>(16, 20);
  mat v = w * h;
  size_t r = 16;

  NMF<RandomInitialization,
      WAlternatingLeastSquaresRule,
      HAlternatingLeastSquaresRule> nmf;
  nmf.Apply(v, r, w, h);

  mat wh = w * h;

  for (size_t row = 0; row < 5; row++)
    for (size_t col = 0; col < 5; col++)
      BOOST_REQUIRE_CLOSE(v(row, col), wh(row, col), 10.0);
}

BOOST_AUTO_TEST_SUITE_END();
