/**
 * @file nmf_test.cpp
 * @author Mohan Rajendran
 *
 * Test file for NMF class.
 *
 * This file is part of MLPACK 1.0.7.
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
 * input matrix. Default case.
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

  for (size_t row = 0; row < 20; row++)
    for (size_t col = 0; col < 20; col++)
      BOOST_REQUIRE_CLOSE(v(row, col), wh(row, col), 10.0);
}

/**
 * Check the if the product of the calculated factorization is close to the
 * input matrix. Random Acol initialization distance minimization update.
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

  for (size_t row = 0; row < 20; row++)
    for (size_t col = 0; col < 20; col++)
      BOOST_REQUIRE_CLOSE(v(row, col), wh(row, col), 10.0);
}

/**
 * Check the if the product of the calculated factorization is close to the
 * input matrix. Random initialization divergence minimization update.
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

  for (size_t row = 0; row < 20; row++)
    for (size_t col = 0; col < 20; col++)
      BOOST_REQUIRE_CLOSE(v(row, col), wh(row, col), 10.0);
}

/**
 * Check that the product of the calculated factorization is close to the
 * input matrix.  This uses the random initialization and alternating least
 * squares update rule.
 */
BOOST_AUTO_TEST_CASE(NMFALSTest)
{
  mat w = randu<mat>(20, 16);
  mat h = randu<mat>(16, 20);
  mat v = w * h;
  size_t r = 16;

  NMF<RandomInitialization,
      WAlternatingLeastSquaresRule,
      HAlternatingLeastSquaresRule> nmf(50000, 1e-15);
  nmf.Apply(v, r, w, h);

  mat wh = w * h;

  for (size_t row = 0; row < 20; row++)
    for (size_t col = 0; col < 20; col++)
      BOOST_REQUIRE_CLOSE(v(row, col), wh(row, col), 15.0);
}

/**
 * Check the if the product of the calculated factorization is close to the
 * input matrix, with a sparse input matrix.. Default case.
 */
BOOST_AUTO_TEST_CASE(SparseNMFDefaultTest)
{
  mat w, h;
  sp_mat v;
  v.sprandu(20, 20, 0.2);
  mat dv(v); // Make a dense copy.
  mat dw, dh;
  size_t r = 18;

  // It seems to hit the iteration limit first.
  NMF<> nmf(10000, 1e-20);
  mlpack::math::RandomSeed(1000); // Set random seed so results are the same.
  nmf.Apply(v, r, w, h);
  mlpack::math::RandomSeed(1000);
  nmf.Apply(dv, r, dw, dh);

  // Make sure the results are about equal for the W and H matrices.
  for (size_t i = 0; i < w.n_elem; ++i)
  {
    if (w(i) == 0.0)
      BOOST_REQUIRE_SMALL(dw(i), 1e-15);
    else
      BOOST_REQUIRE_CLOSE(w(i), dw(i), 1e-5);
  }

  for (size_t i = 0; i < h.n_elem; ++i)
  {
    if (h(i) == 0.0)
      BOOST_REQUIRE_SMALL(dh(i), 1e-15);
    else
      BOOST_REQUIRE_CLOSE(h(i), dh(i), 1e-5);
  }
}

/**
 * Check the if the product of the calculated factorization is close to the
 * input matrix, with a sparse input matrix. Random Acol initialization,
 * distance minimization update.
 */
BOOST_AUTO_TEST_CASE(SparseNMFAcolDistTest)
{
  mat w, h;
  sp_mat v;
  v.sprandu(20, 20, 0.3);
  mat dv(v); // Make a dense copy.
  mat dw, dh;
  size_t r = 16;

  NMF<RandomAcolInitialization<> > nmf;
  mlpack::math::RandomSeed(1000); // Set random seed so results are the same.
  nmf.Apply(v, r, w, h);
  mlpack::math::RandomSeed(1000);
  nmf.Apply(dv, r, dw, dh);

  // Make sure the results are about equal for the W and H matrices.
  for (size_t i = 0; i < w.n_elem; ++i)
  {
    if (w(i) == 0.0)
      BOOST_REQUIRE_SMALL(dw(i), 1e-15);
    else
      BOOST_REQUIRE_CLOSE(w(i), dw(i), 1e-5);
  }

  for (size_t i = 0; i < h.n_elem; ++i)
  {
    if (h(i) == 0.0)
      BOOST_REQUIRE_SMALL(dh(i), 1e-15);
    else
      BOOST_REQUIRE_CLOSE(h(i), dh(i), 1e-5);
  }
}

/**
 * Check the if the product of the calculated factorization is close to the
 * input matrix, with a sparse input matrix. Random initialization, divergence
 * minimization update.
 */
BOOST_AUTO_TEST_CASE(SparseNMFRandomDivTest)
{
  mat w, h;
  sp_mat v;
  v.sprandu(20, 20, 0.3);
  mat dv(v); // Make a dense copy.
  mat dw, dh;
  size_t r = 16;

  NMF<RandomInitialization,
      WMultiplicativeDivergenceRule,
      HMultiplicativeDivergenceRule> nmf;
  mlpack::math::RandomSeed(10); // Set random seed so the results are the same.
  nmf.Apply(v, r, w, h);
  mlpack::math::RandomSeed(10);
  nmf.Apply(dv, r, dw, dh);

  // Make sure the results are about equal for the W and H matrices.
  for (size_t i = 0; i < w.n_elem; ++i)
  {
    if (w(i) == 0.0)
      BOOST_REQUIRE_SMALL(dw(i), 1e-15);
    else
      BOOST_REQUIRE_CLOSE(w(i), dw(i), 1e-5);
  }

  for (size_t i = 0; i < h.n_elem; ++i)
  {
    if (h(i) == 0.0)
      BOOST_REQUIRE_SMALL(dh(i), 1e-15);
    else
      BOOST_REQUIRE_CLOSE(h(i), dh(i), 1e-5);
  }
}

/**
 * Check that the product of the calculated factorization is close to the
 * input matrix, with a sparse input matrix.  This uses the random
 * initialization and alternating least squares update rule.
 */
BOOST_AUTO_TEST_CASE(SparseNMFALSTest)
{
  mat w, h;
  sp_mat v;
  v.sprandu(10, 10, 0.3);
  mat dv(v); // Make a dense copy.
  mat dw, dh;
  size_t r = 8;

  NMF<RandomInitialization,
      WAlternatingLeastSquaresRule,
      HAlternatingLeastSquaresRule> nmf;
  mlpack::math::RandomSeed(40);
  nmf.Apply(v, r, w, h);
  mlpack::math::RandomSeed(40);
  nmf.Apply(dv, r, dw, dh);

  // Make sure the results are about equal for the W and H matrices.
  for (size_t i = 0; i < w.n_elem; ++i)
  {
    if (w(i) == 0.0)
      BOOST_REQUIRE_SMALL(dw(i), 1e-15);
    else
      BOOST_REQUIRE_CLOSE(w(i), dw(i), 1e-5);
  }

  for (size_t i = 0; i < h.n_elem; ++i)
  {
    if (h(i) == 0.0)
      BOOST_REQUIRE_SMALL(dh(i), 1e-15);
    else
      BOOST_REQUIRE_CLOSE(h(i), dh(i), 1e-5);
  }
}

BOOST_AUTO_TEST_SUITE_END();
