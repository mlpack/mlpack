/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file uselapack_test.cc
 *
 * Tests for LAPACK integration.
 */

#include "bounds.h"
//#include "bounds.h"

#include "../base/test.h"

TEST_SUITE_BEGIN(uselapack);

void TestBallBound() {
  DBallBound<> b1;
  DBallBound<> b2;
  const double EPS = 1.0e-14;

  // Create two balls with a center distance of 1 from each other.
  // Give the first one a radius of 0.3 and the second a radius of 0.4.

  b1.center().set_size(3);
  b1.center()[0] = 1;
  b1.center()[1] = 2;
  b1.center()[2] = 3;
  b1.set_radius(0.3);

  b2.center().set_size(3);
  b2.center()[0] = 1;
  b2.center()[1] = 2;
  b2.center()[2] = 4;
  b2.set_radius(0.4);

  TEST_DOUBLE_APPROX(sqrt(b1.MinDistanceSq(b2)), 1-0.3-0.4, EPS);
  TEST_DOUBLE_APPROX(sqrt(b1.MaxDistanceSq(b2)), 1+0.3+0.4, EPS);
  TEST_DOUBLE_APPROX(sqrt(b1.RangeDistanceSq(b2).hi), 1+0.3+0.4, EPS);
  TEST_DOUBLE_APPROX(sqrt(b1.RangeDistanceSq(b2).lo), 1-0.3-0.4, EPS);
  TEST_DOUBLE_APPROX(b1.RangeDistance(b2).hi, 1+0.3+0.4, EPS);
  TEST_DOUBLE_APPROX(b1.RangeDistance(b2).lo, 1-0.3-0.4, EPS);
  TEST_DOUBLE_APPROX(sqrt(b1.MinToMidSq(b2)), 1-0.3, EPS);
  TEST_DOUBLE_APPROX(sqrt(b1.MinimaxDistanceSq(b2)), 1-0.3+0.4, EPS);
  TEST_DOUBLE_APPROX(sqrt(b1.MidDistanceSq(b2)), 1, EPS);

  TEST_DOUBLE_APPROX(sqrt(b2.MinDistanceSq(b1)), 1-0.3-0.4, EPS);
  TEST_DOUBLE_APPROX(sqrt(b2.MaxDistanceSq(b1)), 1+0.3+0.4, EPS);
  TEST_DOUBLE_APPROX(sqrt(b2.RangeDistanceSq(b1).hi), 1+0.3+0.4, EPS);
  TEST_DOUBLE_APPROX(sqrt(b2.RangeDistanceSq(b1).lo), 1-0.3-0.4, EPS);
  TEST_DOUBLE_APPROX(sqrt(b2.MinToMidSq(b1)), 1-0.4, EPS);
  TEST_DOUBLE_APPROX(sqrt(b2.MinimaxDistanceSq(b1)), 1-0.4+0.3, EPS);
  TEST_DOUBLE_APPROX(sqrt(b2.MidDistanceSq(b1)), 1, EPS);

  TEST_ASSERT(b1.Contains(b1.center()));
  TEST_ASSERT(!b1.Contains(b2.center()));
  TEST_ASSERT(!b2.Contains(b1.center()));
  TEST_ASSERT(b2.Contains(b2.center()));
  arma::vec b2point(3); // a point that's within the radius bot not the center
  b2point[0] = 1.1;
  b2point[1] = 2.1;
  b2point[2] = 4.1;
  TEST_ASSERT(b2.Contains(b2point));

  TEST_DOUBLE_APPROX(sqrt(b1.MinDistanceSq(b1.center())), 0, EPS);
  TEST_DOUBLE_APPROX(sqrt(b1.MinDistanceSq(b2.center())), 1-0.3, EPS);
  TEST_DOUBLE_APPROX(sqrt(b2.MinDistanceSq(b1.center())), 1-0.4, EPS);
  TEST_DOUBLE_APPROX(sqrt(b2.MaxDistanceSq(b1.center())), 1+0.4, EPS);
  TEST_DOUBLE_APPROX(sqrt(b1.MaxDistanceSq(b2.center())), 1+0.3, EPS);
}

TEST_SUITE_END(uselapack,
    TestBallBound,
    );
