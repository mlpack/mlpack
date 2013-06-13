/**
 * @file union_find_test.cpp
 * @author Bill March (march@gatech.edu)
 *
 * Unit tests for the Union-Find data structure.
 *
 * This file is part of MLPACK 1.0.6.
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
#include <mlpack/methods/emst/union_find.hpp>

#include <mlpack/core.hpp>
#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::emst;

BOOST_AUTO_TEST_SUITE(UnionFindTest);

BOOST_AUTO_TEST_CASE(TestFind)
{
  static const size_t testSize_ = 10;
  UnionFind testUnionFind_(testSize_);

  for (size_t i = 0; i < testSize_; i++)
    BOOST_REQUIRE(testUnionFind_.Find(i) == i);

  testUnionFind_.Union(0, 1);
  testUnionFind_.Union(1, 2);

  BOOST_REQUIRE(testUnionFind_.Find(2) == testUnionFind_.Find(0));
}

BOOST_AUTO_TEST_CASE(TestUnion)
{
  static const size_t testSize_ = 10;
  UnionFind testUnionFind_(testSize_);

  testUnionFind_.Union(0, 1);
  testUnionFind_.Union(2, 3);
  testUnionFind_.Union(0, 2);
  testUnionFind_.Union(5, 0);
  testUnionFind_.Union(0, 6);

  BOOST_REQUIRE(testUnionFind_.Find(0) == testUnionFind_.Find(1));
  BOOST_REQUIRE(testUnionFind_.Find(2) == testUnionFind_.Find(3));
  BOOST_REQUIRE(testUnionFind_.Find(1) == testUnionFind_.Find(5));
  BOOST_REQUIRE(testUnionFind_.Find(6) == testUnionFind_.Find(3));
}

BOOST_AUTO_TEST_SUITE_END();
