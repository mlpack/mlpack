/**
 * @file union_find_test.cpp
 * @author Bill March (march@gatech.edu)
 *
 * Unit tests for the Union-Find data structure.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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
