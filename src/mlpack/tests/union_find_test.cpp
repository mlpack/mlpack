/**
 * @file union_find_test.cpp
 * @author Bill March (march@gatech.edu)
 *
 * Unit tests for the Union-Find data structure.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/methods/emst/union_find.hpp>

#include <mlpack/core.hpp>
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::emst;

BOOST_AUTO_TEST_SUITE(UnionFindTest);

BOOST_AUTO_TEST_CASE(TestFind)
{
  static const size_t testSize = 10;
  UnionFind testUnionFind(testSize);

  for (size_t i = 0; i < testSize; i++)
    BOOST_REQUIRE(testUnionFind.Find(i) == i);

  testUnionFind.Union(0, 1);
  testUnionFind.Union(1, 2);

  BOOST_REQUIRE(testUnionFind.Find(2) == testUnionFind.Find(0));
}

BOOST_AUTO_TEST_CASE(TestUnion)
{
  static const size_t testSize = 10;
  UnionFind testUnionFind(testSize);

  testUnionFind.Union(0, 1);
  testUnionFind.Union(2, 3);
  testUnionFind.Union(0, 2);
  testUnionFind.Union(5, 0);
  testUnionFind.Union(0, 6);

  BOOST_REQUIRE(testUnionFind.Find(0) == testUnionFind.Find(1));
  BOOST_REQUIRE(testUnionFind.Find(2) == testUnionFind.Find(3));
  BOOST_REQUIRE(testUnionFind.Find(1) == testUnionFind.Find(5));
  BOOST_REQUIRE(testUnionFind.Find(6) == testUnionFind.Find(3));
}

BOOST_AUTO_TEST_SUITE_END();
