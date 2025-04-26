/**
 * @file tests/union_find_test.cpp
 * @author Bill March (march@gatech.edu)
 *
 * Unit tests for the Union-Find data structure.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/emst/union_find.hpp>
#include "catch.hpp"

using namespace mlpack;

TEST_CASE("TestFind", "[UnionFindTest]")
{
  static const size_t testSize = 10;
  UnionFind testUnionFind(testSize);

  for (size_t i = 0; i < testSize; ++i)
    REQUIRE(testUnionFind.Find(i) == i);

  testUnionFind.Union(0, 1);
  testUnionFind.Union(1, 2);

  REQUIRE(testUnionFind.Find(2) == testUnionFind.Find(0));
}

TEST_CASE("TestUnion", "[UnionFindTest]")
{
  static const size_t testSize = 10;
  UnionFind testUnionFind(testSize);

  testUnionFind.Union(0, 1);
  testUnionFind.Union(2, 3);
  testUnionFind.Union(0, 2);
  testUnionFind.Union(5, 0);
  testUnionFind.Union(0, 6);

  REQUIRE(testUnionFind.Find(0) == testUnionFind.Find(1));
  REQUIRE(testUnionFind.Find(2) == testUnionFind.Find(3));
  REQUIRE(testUnionFind.Find(1) == testUnionFind.Find(5));
  REQUIRE(testUnionFind.Find(6) == testUnionFind.Find(3));
}

TEST_CASE("RedundantUnionTest", "[UnionFindTest]")
{
  UnionFind uf(5);
  uf.Union(1, 2);
  auto rep = uf.Find(1);

  // Re-union the same pair.
  uf.Union(1, 2);

  REQUIRE(uf.Find(1) == rep);
  REQUIRE(uf.Find(2) == rep);
}

TEST_CASE("FullUnionTest", "[UnionFindTest]")
{
  const size_t N = 10;
  UnionFind uf(N);

  for (size_t i = 1; i < N; ++i)
    uf.Union(0, i);

  const size_t root = uf.Find(0);
  for (size_t i = 1; i < N; ++i)
    REQUIRE(uf.Find(i) == root);
}

TEST_CASE("DisjointSetsRemainSeparateTest", "[UnionFindTest]")
{
  UnionFind uf(6);
  uf.Union(0, 1);
  uf.Union(2, 3);

  REQUIRE(uf.Find(0) != uf.Find(2));
  REQUIRE(uf.Find(1) == uf.Find(0));
  REQUIRE(uf.Find(3) == uf.Find(2));
  REQUIRE(uf.Find(4) != uf.Find(5)); // untouched elements
}

