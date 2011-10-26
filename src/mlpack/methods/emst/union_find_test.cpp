/**
 * @file union_find_test.cc
 *
 * @author Bill March (march@gatech.edu)
 *
 * Unit tests for the Union-Find data structure.
 */

#include "union_find.hpp"

#include <mlpack/core.h>

#define BOOST_TEST_MODULE UnionFindTest
#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::emst;

BOOST_AUTO_TEST_CASE(TestFind) {
  static const size_t test_size_ = 10;
  UnionFind test_union_find_;
  test_union_find_.Init(test_size_);

  for (size_t i = 0; i < test_size_; i++) {
    BOOST_REQUIRE(test_union_find_.Find(i) == i);
  }
  test_union_find_.Union(0,1);
  test_union_find_.Union(1, 2);

  BOOST_REQUIRE(test_union_find_.Find(2) == test_union_find_.Find(0));

}

BOOST_AUTO_TEST_CASE(TestUnion) {
  static const size_t test_size_ = 10;
  UnionFind test_union_find_;
  test_union_find_.Init(test_size_);

  test_union_find_.Union(0, 1);
  test_union_find_.Union(2, 3);
  test_union_find_.Union(0, 2);
  test_union_find_.Union(5, 0);
  test_union_find_.Union(0, 6);

  BOOST_REQUIRE(test_union_find_.Find(0) == test_union_find_.Find(1));
  BOOST_REQUIRE(test_union_find_.Find(2) == test_union_find_.Find(3));
  BOOST_REQUIRE(test_union_find_.Find(1) == test_union_find_.Find(5));
  BOOST_REQUIRE(test_union_find_.Find(6) == test_union_find_.Find(3));
}

