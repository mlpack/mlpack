/**
 * @file tree_traits_test.cpp
 * @author Ryan Curtin
 *
 * Tests for the TreeTraits class.  These could all be known at compile-time,
 * but realistically the function is to be sure that nobody changes tree traits
 * without breaking something.  Thus, people must be certain when they make a
 * change like that (because they have to change the test too).  That's the
 * hope, at least...
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/tree/tree_traits.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/tree/cover_tree.hpp>
#include <mlpack/core/tree/rectangle_tree.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::metric;

BOOST_AUTO_TEST_SUITE(TreeTraitsTest);

// Be careful!  When writing new tests, always get the boolean value of each
// trait and store it in a temporary, because the Boost unit test macros do
// weird things and will cause bizarre problems.

// Test the defaults.
BOOST_AUTO_TEST_CASE(DefaultsTraitsTest)
{
  // An irrelevant non-tree type class is used here so that the default
  // implementation of TreeTraits is chosen.
  bool b = TreeTraits<int>::HasOverlappingChildren;
  BOOST_REQUIRE_EQUAL(b, true);
  b = TreeTraits<int>::HasSelfChildren;
  BOOST_REQUIRE_EQUAL(b, false);
  b = TreeTraits<int>::FirstPointIsCentroid;
  BOOST_REQUIRE_EQUAL(b, false);
  b = TreeTraits<int>::RearrangesDataset;
  BOOST_REQUIRE_EQUAL(b, false);
  b = TreeTraits<int>::BinaryTree;
  BOOST_REQUIRE_EQUAL(b, false);
}

// Test the binary space tree traits.
BOOST_AUTO_TEST_CASE(BinarySpaceTreeTraitsTest)
{
  typedef BinarySpaceTree<LMetric<2, false>> TreeType;

  // Children are non-overlapping.
  bool b = TreeTraits<TreeType>::HasOverlappingChildren;
  BOOST_REQUIRE_EQUAL(b, false);

  // Points are not contained at multiple levels.
  b = TreeTraits<TreeType>::HasSelfChildren;
  BOOST_REQUIRE_EQUAL(b, false);

  // The first point is not the centroid.
  b = TreeTraits<TreeType>::FirstPointIsCentroid;
  BOOST_REQUIRE_EQUAL(b, false);

  // The dataset gets rearranged at build time.
  b = TreeTraits<TreeType>::RearrangesDataset;
  BOOST_REQUIRE_EQUAL(b, true);

  // It is a binary tree.
  b = TreeTraits<TreeType>::BinaryTree;
  BOOST_REQUIRE_EQUAL(b, true);
}

// Test the cover tree traits.
BOOST_AUTO_TEST_CASE(CoverTreeTraitsTest)
{
  // Children may be overlapping.
  bool b = TreeTraits<CoverTree<>>::HasOverlappingChildren;
  BOOST_REQUIRE_EQUAL(b, true);

  // The cover tree has self-children.
  b = TreeTraits<CoverTree<>>::HasSelfChildren;
  BOOST_REQUIRE_EQUAL(b, true);

  // The first point is the center of the node.
  b = TreeTraits<CoverTree<>>::FirstPointIsCentroid;
  BOOST_REQUIRE_EQUAL(b, true);

  b = TreeTraits<CoverTree<>>::RearrangesDataset;
  BOOST_REQUIRE_EQUAL(b, false);

  b = TreeTraits<CoverTree<>>::BinaryTree;
  BOOST_REQUIRE_EQUAL(b, false); // Not necessarily binary.
}

BOOST_AUTO_TEST_SUITE_END();
