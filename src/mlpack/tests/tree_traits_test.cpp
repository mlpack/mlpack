/**
 * @file tree_traits_test.cpp
 * @author Ryan Curtin
 *
 * Tests for the TreeTraits class.  These could all be known at compile-time,
 * but realistically the function is to be sure that nobody changes tree traits
 * without breaking something.  Thus, people must be certain when they make a
 * change like that (because they have to change the test too).  That's the
 * hope, at least...
 */
#include <mlpack/core.hpp>
#include <mlpack/core/tree/tree_traits.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/tree/cover_tree.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::metric;

BOOST_AUTO_TEST_SUITE(TreeTraitsTest);

// Be careful!  When writing new tests, always get the boolean value and store
// it in a temporary, because the Boost unit test macros do weird things and
// will cause bizarre problems.

// Test the defaults.
BOOST_AUTO_TEST_CASE(DefaultsTraitsTest)
{
  // An irrelevant non-tree type class is used here so that the default
  // implementation of TreeTraits is chosen.
  bool b = TreeTraits<int>::HasParentDistance;
  BOOST_REQUIRE_EQUAL(b, false);
  b = TreeTraits<int>::HasOverlappingChildren;
  BOOST_REQUIRE_EQUAL(b, true);
}

// Test the binary space tree traits.
BOOST_AUTO_TEST_CASE(BinarySpaceTreeTraitsTest)
{
  // ParentDistance() is not available.
  bool b = TreeTraits<BinarySpaceTree<LMetric<2, false> > >::HasParentDistance;
  BOOST_REQUIRE_EQUAL(b, false);

  // Children are non-overlapping.
  b = TreeTraits<BinarySpaceTree<LMetric<2, false> > >::HasOverlappingChildren;
  BOOST_REQUIRE_EQUAL(b, false);
}

// Test the cover tree traits.
BOOST_AUTO_TEST_CASE(CoverTreeTraitsTest)
{
  // ParentDistance() is available.
  bool b = TreeTraits<CoverTree<> >::HasParentDistance;
  BOOST_REQUIRE_EQUAL(b, true);

  // Children may be overlapping.
  b = TreeTraits<CoverTree<> >::HasOverlappingChildren;
  BOOST_REQUIRE_EQUAL(b, true);
}

BOOST_AUTO_TEST_SUITE_END();
