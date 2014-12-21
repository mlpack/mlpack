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
 * This file is part of MLPACK 1.0.9.
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
  bool b = TreeTraits<int>::HasOverlappingChildren;
  BOOST_REQUIRE_EQUAL(b, true);
  b = TreeTraits<int>::HasSelfChildren;
  BOOST_REQUIRE_EQUAL(b, false);
}

// Test the binary space tree traits.
BOOST_AUTO_TEST_CASE(BinarySpaceTreeTraitsTest)
{
  // Children are non-overlapping.
  bool b =
      TreeTraits<BinarySpaceTree<LMetric<2, false> > >::HasOverlappingChildren;
  BOOST_REQUIRE_EQUAL(b, false);

  // Points are not contained at multiple levels.
  b = TreeTraits<BinarySpaceTree<LMetric<2, false> > >::HasSelfChildren;
  BOOST_REQUIRE_EQUAL(b, false);
}

// Test the cover tree traits.
BOOST_AUTO_TEST_CASE(CoverTreeTraitsTest)
{
  // Children may be overlapping.
  bool b = TreeTraits<CoverTree<> >::HasOverlappingChildren;
  BOOST_REQUIRE_EQUAL(b, true);

  // The cover tree has self-children.
  b = TreeTraits<CoverTree<> >::HasSelfChildren;
  BOOST_REQUIRE_EQUAL(b, true);
}

BOOST_AUTO_TEST_SUITE_END();
