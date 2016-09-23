/**
 * @file octree_test.cpp
 * @author Ryan Curtin
 *
 * Test various properties of the Octree.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/tree/octree.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::math;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::bound;

BOOST_AUTO_TEST_SUITE(OctreeTest);

/**
 * Build a quad-tree (2-d octree) on 4 points, and guarantee four points are
 * created.
 */
BOOST_AUTO_TEST_CASE(SimpleQuadtreeTest)
{
  // Four corners of the unit square.
  arma::mat dataset("0 0 1 1; 0 1 0 1");

  Octree<> t(dataset, 1);

  BOOST_REQUIRE_EQUAL(t.NumChildren(), 4);
  BOOST_REQUIRE_EQUAL(t.Dataset().n_cols, 4);
  BOOST_REQUIRE_EQUAL(t.Dataset().n_rows, 2);
  BOOST_REQUIRE_EQUAL(t.NumDescendants(), 4);
  BOOST_REQUIRE_EQUAL(t.NumPoints(), 0);
  for (size_t i = 0; i < 4; ++i)
  {
    BOOST_REQUIRE_EQUAL(t.Child(i).NumDescendants(), 1);
    BOOST_REQUIRE_EQUAL(t.Child(i).NumPoints(), 1);
  }
}

/**
 * Build an octree on 3 points and make sure that only three children are
 * created.
 */
BOOST_AUTO_TEST_CASE(OctreeMissingChildTest)
{
  // Only three corners of the unit square.
  arma::mat dataset("0 0 1; 0 1 1");

  Octree<> t(dataset, 1);

  BOOST_REQUIRE_EQUAL(t.NumChildren(), 3);
  BOOST_REQUIRE_EQUAL(t.Dataset().n_cols, 3);
  BOOST_REQUIRE_EQUAL(t.Dataset().n_rows, 2);
  BOOST_REQUIRE_EQUAL(t.NumDescendants(), 3);
  BOOST_REQUIRE_EQUAL(t.NumPoints(), 0);
  for (size_t i = 0; i < 3; ++i)
  {
    BOOST_REQUIRE_EQUAL(t.Child(i).NumDescendants(), 1);
    BOOST_REQUIRE_EQUAL(t.Child(i).NumPoints(), 1);
  }
}

/**
 * Ensure that building an empty octree does not fail.
 */
BOOST_AUTO_TEST_CASE(EmptyOctreeTest)
{
  arma::mat dataset;
  Octree<> t(dataset);

  BOOST_REQUIRE_EQUAL(t.NumChildren(), 0);
  BOOST_REQUIRE_EQUAL(t.Dataset().n_cols, 0);
  BOOST_REQUIRE_EQUAL(t.Dataset().n_rows, 0);
  BOOST_REQUIRE_EQUAL(t.NumDescendants(), 0);
  BOOST_REQUIRE_EQUAL(t.NumPoints(), 0);
}

/**
 * Ensure that maxLeafSize is respected.
 */
BOOST_AUTO_TEST_CASE(MaxLeafSizeTest)
{
  arma::mat dataset(5, 15, arma::fill::randu);
  Octree<> t1(dataset, 20);
  Octree<> t2(std::move(dataset), 20);

  BOOST_REQUIRE_EQUAL(t1.NumChildren(), 0);
  BOOST_REQUIRE_EQUAL(t1.NumDescendants(), 15);
  BOOST_REQUIRE_EQUAL(t1.NumPoints(), 15);

  BOOST_REQUIRE_EQUAL(t2.NumChildren(), 0);
  BOOST_REQUIRE_EQUAL(t2.NumDescendants(), 15);
  BOOST_REQUIRE_EQUAL(t2.NumPoints(), 15);
}

/**
 * Check that the mappings given are correct.
 */
BOOST_AUTO_TEST_CASE(MappingsTest)
{
  // Test with both constructors.
  arma::mat dataset(3, 5, arma::fill::randu);
  arma::mat datacopy(dataset);
  std::vector<size_t> oldFromNewCopy, oldFromNewMove;

  Octree<> t1(dataset, oldFromNewCopy, 1);
  Octree<> t2(std::move(dataset), oldFromNewMove, 1);

  for (size_t i = 0; i < oldFromNewCopy.size(); ++i)
  {
    BOOST_REQUIRE_SMALL(arma::norm(datacopy.col(oldFromNewCopy[i]) -
        t1.Dataset().col(i)), 1e-3);
    BOOST_REQUIRE_SMALL(arma::norm(datacopy.col(oldFromNewMove[i]) -
        t2.Dataset().col(i)), 1e-3);
  }
}

/**
 * Check that the reverse mappings are correct too.
 */
BOOST_AUTO_TEST_CASE(ReverseMappingsTest)
{
  // Test with both constructors.
  arma::mat dataset(3, 300, arma::fill::randu);
  arma::mat datacopy(dataset);
  std::vector<size_t> oldFromNewCopy, oldFromNewMove, newFromOldCopy,
      newFromOldMove;

  Octree<> t1(dataset, oldFromNewCopy, newFromOldCopy);
  Octree<> t2(std::move(dataset), oldFromNewMove, newFromOldMove);

  for (size_t i = 0; i < oldFromNewCopy.size(); ++i)
  {
    BOOST_REQUIRE_SMALL(arma::norm(datacopy.col(oldFromNewCopy[i]) -
        t1.Dataset().col(i)), 1e-3);
    BOOST_REQUIRE_SMALL(arma::norm(datacopy.col(oldFromNewMove[i]) -
        t2.Dataset().col(i)), 1e-3);

    BOOST_REQUIRE_EQUAL(newFromOldCopy[oldFromNewCopy[i]], i);
    BOOST_REQUIRE_EQUAL(newFromOldMove[oldFromNewMove[i]], i);
  }
}

BOOST_AUTO_TEST_SUITE_END();
