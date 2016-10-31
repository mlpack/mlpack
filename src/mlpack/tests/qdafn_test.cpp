/**
 * @file qdafn_test.cpp
 * @author Ryan Curtin
 *
 * Test the QDAFN functionality.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"
#include "serialization.hpp"

#include <mlpack/core.hpp>
#include <mlpack/methods/approx_kfn/qdafn.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::neighbor;

BOOST_AUTO_TEST_SUITE(QDAFNTest);

/**
 * With one reference point, make sure that is the one that is returned.
 */
BOOST_AUTO_TEST_CASE(QDAFNTrivialTest)
{
  arma::mat refSet(5, 1);
  refSet.randu();

  // 5 tables, 1 point.
  QDAFN<> qdafn(refSet, 5, 1);

  arma::mat querySet(5, 5);
  querySet.randu();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  qdafn.Search(querySet, 1, neighbors, distances);

  // Check sizes.
  BOOST_REQUIRE_EQUAL(neighbors.n_rows, 1);
  BOOST_REQUIRE_EQUAL(neighbors.n_cols, 5);
  BOOST_REQUIRE_EQUAL(distances.n_rows, 1);
  BOOST_REQUIRE_EQUAL(distances.n_cols, 5);

  for (size_t i = 0; i < 5; ++i)
  {
    BOOST_REQUIRE_EQUAL(neighbors[i], 0);
    const double dist = metric::EuclideanDistance::Evaluate(querySet.col(i),
        refSet.col(0));
    BOOST_REQUIRE_CLOSE(distances[i], dist, 1e-5);
  }
}

/**
 * Given a random uniform reference set, ensure that we get a neighbor and
 * distance within 10% of the actual true furthest neighbor distance at least
 * 70% of the time.
 */
BOOST_AUTO_TEST_CASE(QDAFNUniformSet)
{
  arma::mat uniformSet = arma::randu<arma::mat>(25, 1000);

  QDAFN<> qdafn(uniformSet, 10, 30);

  // Get the actual neighbors.
  AllkFN kfn(uniformSet);
  arma::Mat<size_t> trueNeighbors;
  arma::mat trueDistances;

  kfn.Search(1000, trueNeighbors, trueDistances);

  arma::Mat<size_t> qdafnNeighbors;
  arma::mat qdafnDistances;

  qdafn.Search(uniformSet, 1, qdafnNeighbors, qdafnDistances);

  BOOST_REQUIRE_EQUAL(qdafnNeighbors.n_rows, 1);
  BOOST_REQUIRE_EQUAL(qdafnNeighbors.n_cols, 1000);
  BOOST_REQUIRE_EQUAL(qdafnDistances.n_rows, 1);
  BOOST_REQUIRE_EQUAL(qdafnDistances.n_cols, 1000);

  size_t successes = 0;
  for (size_t i = 0; i < 1000; ++i)
  {
    // Find the true neighbor.
    size_t trueIndex = 1000;
    for (size_t j = 0; j < 1000; ++j)
    {
      if (trueNeighbors(j, i) == qdafnNeighbors(0, i))
      {
        trueIndex = j;
        break;
      }
    }

    BOOST_REQUIRE_NE(trueIndex, 1000);
    if (0.9 * trueDistances(0, i) <= qdafnDistances(0, i))
      ++successes;
  }

  BOOST_REQUIRE_GE(successes, 700);
}

/**
 * Test re-training method.
 */
BOOST_AUTO_TEST_CASE(RetrainTest)
{
  arma::mat dataset = arma::randu<arma::mat>(25, 500);
  arma::mat newDataset = arma::randu<arma::mat>(15, 600);

  QDAFN<> qdafn(dataset, 20, 60);

  qdafn.Train(newDataset, 10, 50);

  BOOST_REQUIRE_EQUAL(qdafn.NumProjections(), 10);
  for (size_t i = 0; i < 10; ++i)
  {
    BOOST_REQUIRE_EQUAL(qdafn.CandidateSet(i).n_rows, 15);
    BOOST_REQUIRE_EQUAL(qdafn.CandidateSet(i).n_cols, 50);
  }
}

/**
 * Test serialization of QDAFN.
 */
BOOST_AUTO_TEST_CASE(SerializationTest)
{
  // Use a random dataset.
  arma::mat dataset = arma::randu<arma::mat>(15, 300);

  QDAFN<> qdafn(dataset, 10, 50);

  arma::mat fakeDataset1 = arma::randu<arma::mat>(10, 200);
  arma::mat fakeDataset2 = arma::randu<arma::mat>(50, 500);
  QDAFN<> qdafnXml(fakeDataset1, 5, 10);
  QDAFN<> qdafnText(6, 50);
  QDAFN<> qdafnBinary(7, 15);
  qdafnBinary.Train(fakeDataset2);

  // Serialize the objects.
  SerializeObjectAll(qdafn, qdafnXml, qdafnText, qdafnBinary);

  // Check that the tables are all the same.
  BOOST_REQUIRE_EQUAL(qdafnXml.NumProjections(), qdafn.NumProjections());
  BOOST_REQUIRE_EQUAL(qdafnText.NumProjections(), qdafn.NumProjections());
  BOOST_REQUIRE_EQUAL(qdafnBinary.NumProjections(), qdafn.NumProjections());

  for (size_t i = 0; i < qdafn.NumProjections(); ++i)
  {
    BOOST_REQUIRE_EQUAL(qdafnXml.CandidateSet(i).n_rows,
        qdafn.CandidateSet(i).n_rows);
    BOOST_REQUIRE_EQUAL(qdafnText.CandidateSet(i).n_rows,
        qdafn.CandidateSet(i).n_rows);
    BOOST_REQUIRE_EQUAL(qdafnBinary.CandidateSet(i).n_rows,
        qdafn.CandidateSet(i).n_rows);

    BOOST_REQUIRE_EQUAL(qdafnXml.CandidateSet(i).n_cols,
        qdafn.CandidateSet(i).n_cols);
    BOOST_REQUIRE_EQUAL(qdafnText.CandidateSet(i).n_cols,
        qdafn.CandidateSet(i).n_cols);
    BOOST_REQUIRE_EQUAL(qdafnBinary.CandidateSet(i).n_cols,
        qdafn.CandidateSet(i).n_cols);

    for (size_t j = 0; j < qdafn.CandidateSet(i).n_elem; ++j)
    {
      if (std::abs(qdafn.CandidateSet(i)[j]) < 1e-5)
      {
        BOOST_REQUIRE_SMALL(qdafnXml.CandidateSet(i)[j], 1e-5);
        BOOST_REQUIRE_SMALL(qdafnText.CandidateSet(i)[j], 1e-5);
        BOOST_REQUIRE_SMALL(qdafnBinary.CandidateSet(i)[j], 1e-5);
      }
      else
      {
        const double value = qdafn.CandidateSet(i)[j];
        BOOST_REQUIRE_CLOSE(qdafnXml.CandidateSet(i)[j], value, 1e-5);
        BOOST_REQUIRE_CLOSE(qdafnText.CandidateSet(i)[j], value, 1e-5);
        BOOST_REQUIRE_CLOSE(qdafnBinary.CandidateSet(i)[j], value, 1e-5);
      }
    }
  }
}

// Make sure QDAFN works with sparse data.
BOOST_AUTO_TEST_CASE(SparseTest)
{
  arma::sp_mat dataset;
  dataset.sprandu(200, 1000, 0.3);

  // Create a sparse version.
  QDAFN<arma::sp_mat> sparse(dataset, 15, 50);

  // Make sure the results are of the right shape.  It's hard to test anything
  // more than that because we don't have easy-to-check performance guarantees.
  arma::Mat<size_t> neighbors;
  arma::mat distances;
  sparse.Search(dataset, 3, neighbors, distances);

  BOOST_REQUIRE_EQUAL(neighbors.n_rows, 3);
  BOOST_REQUIRE_EQUAL(neighbors.n_cols, 1000);
  BOOST_REQUIRE_EQUAL(distances.n_rows, 3);
  BOOST_REQUIRE_EQUAL(distances.n_cols, 1000);
}

BOOST_AUTO_TEST_SUITE_END();
