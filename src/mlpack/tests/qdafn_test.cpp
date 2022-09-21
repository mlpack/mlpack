/**
 * @file tests/qdafn_test.cpp
 * @author Ryan Curtin
 *
 * Test the QDAFN functionality.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/approx_kfn.hpp>
#include <mlpack/methods/neighbor_search.hpp>

#include "catch.hpp"
#include "serialization.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;

/**
 * With one reference point, make sure that is the one that is returned.
 */
TEST_CASE("QDAFNTrivialTest", "[QDAFNTest]")
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
  REQUIRE(neighbors.n_rows == 1);
  REQUIRE(neighbors.n_cols == 5);
  REQUIRE(distances.n_rows == 1);
  REQUIRE(distances.n_cols == 5);

  for (size_t i = 0; i < 5; ++i)
  {
    REQUIRE(neighbors[i] == 0);
    const double dist = EuclideanDistance::Evaluate(querySet.col(i),
        refSet.col(0));
    REQUIRE(distances[i] == Approx(dist).epsilon(1e-7));
  }
}

/**
 * Given a random uniform reference set, ensure that we get a neighbor and
 * distance within 10% of the actual true furthest neighbor distance at least
 * 70% of the time.
 */
TEST_CASE("QDAFNUniformSet", "[QDAFNTest]")
{
  arma::mat uniformSet = arma::randu<arma::mat>(25, 1000);

  QDAFN<> qdafn(uniformSet, 10, 30);

  // Get the actual neighbors.
  KFN kfn(uniformSet);
  arma::Mat<size_t> trueNeighbors;
  arma::mat trueDistances;

  kfn.Search(999, trueNeighbors, trueDistances);

  arma::Mat<size_t> qdafnNeighbors;
  arma::mat qdafnDistances;

  qdafn.Search(uniformSet, 1, qdafnNeighbors, qdafnDistances);

  REQUIRE(qdafnNeighbors.n_rows == 1);
  REQUIRE(qdafnNeighbors.n_cols == 1000);
  REQUIRE(qdafnDistances.n_rows == 1);
  REQUIRE(qdafnDistances.n_cols == 1000);

  size_t successes = 0;
  for (size_t i = 0; i < 999; ++i)
  {
    // Find the true neighbor.
    size_t trueIndex = 999;
    for (size_t j = 0; j < 999; ++j)
    {
      if (trueNeighbors(j, i) == qdafnNeighbors(0, i))
      {
        trueIndex = j;
        break;
      }
    }

    REQUIRE(trueIndex != 999);
    if (0.9 * trueDistances(0, i) <= qdafnDistances(0, i))
      ++successes;
  }

  REQUIRE(successes >= 695);
}

/**
 * Make sure that more than one valid neighbor is returned when k > 1.
 */
TEST_CASE("QDAFNMultipleNeighbors", "[QDAFNTest]")
{
  arma::mat uniformSet = arma::randu<arma::mat>(25, 1000);

  QDAFN<> qdafn(uniformSet, 10, 30);

  // Get the actual neighbors.
  KFN kfn(uniformSet);
  arma::Mat<size_t> trueNeighbors;
  arma::mat trueDistances;

  kfn.Search(999, trueNeighbors, trueDistances);

  arma::Mat<size_t> qdafnNeighbors;
  arma::mat qdafnDistances;

  qdafn.Search(uniformSet, 3, qdafnNeighbors, qdafnDistances);

  REQUIRE(qdafnNeighbors.n_rows == 3);
  REQUIRE(qdafnNeighbors.n_cols == 1000);
  REQUIRE(qdafnDistances.n_rows == 3);
  REQUIRE(qdafnDistances.n_cols == 1000);

  // We expect to find a neighbor for each point.
  for (size_t i = 0; i < 999; ++i)
  {
    REQUIRE(qdafnNeighbors(0, i) < 1000);
    REQUIRE(qdafnNeighbors(1, i) < 1000);
    REQUIRE(qdafnNeighbors(2, i) < 1000);
    REQUIRE(qdafnNeighbors(0, i) != qdafnNeighbors(1, i));
    REQUIRE(qdafnNeighbors(0, i) != qdafnNeighbors(2, i));
  }
}

/**
 * Test re-training method.
 */
TEST_CASE("RetrainTest", "[QDAFNTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(25, 500);
  arma::mat newDataset = arma::randu<arma::mat>(15, 600);

  QDAFN<> qdafn(dataset, 20, 60);

  qdafn.Train(newDataset, 10, 50);

  REQUIRE(qdafn.NumProjections() == 10);
  for (size_t i = 0; i < 10; ++i)
  {
    REQUIRE(qdafn.CandidateSet(i).n_rows == 15);
    REQUIRE(qdafn.CandidateSet(i).n_cols == 50);
  }
}

/**
 * Test serialization of QDAFN.
 */
TEST_CASE("QDAFNSerializationTest", "[QDAFNTest]")
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
  REQUIRE(qdafnXml.NumProjections() == qdafn.NumProjections());
  REQUIRE(qdafnText.NumProjections() == qdafn.NumProjections());
  REQUIRE(qdafnBinary.NumProjections() == qdafn.NumProjections());

  for (size_t i = 0; i < qdafn.NumProjections(); ++i)
  {
    REQUIRE(qdafnXml.CandidateSet(i).n_rows ==
        qdafn.CandidateSet(i).n_rows);
    REQUIRE(qdafnText.CandidateSet(i).n_rows ==
        qdafn.CandidateSet(i).n_rows);
    REQUIRE(qdafnBinary.CandidateSet(i).n_rows ==
        qdafn.CandidateSet(i).n_rows);

    REQUIRE(qdafnXml.CandidateSet(i).n_cols ==
        qdafn.CandidateSet(i).n_cols);
    REQUIRE(qdafnText.CandidateSet(i).n_cols ==
        qdafn.CandidateSet(i).n_cols);
    REQUIRE(qdafnBinary.CandidateSet(i).n_cols ==
        qdafn.CandidateSet(i).n_cols);

    for (size_t j = 0; j < qdafn.CandidateSet(i).n_elem; ++j)
    {
      if (std::abs(qdafn.CandidateSet(i)[j]) < 1e-5)
      {
        REQUIRE(qdafnXml.CandidateSet(i)[j] == Approx(0.0).margin(1e-5));
        REQUIRE(qdafnText.CandidateSet(i)[j] == Approx(0.0).margin(1e-5));
        REQUIRE(qdafnBinary.CandidateSet(i)[j] == Approx(0.0).margin(1e-5));
      }
      else
      {
        const double value = qdafn.CandidateSet(i)[j];
        REQUIRE(qdafnXml.CandidateSet(i)[j] == Approx(value).epsilon(1e-7));
        REQUIRE(qdafnText.CandidateSet(i)[j] == Approx(value).epsilon(1e-7));
        REQUIRE(qdafnBinary.CandidateSet(i)[j] == Approx(value).epsilon(1e-7));
      }
    }
  }
}

// Make sure QDAFN works with sparse data.
TEST_CASE("QDAFNSparseTest", "[QDAFNTest]")
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

  REQUIRE(neighbors.n_rows == 3);
  REQUIRE(neighbors.n_cols == 1000);
  REQUIRE(distances.n_rows == 3);
  REQUIRE(distances.n_cols == 1000);
}
