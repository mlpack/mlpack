/**
 * @file tests/drusilla_select_test.cpp
 * @author Ryan Curtin
 *
 * Test for DrusillaSelect.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/methods/approx_kfn.hpp>
#include <mlpack/methods/neighbor_search.hpp>

#include "catch.hpp"
#include "serialization.hpp"

using namespace mlpack;

// If we have a dataset with an extreme outlier, then every point (except that
// one) should end up with that point as the furthest neighbor candidate.
TEST_CASE("DrusillaSelectExtremeOutlierTest", "[DrusillaSelectTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(5, 100);
  dataset.col(99) += 100; // Make last column very large.

  // Construct with some reasonable parameters.
  DrusillaSelect<> ds(dataset, 5, 5);

  // Query with every point except the extreme point.
  arma::mat distances;
  arma::Mat<size_t> neighbors;
  ds.Search(dataset.cols(0, 98), 1, neighbors, distances);

  REQUIRE(neighbors.n_cols == 99);
  REQUIRE(neighbors.n_rows == 1);
  REQUIRE(distances.n_cols == 99);
  REQUIRE(distances.n_rows == 1);

  for (size_t i = 0; i < 99; ++i)
  {
    REQUIRE(neighbors[i] == 99);
  }
}

// If we use only one projection with the number of points equal to what is in
// the dataset, we should end up with the exact result.
TEST_CASE("DrusillaSelectExhaustiveExactTest", "[DrusillaSelectTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(5, 100);

  // Construct with one projection and 100 points in that projection.
  DrusillaSelect<> ds(dataset, 100, 1);

  arma::mat distances, distancesTrue;
  arma::Mat<size_t> neighbors, neighborsTrue;

  ds.Search(dataset, 5, neighbors, distances);

  KFN kfn(dataset);
  kfn.Search(dataset, 5, neighborsTrue, distancesTrue);

  REQUIRE(neighborsTrue.n_cols == neighbors.n_cols);
  REQUIRE(neighborsTrue.n_rows == neighbors.n_rows);
  REQUIRE(distancesTrue.n_cols == distances.n_cols);
  REQUIRE(distancesTrue.n_rows == distances.n_rows);

  for (size_t i = 0; i < distances.n_elem; ++i)
  {
    REQUIRE(neighbors[i] == neighborsTrue[i]);
    REQUIRE(distances[i] == Approx(distancesTrue[i]).epsilon(1e-7));
  }
}

// Test that we can call Train() after calling the constructor.
TEST_CASE("DrusillaSelectRetrainTest", "[DrusillaSelectTest]")
{
  arma::mat firstDataset = arma::randu<arma::mat>(3, 10);
  arma::mat dataset = arma::randu<arma::mat>(3, 200);

  DrusillaSelect<> ds(firstDataset, 3, 3);
  ds.Train(std::move(dataset), 2, 2);

  arma::mat distances;
  arma::Mat<size_t> neighbors;
  ds.Search(dataset, 1, neighbors, distances);

  REQUIRE(neighbors.n_cols == 200);
  REQUIRE(neighbors.n_rows == 1);
  REQUIRE(distances.n_cols == 200);
  REQUIRE(distances.n_rows == 1);
}

// Test serialization.
TEST_CASE("DrusillaSelectSerializationTest", "[DrusillaSelectTest]")
{
  // Create a random dataset.
  arma::mat dataset = arma::randu<arma::mat>(3, 100);

  DrusillaSelect<> ds(dataset, 3, 3);

  arma::mat fakeDataset1 = arma::randu<arma::mat>(2, 15);
  arma::mat fakeDataset2 = arma::randu<arma::mat>(10, 18);
  DrusillaSelect<> dsXml(fakeDataset1, 5, 3);
  DrusillaSelect<> dsText(2, 2);
  DrusillaSelect<> dsBinary(5, 2);
  dsBinary.Train(fakeDataset2);

  // Now do the serialization.
  SerializeObjectAll(ds, dsXml, dsText, dsBinary);

  // Now do a search and make sure all the results are the same.
  arma::Mat<size_t> neighbors, neighborsXml, neighborsText, neighborsBinary;
  arma::mat distances, distancesXml, distancesText, distancesBinary;

  ds.Search(dataset, 3, neighbors, distances);
  dsXml.Search(dataset, 3, neighborsXml, distancesXml);
  dsText.Search(dataset, 3, neighborsText, distancesText);
  dsBinary.Search(dataset, 3, neighborsBinary, distancesBinary);

  REQUIRE(neighbors.n_rows == neighborsXml.n_rows);
  REQUIRE(neighbors.n_cols == neighborsXml.n_cols);
  REQUIRE(neighbors.n_rows == neighborsText.n_rows);
  REQUIRE(neighbors.n_cols == neighborsText.n_cols);
  REQUIRE(neighbors.n_rows == neighborsBinary.n_rows);
  REQUIRE(neighbors.n_cols == neighborsBinary.n_cols);

  REQUIRE(distances.n_rows == distancesXml.n_rows);
  REQUIRE(distances.n_cols == distancesXml.n_cols);
  REQUIRE(distances.n_rows == distancesText.n_rows);
  REQUIRE(distances.n_cols == distancesText.n_cols);
  REQUIRE(distances.n_rows == distancesBinary.n_rows);
  REQUIRE(distances.n_cols == distancesBinary.n_cols);

  for (size_t i = 0; i < neighbors.n_elem; ++i)
  {
    REQUIRE(neighbors[i] == neighborsXml[i]);
    REQUIRE(neighbors[i] == neighborsText[i]);
    REQUIRE(neighbors[i] == neighborsBinary[i]);

    REQUIRE(distances[i] == Approx(distancesXml[i]).epsilon(1e-7));
    REQUIRE(distances[i] == Approx(distancesText[i]).epsilon(1e-7));
    REQUIRE(distances[i] == Approx(distancesBinary[i]).epsilon(1e-7));
  }
}

// Make sure we can create the object with a sparse matrix.
TEST_CASE("SparseTest", "[DrusillaSelectTest]")
{
  arma::sp_mat dataset;
  dataset.sprandu(50, 1000, 0.3);

  DrusillaSelect<arma::sp_mat> ds(dataset, 5, 10);

  // Run a search.
  arma::mat distances;
  arma::Mat<size_t> neighbors;
  ds.Search(dataset, 3, neighbors, distances);

  REQUIRE(neighbors.n_cols == 1000);
  REQUIRE(neighbors.n_rows == 3);
  REQUIRE(distances.n_cols == 1000);
  REQUIRE(distances.n_rows == 3);
}
