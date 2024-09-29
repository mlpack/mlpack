/**
 * @file tests/range_search_test.cpp
 * @author Ryan Curtin
 *
 * Test file for RangeSearch<> class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/range_search.hpp>
#include <mlpack/methods/range_search/rs_model.hpp>

#include "catch.hpp"
#include "test_catch_tools.hpp"

using namespace mlpack;
using namespace std;

// Get our results into a sorted format, so we can actually then test for
// correctness.
void SortResults(const vector<vector<size_t>>& neighbors,
                 const vector<vector<double>>& distances,
                 vector<vector<pair<double, size_t>>>& output)
{
  output.resize(neighbors.size());
  for (size_t i = 0; i < neighbors.size(); ++i)
  {
    output[i].resize(neighbors[i].size());
    for (size_t j = 0; j < neighbors[i].size(); ++j)
      output[i][j] = make_pair(distances[i][j], neighbors[i][j]);

    // Now that it's constructed, sort it.
    sort(output[i].begin(), output[i].end());
  }
}

// Clean a tree's statistics.
template<typename TreeType>
void CleanTree(TreeType& node)
{
  node.Stat().LastDistance() = 0.0;

  for (size_t i = 0; i < node.NumChildren(); ++i)
    CleanTree(node.Child(i));
}

/**
 * Simple range-search test with small, synthetic dataset.  This is an
 * exhaustive test, which checks that each method for performing the calculation
 * (dual-tree, single-tree, naive) produces the correct results.  An
 * eleven-point dataset and the points within three ranges are taken.  The
 * dataset is in one dimension for simplicity -- the correct functionality of
 * distance functions is not tested here.
 */
TEST_CASE("ExhaustiveSyntheticTest", "[RangeSearchTest]")
{
  // Set up our data.
  arma::mat data(1, 11);
  data[0] = 0.05; // Row addressing is unnecessary (they are all 0).
  data[1] = 0.35;
  data[2] = 0.15;
  data[3] = 1.25;
  data[4] = 5.05;
  data[5] = -0.22;
  data[6] = -2.00;
  data[7] = -1.30;
  data[8] = 0.45;
  data[9] = 0.90;
  data[10] = 1.00;

  typedef KDTree<EuclideanDistance, RangeSearchStat, arma::mat> TreeType;

  // We will loop through three times, one for each method of performing the
  // calculation.
  std::vector<size_t> oldFromNew;
  std::vector<size_t> newFromOld;
  TreeType* tree = new TreeType(data, oldFromNew, newFromOld, 1);
  for (int i = 0; i < 3; ++i)
  {
    RangeSearch<>* rs;

    switch (i)
    {
      case 0: // Use the naive method.
        rs = new RangeSearch<>(tree->Dataset(), true);
        break;
      case 1: // Use the single-tree method.
        rs = new RangeSearch<>(tree, true);
        break;
      case 2: // Use the dual-tree method.
        rs = new RangeSearch<>(tree);
        break;
    }

    // Now perform the first calculation.  Points within 0.50.
    vector<vector<size_t>> neighbors;
    vector<vector<double>> distances;
    rs->Search(Range(0.0, sqrt(0.5)), neighbors, distances);

    // Now the exhaustive check for correctness.  This will be long.
    vector<vector<pair<double, size_t>>> sortedOutput;
    SortResults(neighbors, distances, sortedOutput);

    REQUIRE(sortedOutput[newFromOld[0]].size() == 4);
    REQUIRE(sortedOutput[newFromOld[0]][0].second == newFromOld[2]);
    REQUIRE(sortedOutput[newFromOld[0]][0].first == Approx(0.10).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[0]][1].second == newFromOld[5]);
    REQUIRE(sortedOutput[newFromOld[0]][1].first == Approx(0.27).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[0]][2].second == newFromOld[1]);
    REQUIRE(sortedOutput[newFromOld[0]][2].first == Approx(0.30).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[0]][3].second == newFromOld[8]);
    REQUIRE(sortedOutput[newFromOld[0]][3].first == Approx(0.40).epsilon(1e-7));

    // Neighbors of point 1.
    REQUIRE(sortedOutput[newFromOld[1]].size() == 6);
    REQUIRE(sortedOutput[newFromOld[1]][0].second == newFromOld[8]);
    REQUIRE(sortedOutput[newFromOld[1]][0].first == Approx(0.10).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[1]][1].second == newFromOld[2]);
    REQUIRE(sortedOutput[newFromOld[1]][1].first == Approx(0.20).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[1]][2].second == newFromOld[0]);
    REQUIRE(sortedOutput[newFromOld[1]][2].first == Approx(0.30).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[1]][3].second == newFromOld[9]);
    REQUIRE(sortedOutput[newFromOld[1]][3].first == Approx(0.55).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[1]][4].second == newFromOld[5]);
    REQUIRE(sortedOutput[newFromOld[1]][4].first == Approx(0.57).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[1]][5].second == newFromOld[10]);
    REQUIRE(sortedOutput[newFromOld[1]][5].first == Approx(0.65).epsilon(1e-7));

    // Neighbors of point 2.
    REQUIRE(sortedOutput[newFromOld[2]].size() == 4);
    REQUIRE(sortedOutput[newFromOld[2]][0].second == newFromOld[0]);
    REQUIRE(sortedOutput[newFromOld[2]][0].first == Approx(0.10).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[2]][1].second == newFromOld[1]);
    REQUIRE(sortedOutput[newFromOld[2]][1].first == Approx(0.20).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[2]][2].second == newFromOld[8]);
    REQUIRE(sortedOutput[newFromOld[2]][2].first == Approx(0.30).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[2]][3].second == newFromOld[5]);
    REQUIRE(sortedOutput[newFromOld[2]][3].first == Approx(0.37).epsilon(1e-7));

    // Neighbors of point 3.
    REQUIRE(sortedOutput[newFromOld[3]].size() == 2);
    REQUIRE(sortedOutput[newFromOld[3]][0].second == newFromOld[10]);
    REQUIRE(sortedOutput[newFromOld[3]][0].first == Approx(0.25).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[3]][1].second == newFromOld[9]);
    REQUIRE(sortedOutput[newFromOld[3]][1].first == Approx(0.35).epsilon(1e-7));

    // Neighbors of point 4.
    REQUIRE(sortedOutput[newFromOld[4]].size() == 0);

    // Neighbors of point 5.
    REQUIRE(sortedOutput[newFromOld[5]].size() == 4);
    REQUIRE(sortedOutput[newFromOld[5]][0].second == newFromOld[0]);
    REQUIRE(sortedOutput[newFromOld[5]][0].first == Approx(0.27).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[5]][1].second == newFromOld[2]);
    REQUIRE(sortedOutput[newFromOld[5]][1].first == Approx(0.37).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[5]][2].second == newFromOld[1]);
    REQUIRE(sortedOutput[newFromOld[5]][2].first == Approx(0.57).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[5]][3].second == newFromOld[8]);
    REQUIRE(sortedOutput[newFromOld[5]][3].first == Approx(0.67).epsilon(1e-7));

    // Neighbors of point 6.
    REQUIRE(sortedOutput[newFromOld[6]].size() == 1);
    REQUIRE(sortedOutput[newFromOld[6]][0].second == newFromOld[7]);
    REQUIRE(sortedOutput[newFromOld[6]][0].first == Approx(0.70).epsilon(1e-7));

    // Neighbors of point 7.
    REQUIRE(sortedOutput[newFromOld[7]].size() == 1);
    REQUIRE(sortedOutput[newFromOld[7]][0].second == newFromOld[6]);
    REQUIRE(sortedOutput[newFromOld[7]][0].first == Approx(0.70).epsilon(1e-7));

    // Neighbors of point 8.
    REQUIRE(sortedOutput[newFromOld[8]].size() == 6);
    REQUIRE(sortedOutput[newFromOld[8]][0].second == newFromOld[1]);
    REQUIRE(sortedOutput[newFromOld[8]][0].first == Approx(0.10).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[8]][1].second == newFromOld[2]);
    REQUIRE(sortedOutput[newFromOld[8]][1].first == Approx(0.30).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[8]][2].second == newFromOld[0]);
    REQUIRE(sortedOutput[newFromOld[8]][2].first == Approx(0.40).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[8]][3].second == newFromOld[9]);
    REQUIRE(sortedOutput[newFromOld[8]][3].first == Approx(0.45).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[8]][4].second == newFromOld[10]);
    REQUIRE(sortedOutput[newFromOld[8]][4].first == Approx(0.55).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[8]][5].second == newFromOld[5]);
    REQUIRE(sortedOutput[newFromOld[8]][5].first == Approx(0.67).epsilon(1e-7));

    // Neighbors of point 9.
    REQUIRE(sortedOutput[newFromOld[9]].size() == 4);
    REQUIRE(sortedOutput[newFromOld[9]][0].second == newFromOld[10]);
    REQUIRE(sortedOutput[newFromOld[9]][0].first == Approx(0.10).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[9]][1].second == newFromOld[3]);
    REQUIRE(sortedOutput[newFromOld[9]][1].first == Approx(0.35).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[9]][2].second == newFromOld[8]);
    REQUIRE(sortedOutput[newFromOld[9]][2].first == Approx(0.45).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[9]][3].second == newFromOld[1]);
    REQUIRE(sortedOutput[newFromOld[9]][3].first == Approx(0.55).epsilon(1e-7));

    // Neighbors of point 10.
    REQUIRE(sortedOutput[newFromOld[10]].size() == 4);
    REQUIRE(sortedOutput[newFromOld[10]][0].second == newFromOld[9]);
    REQUIRE(sortedOutput[newFromOld[10]][0].first ==
        Approx(0.10).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[10]][1].second ==
        newFromOld[3]);
    REQUIRE(sortedOutput[newFromOld[10]][1].first ==
        Approx(0.25).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[10]][2].second ==
        newFromOld[8]);
    REQUIRE(sortedOutput[newFromOld[10]][2].first ==
        Approx(0.55).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[10]][3].second ==
        newFromOld[1]);
    REQUIRE(sortedOutput[newFromOld[10]][3].first ==
        Approx(0.65).epsilon(1e-7));

    // Now do it again with a different range: [sqrt(0.5) 1.0].
    if (rs->ReferenceTree())
      CleanTree(*rs->ReferenceTree());
    rs->Search(Range(sqrt(0.5), 1.0), neighbors, distances);
    SortResults(neighbors, distances, sortedOutput);

    // Neighbors of point 0.
    REQUIRE(sortedOutput[newFromOld[0]].size() == 2);
    REQUIRE(sortedOutput[newFromOld[0]][0].second == newFromOld[9]);
    REQUIRE(sortedOutput[newFromOld[0]][0].first == Approx(0.85).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[0]][1].second == newFromOld[10]);
    REQUIRE(sortedOutput[newFromOld[0]][1].first == Approx(0.95).epsilon(1e-7));

    // Neighbors of point 1.
    REQUIRE(sortedOutput[newFromOld[1]].size() == 1);
    REQUIRE(sortedOutput[newFromOld[1]][0].second == newFromOld[3]);
    REQUIRE(sortedOutput[newFromOld[1]][0].first == Approx(0.90).epsilon(1e-7));

    // Neighbors of point 2.
    REQUIRE(sortedOutput[newFromOld[2]].size() == 2);
    REQUIRE(sortedOutput[newFromOld[2]][0].second == newFromOld[9]);
    REQUIRE(sortedOutput[newFromOld[2]][0].first == Approx(0.75).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[2]][1].second == newFromOld[10]);
    REQUIRE(sortedOutput[newFromOld[2]][1].first == Approx(0.85).epsilon(1e-7));

    // Neighbors of point 3.
    REQUIRE(sortedOutput[newFromOld[3]].size() == 2);
    REQUIRE(sortedOutput[newFromOld[3]][0].second == newFromOld[8]);
    REQUIRE(sortedOutput[newFromOld[3]][0].first == Approx(0.80).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[3]][1].second == newFromOld[1]);
    REQUIRE(sortedOutput[newFromOld[3]][1].first == Approx(0.90).epsilon(1e-7));

    // Neighbors of point 4.
    REQUIRE(sortedOutput[newFromOld[4]].size() == 0);

    // Neighbors of point 5.
    REQUIRE(sortedOutput[newFromOld[5]].size() == 0);

    // Neighbors of point 6.
    REQUIRE(sortedOutput[newFromOld[6]].size() == 0);

    // Neighbors of point 7.
    REQUIRE(sortedOutput[newFromOld[7]].size() == 0);

    // Neighbors of point 8.
    REQUIRE(sortedOutput[newFromOld[8]].size() == 1);
    REQUIRE(sortedOutput[newFromOld[8]][0].second == newFromOld[3]);
    REQUIRE(sortedOutput[newFromOld[8]][0].first == Approx(0.80).epsilon(1e-7));

    // Neighbors of point 9.
    REQUIRE(sortedOutput[newFromOld[9]].size() == 2);
    REQUIRE(sortedOutput[newFromOld[9]][0].second == newFromOld[2]);
    REQUIRE(sortedOutput[newFromOld[9]][0].first == Approx(0.75).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[9]][1].second == newFromOld[0]);
    REQUIRE(sortedOutput[newFromOld[9]][1].first == Approx(0.85).epsilon(1e-7));

    // Neighbors of point 10.
    REQUIRE(sortedOutput[newFromOld[10]].size() == 2);
    REQUIRE(sortedOutput[newFromOld[10]][0].second == newFromOld[2]);
    REQUIRE(sortedOutput[newFromOld[10]][0].first ==
        Approx(0.85).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[10]][1].second == newFromOld[0]);
    REQUIRE(sortedOutput[newFromOld[10]][1].first ==
        Approx(0.95).epsilon(1e-7));

    // Now do it again with a different range: [1.0 inf].
    if (rs->ReferenceTree())
      CleanTree(*rs->ReferenceTree());
    rs->Search(Range(1.0, numeric_limits<double>::infinity()), neighbors,
        distances);
    SortResults(neighbors, distances, sortedOutput);

    // Neighbors of point 0.
    REQUIRE(sortedOutput[newFromOld[0]].size() == 4);
    REQUIRE(sortedOutput[newFromOld[0]][0].second == newFromOld[3]);
    REQUIRE(sortedOutput[newFromOld[0]][0].first == Approx(1.20).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[0]][1].second == newFromOld[7]);
    REQUIRE(sortedOutput[newFromOld[0]][1].first == Approx(1.35).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[0]][2].second == newFromOld[6]);
    REQUIRE(sortedOutput[newFromOld[0]][2].first == Approx(2.05).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[0]][3].second == newFromOld[4]);
    REQUIRE(sortedOutput[newFromOld[0]][3].first == Approx(5.00).epsilon(1e-7));

    // Neighbors of point 1.
    REQUIRE(sortedOutput[newFromOld[1]].size() == 3);
    REQUIRE(sortedOutput[newFromOld[1]][0].second == newFromOld[7]);
    REQUIRE(sortedOutput[newFromOld[1]][0].first == Approx(1.65).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[1]][1].second == newFromOld[6]);
    REQUIRE(sortedOutput[newFromOld[1]][1].first == Approx(2.35).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[1]][2].second == newFromOld[4]);
    REQUIRE(sortedOutput[newFromOld[1]][2].first == Approx(4.70).epsilon(1e-7));

    // Neighbors of point 2.
    REQUIRE(sortedOutput[newFromOld[2]].size() == 4);
    REQUIRE(sortedOutput[newFromOld[2]][0].second == newFromOld[3]);
    REQUIRE(sortedOutput[newFromOld[2]][0].first == Approx(1.10).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[2]][1].second == newFromOld[7]);
    REQUIRE(sortedOutput[newFromOld[2]][1].first == Approx(1.45).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[2]][2].second == newFromOld[6]);
    REQUIRE(sortedOutput[newFromOld[2]][2].first == Approx(2.15).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[2]][3].second == newFromOld[4]);
    REQUIRE(sortedOutput[newFromOld[2]][3].first == Approx(4.90).epsilon(1e-7));

    // Neighbors of point 3.
    REQUIRE(sortedOutput[newFromOld[3]].size() == 6);
    REQUIRE(sortedOutput[newFromOld[3]][0].second == newFromOld[2]);
    REQUIRE(sortedOutput[newFromOld[3]][0].first == Approx(1.10).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[3]][1].second == newFromOld[0]);
    REQUIRE(sortedOutput[newFromOld[3]][1].first == Approx(1.20).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[3]][2].second == newFromOld[5]);
    REQUIRE(sortedOutput[newFromOld[3]][2].first == Approx(1.47).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[3]][3].second == newFromOld[7]);
    REQUIRE(sortedOutput[newFromOld[3]][3].first == Approx(2.55).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[3]][4].second == newFromOld[6]);
    REQUIRE(sortedOutput[newFromOld[3]][4].first == Approx(3.25).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[3]][5].second == newFromOld[4]);
    REQUIRE(sortedOutput[newFromOld[3]][5].first == Approx(3.80).epsilon(1e-7));

    // Neighbors of point 4.
    REQUIRE(sortedOutput[newFromOld[4]].size() == 10);
    REQUIRE(sortedOutput[newFromOld[4]][0].second == newFromOld[3]);
    REQUIRE(sortedOutput[newFromOld[4]][0].first == Approx(3.80).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[4]][1].second == newFromOld[10]);
    REQUIRE(sortedOutput[newFromOld[4]][1].first == Approx(4.05).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[4]][2].second == newFromOld[9]);
    REQUIRE(sortedOutput[newFromOld[4]][2].first == Approx(4.15).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[4]][3].second == newFromOld[8]);
    REQUIRE(sortedOutput[newFromOld[4]][3].first == Approx(4.60).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[4]][4].second == newFromOld[1]);
    REQUIRE(sortedOutput[newFromOld[4]][4].first == Approx(4.70).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[4]][5].second == newFromOld[2]);
    REQUIRE(sortedOutput[newFromOld[4]][5].first == Approx(4.90).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[4]][6].second == newFromOld[0]);
    REQUIRE(sortedOutput[newFromOld[4]][6].first == Approx(5.00).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[4]][7].second == newFromOld[5]);
    REQUIRE(sortedOutput[newFromOld[4]][7].first == Approx(5.27).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[4]][8].second == newFromOld[7]);
    REQUIRE(sortedOutput[newFromOld[4]][8].first == Approx(6.35).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[4]][9].second == newFromOld[6]);
    REQUIRE(sortedOutput[newFromOld[4]][9].first == Approx(7.05).epsilon(1e-7));

    // Neighbors of point 5.
    REQUIRE(sortedOutput[newFromOld[5]].size() == 6);
    REQUIRE(sortedOutput[newFromOld[5]][0].second == newFromOld[7]);
    REQUIRE(sortedOutput[newFromOld[5]][0].first == Approx(1.08).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[5]][1].second == newFromOld[9]);
    REQUIRE(sortedOutput[newFromOld[5]][1].first == Approx(1.12).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[5]][2].second == newFromOld[10]);
    REQUIRE(sortedOutput[newFromOld[5]][2].first == Approx(1.22).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[5]][3].second == newFromOld[3]);
    REQUIRE(sortedOutput[newFromOld[5]][3].first == Approx(1.47).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[5]][4].second == newFromOld[6]);
    REQUIRE(sortedOutput[newFromOld[5]][4].first == Approx(1.78).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[5]][5].second == newFromOld[4]);
    REQUIRE(sortedOutput[newFromOld[5]][5].first == Approx(5.27).epsilon(1e-7));

    // Neighbors of point 6.
    REQUIRE(sortedOutput[newFromOld[6]].size() == 9);
    REQUIRE(sortedOutput[newFromOld[6]][0].second == newFromOld[5]);
    REQUIRE(sortedOutput[newFromOld[6]][0].first == Approx(1.78).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[6]][1].second == newFromOld[0]);
    REQUIRE(sortedOutput[newFromOld[6]][1].first == Approx(2.05).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[6]][2].second == newFromOld[2]);
    REQUIRE(sortedOutput[newFromOld[6]][2].first == Approx(2.15).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[6]][3].second == newFromOld[1]);
    REQUIRE(sortedOutput[newFromOld[6]][3].first == Approx(2.35).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[6]][4].second == newFromOld[8]);
    REQUIRE(sortedOutput[newFromOld[6]][4].first == Approx(2.45).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[6]][5].second == newFromOld[9]);
    REQUIRE(sortedOutput[newFromOld[6]][5].first == Approx(2.90).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[6]][6].second == newFromOld[10]);
    REQUIRE(sortedOutput[newFromOld[6]][6].first == Approx(3.00).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[6]][7].second == newFromOld[3]);
    REQUIRE(sortedOutput[newFromOld[6]][7].first == Approx(3.25).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[6]][8].second == newFromOld[4]);
    REQUIRE(sortedOutput[newFromOld[6]][8].first == Approx(7.05).epsilon(1e-7));

    // Neighbors of point 7.
    REQUIRE(sortedOutput[newFromOld[7]].size() == 9);
    REQUIRE(sortedOutput[newFromOld[7]][0].second == newFromOld[5]);
    REQUIRE(sortedOutput[newFromOld[7]][0].first == Approx(1.08).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[7]][1].second == newFromOld[0]);
    REQUIRE(sortedOutput[newFromOld[7]][1].first == Approx(1.35).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[7]][2].second == newFromOld[2]);
    REQUIRE(sortedOutput[newFromOld[7]][2].first == Approx(1.45).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[7]][3].second == newFromOld[1]);
    REQUIRE(sortedOutput[newFromOld[7]][3].first == Approx(1.65).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[7]][4].second == newFromOld[8]);
    REQUIRE(sortedOutput[newFromOld[7]][4].first == Approx(1.75).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[7]][5].second == newFromOld[9]);
    REQUIRE(sortedOutput[newFromOld[7]][5].first == Approx(2.20).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[7]][6].second == newFromOld[10]);
    REQUIRE(sortedOutput[newFromOld[7]][6].first == Approx(2.30).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[7]][7].second == newFromOld[3]);
    REQUIRE(sortedOutput[newFromOld[7]][7].first == Approx(2.55).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[7]][8].second == newFromOld[4]);
    REQUIRE(sortedOutput[newFromOld[7]][8].first == Approx(6.35).epsilon(1e-7));

    // Neighbors of point 8.
    REQUIRE(sortedOutput[newFromOld[8]].size() == 3);
    REQUIRE(sortedOutput[newFromOld[8]][0].second == newFromOld[7]);
    REQUIRE(sortedOutput[newFromOld[8]][0].first == Approx(1.75).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[8]][1].second == newFromOld[6]);
    REQUIRE(sortedOutput[newFromOld[8]][1].first == Approx(2.45).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[8]][2].second == newFromOld[4]);
    REQUIRE(sortedOutput[newFromOld[8]][2].first == Approx(4.60).epsilon(1e-7));

    // Neighbors of point 9.
    REQUIRE(sortedOutput[newFromOld[9]].size() == 4);
    REQUIRE(sortedOutput[newFromOld[9]][0].second == newFromOld[5]);
    REQUIRE(sortedOutput[newFromOld[9]][0].first == Approx(1.12).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[9]][1].second == newFromOld[7]);
    REQUIRE(sortedOutput[newFromOld[9]][1].first == Approx(2.20).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[9]][2].second == newFromOld[6]);
    REQUIRE(sortedOutput[newFromOld[9]][2].first == Approx(2.90).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[9]][3].second == newFromOld[4]);
    REQUIRE(sortedOutput[newFromOld[9]][3].first == Approx(4.15).epsilon(1e-7));

    // Neighbors of point 10.
    REQUIRE(sortedOutput[newFromOld[10]].size() == 4);
    REQUIRE(sortedOutput[newFromOld[10]][0].second == newFromOld[5]);
    REQUIRE(sortedOutput[newFromOld[10]][0].first ==
        Approx(1.22).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[10]][1].second ==
        newFromOld[7]);
    REQUIRE(sortedOutput[newFromOld[10]][1].first ==
        Approx(2.30).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[10]][2].second ==
        newFromOld[6]);
    REQUIRE(sortedOutput[newFromOld[10]][2].first ==
        Approx(3.00).epsilon(1e-7));
    REQUIRE(sortedOutput[newFromOld[10]][3].second ==
        newFromOld[4]);
    REQUIRE(sortedOutput[newFromOld[10]][3].first ==
        Approx(4.05).epsilon(1e-7));

    // Clean the memory.
    delete rs;
  }

  delete tree;
}

/**
 * Test the dual-tree range search method with the naive method.  This
 * uses both a query and reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
TEST_CASE("DualTreeVsNaive1", "[RangeSearchTest]")
{
  arma::mat dataForTree;

  // Hard-coded filename: bad!
  if (!data::Load("test_data_3_1000.csv", dataForTree))
    FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Set up matrices to work with.
  arma::mat dualQuery(dataForTree);
  arma::mat dualReferences(dataForTree);
  arma::mat naiveQuery(dataForTree);
  arma::mat naiveReferences(dataForTree);

  RangeSearch<> rs(dualReferences);

  RangeSearch<> naive(naiveReferences, true);

  vector<vector<size_t>> neighborsTree;
  vector<vector<double>> distancesTree;
  rs.Search(dualQuery, Range(0.25, 1.05), neighborsTree, distancesTree);
  vector<vector<pair<double, size_t>>> sortedTree;
  SortResults(neighborsTree, distancesTree, sortedTree);

  vector<vector<size_t>> neighborsNaive;
  vector<vector<double>> distancesNaive;
  naive.Search(naiveQuery, Range(0.25, 1.05), neighborsNaive, distancesNaive);
  vector<vector<pair<double, size_t>>> sortedNaive;
  SortResults(neighborsNaive, distancesNaive, sortedNaive);

  for (size_t i = 0; i < sortedTree.size(); ++i)
  {
    REQUIRE(sortedTree[i].size() == sortedNaive[i].size());

    for (size_t j = 0; j < sortedTree[i].size(); ++j)
    {
      REQUIRE(sortedTree[i][j].second == sortedNaive[i][j].second);
      REQUIRE(sortedTree[i][j].first ==
          Approx(sortedNaive[i][j].first).epsilon(1e-7));
    }
  }
}

/**
 * Test the dual-tree range search method with the naive method.  This uses
 * only a reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
TEST_CASE("DualTreeVsNaive2", "[RangeSearchTest]")
{
  arma::mat dataForTree;

  // Hard-coded filename: bad!
  // Code duplication: also bad!
  if (!data::Load("test_data_3_1000.csv", dataForTree))
    FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Set up matrices to work with.
  arma::mat dualQuery(dataForTree);
  arma::mat naiveQuery(dataForTree);

  RangeSearch<> rs(dualQuery);

  // Set naive mode.
  RangeSearch<> naive(naiveQuery, true);

  vector<vector<size_t>> neighborsTree;
  vector<vector<double>> distancesTree;
  rs.Search(Range(0.25, 1.05), neighborsTree, distancesTree);
  vector<vector<pair<double, size_t>>> sortedTree;
  SortResults(neighborsTree, distancesTree, sortedTree);

  vector<vector<size_t>> neighborsNaive;
  vector<vector<double>> distancesNaive;
  naive.Search(Range(0.25, 1.05), neighborsNaive, distancesNaive);
  vector<vector<pair<double, size_t>>> sortedNaive;
  SortResults(neighborsNaive, distancesNaive, sortedNaive);

  for (size_t i = 0; i < sortedTree.size(); ++i)
  {
    REQUIRE(sortedTree[i].size() == sortedNaive[i].size());

    for (size_t j = 0; j < sortedTree[i].size(); ++j)
    {
      REQUIRE(sortedTree[i][j].second == sortedNaive[i][j].second);
      REQUIRE(sortedTree[i][j].first ==
          Approx(sortedNaive[i][j].first).epsilon(1e-7));
    }
  }
}

/**
 * Test the single-tree range search method with the naive method.  This
 * uses only a reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
TEST_CASE("SingleTreeVsNaive", "[RangeSearchTest]")
{
  arma::mat dataForTree;

  // Hard-coded filename: bad!
  // Code duplication: also bad!
  if (!data::Load("test_data_3_1000.csv", dataForTree))
    FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Set up matrices to work with (may not be necessary with no ALIAS_MATRIX?).
  arma::mat singleQuery(dataForTree);
  arma::mat naiveQuery(dataForTree);

  RangeSearch<> single(singleQuery, false, true);

  // Set up computation for naive mode.
  RangeSearch<> naive(naiveQuery, true);

  vector<vector<size_t>> neighborsSingle;
  vector<vector<double>> distancesSingle;
  single.Search(Range(0.25, 1.05), neighborsSingle, distancesSingle);
  vector<vector<pair<double, size_t>>> sortedTree;
  SortResults(neighborsSingle, distancesSingle, sortedTree);

  vector<vector<size_t>> neighborsNaive;
  vector<vector<double>> distancesNaive;
  naive.Search(Range(0.25, 1.05), neighborsNaive, distancesNaive);
  vector<vector<pair<double, size_t>>> sortedNaive;
  SortResults(neighborsNaive, distancesNaive, sortedNaive);

  for (size_t i = 0; i < sortedTree.size(); ++i)
  {
    REQUIRE(sortedTree[i].size() == sortedNaive[i].size());

    for (size_t j = 0; j < sortedTree[i].size(); ++j)
    {
      REQUIRE(sortedTree[i][j].second == sortedNaive[i][j].second);
      REQUIRE(sortedTree[i][j].first ==
          Approx(sortedNaive[i][j].first).epsilon(1e-7));
    }
  }
}

/**
 * Ensure that dual tree range search with cover trees works by comparing
 * with the kd-tree implementation.
 */
TEST_CASE("RSCoverTreeTest", "[RangeSearchTest]")
{
  arma::mat data;
  data.randu(8, 1000); // 1000 points in 8 dimensions.

  // Set up cover tree range search.
  RangeSearch<EuclideanDistance, arma::mat, StandardCoverTree>
      coversearch(data);

  // Four trials with different ranges.
  for (size_t r = 0; r < 4; ++r)
  {
    // Set up kd-tree range search.
    RangeSearch<> kdsearch(data);

    Range range;
    switch (r)
    {
      case 0:
        // Includes zero distance.
        range = Range(0.0, 0.75);
        break;
      case 1:
        // A bounded range on both sides.
        range = Range(0.5, 1.5);
        break;
      case 2:
        // A range with no upper bound.
        range = Range(0.8, DBL_MAX);
        break;
      case 3:
        // A range which should have no results.
        range = Range(15.6, 15.7);
        break;
    }

    // Results for kd-tree search.
    vector<vector<size_t>> kdNeighbors;
    vector<vector<double>> kdDistances;

    // Results for cover tree search.
    vector<vector<size_t>> coverNeighbors;
    vector<vector<double>> coverDistances;

    // Clean the tree statistics.
    CleanTree(*coversearch.ReferenceTree());

    // Run the searches.
    kdsearch.Search(range, kdNeighbors, kdDistances);
    coversearch.Search(range, coverNeighbors, coverDistances);

    // Sort before comparison.
    vector<vector<pair<double, size_t>>> kdSorted;
    vector<vector<pair<double, size_t>>> coverSorted;
    SortResults(kdNeighbors, kdDistances, kdSorted);
    SortResults(coverNeighbors, coverDistances, coverSorted);

    // Now compare the results.
    for (size_t i = 0; i < kdSorted.size(); ++i)
    {
      for (size_t j = 0; j < kdSorted[i].size(); ++j)
      {
        REQUIRE(kdSorted[i][j].second == coverSorted[i][j].second);
        REQUIRE(kdSorted[i][j].first ==
            Approx(coverSorted[i][j].first).epsilon(1e-7));
      }
      REQUIRE(kdSorted[i].size() == coverSorted[i].size());
    }
  }
}

/**
 * Ensure that dual tree range search with cover trees works when using
 * two datasets.
 */
TEST_CASE("CoverTreeTwoDatasetsTest", "[RangeSearchTest]")
{
  arma::mat data;
  data.randu(8, 1000); // 1000 points in 8 dimensions.
  arma::mat queries;
  queries.randu(8, 350); // 350 points in 8 dimensions.

  // Set up cover tree range search.
  RangeSearch<EuclideanDistance, arma::mat, StandardCoverTree>
      coversearch(data);

  // Four trials with different ranges.
  for (size_t r = 0; r < 4; ++r)
  {
    // Set up kd-tree range search.  We don't have an easy way to rebuild the
    // tree, so we'll just reinstantiate it here each loop time.
    RangeSearch<> kdsearch(data);

    Range range;
    switch (r)
    {
      case 0:
        // Includes zero distance.
        range = Range(0.0, 0.75);
        break;
      case 1:
        // A bounded range on both sides.
        range = Range(0.85, 1.05);
        break;
      case 2:
        // A range with no upper bound.
        range = Range(1.35, DBL_MAX);
        break;
      case 3:
        // A range which should have no results.
        range = Range(15.6, 15.7);
        break;
    }

    // Results for kd-tree search.
    vector<vector<size_t>> kdNeighbors;
    vector<vector<double>> kdDistances;

    // Results for cover tree search.
    vector<vector<size_t>> coverNeighbors;
    vector<vector<double>> coverDistances;

    // Clean the trees.
    CleanTree(*coversearch.ReferenceTree());

    // Run the searches.
    coversearch.Search(queries, range, coverNeighbors, coverDistances);
    kdsearch.Search(queries, range, kdNeighbors, kdDistances);

    // Sort before comparison.
    vector<vector<pair<double, size_t>>> kdSorted;
    vector<vector<pair<double, size_t>>> coverSorted;
    SortResults(kdNeighbors, kdDistances, kdSorted);
    SortResults(coverNeighbors, coverDistances, coverSorted);

    // Now compare the results.
    for (size_t i = 0; i < kdSorted.size(); ++i)
    {
      for (size_t j = 0; j < kdSorted[i].size(); ++j)
      {
        REQUIRE(kdSorted[i][j].second == coverSorted[i][j].second);
        REQUIRE(kdSorted[i][j].first ==
            Approx(coverSorted[i][j].first).epsilon(1e-7));
      }
      REQUIRE(kdSorted[i].size() == coverSorted[i].size());
    }
  }
}

/**
 * Ensure that single-tree cover tree range search works.
 */
TEST_CASE("CoverTreeSingleTreeTest", "[RangeSearchTest]")
{
  arma::mat data;
  data.randu(8, 1000); // 1000 points in 8 dimensions.

  // Set up cover tree range search.
  RangeSearch<EuclideanDistance, arma::mat, StandardCoverTree>
      coversearch(data, false, true);

  // Four trials with different ranges.
  for (size_t r = 0; r < 4; ++r)
  {
    // Set up kd-tree range search.
    RangeSearch<> kdsearch(data);

    Range range;
    switch (r)
    {
      case 0:
        // Includes zero distance.
        range = Range(0.0, 0.75);
        break;
      case 1:
        // A bounded range on both sides.
        range = Range(0.5, 1.5);
        break;
      case 2:
        // A range with no upper bound.
        range = Range(0.8, DBL_MAX);
        break;
      case 3:
        // A range which should have no results.
        range = Range(15.6, 15.7);
        break;
    }

    // Results for kd-tree search.
    vector<vector<size_t>> kdNeighbors;
    vector<vector<double>> kdDistances;

    // Results for cover tree search.
    vector<vector<size_t>> coverNeighbors;
    vector<vector<double>> coverDistances;

    // Clean the tree statistics.
    CleanTree(*coversearch.ReferenceTree());

    // Run the searches.
    kdsearch.Search(range, kdNeighbors, kdDistances);
    coversearch.Search(range, coverNeighbors, coverDistances);

    // Sort before comparison.
    vector<vector<pair<double, size_t>>> kdSorted;
    vector<vector<pair<double, size_t>>> coverSorted;
    SortResults(kdNeighbors, kdDistances, kdSorted);
    SortResults(coverNeighbors, coverDistances, coverSorted);

    // Now compare the results.
    for (size_t i = 0; i < kdSorted.size(); ++i)
    {
      for (size_t j = 0; j < kdSorted[i].size(); ++j)
      {
        REQUIRE(kdSorted[i][j].second == coverSorted[i][j].second);
        REQUIRE(kdSorted[i][j].first ==
            Approx(coverSorted[i][j].first).epsilon(1e-7));
      }
      REQUIRE(kdSorted[i].size() == coverSorted[i].size());
    }
  }
}

/**
 * Ensure that single-tree ball tree range search works.
 */
TEST_CASE("SingleBallTreeTest", "[RangeSearchTest]")
{
  arma::mat data;
  data.randu(8, 1000); // 1000 points in 8 dimensions.

  // Set up ball tree range search.
  RangeSearch<EuclideanDistance, arma::mat, BallTree> ballsearch(data, false,
      true);

  // Four trials with different ranges.
  for (size_t r = 0; r < 4; ++r)
  {
    // Set up kd-tree range search.
    RangeSearch<> kdsearch(data);

    Range range;
    switch (r)
    {
      case 0:
        // Includes zero distance.
        range = Range(0.0, 0.75);
        break;
      case 1:
        // A bounded range on both sides.
        range = Range(0.5, 1.5);
        break;
      case 2:
        // A range with no upper bound.
        range = Range(0.8, DBL_MAX);
        break;
      case 3:
        // A range which should have no results.
        range = Range(15.6, 15.7);
        break;
    }

    // Results for kd-tree search.
    vector<vector<size_t>> kdNeighbors;
    vector<vector<double>> kdDistances;

    // Results for ball tree search.
    vector<vector<size_t>> ballNeighbors;
    vector<vector<double>> ballDistances;

    // Clean the tree statistics.
    CleanTree(*ballsearch.ReferenceTree());

    // Run the searches.
    kdsearch.Search(range, kdNeighbors, kdDistances);
    ballsearch.Search(range, ballNeighbors, ballDistances);

    // Sort before comparison.
    vector<vector<pair<double, size_t>>> kdSorted;
    vector<vector<pair<double, size_t>>> ballSorted;
    SortResults(kdNeighbors, kdDistances, kdSorted);
    SortResults(ballNeighbors, ballDistances, ballSorted);

    // Now compare the results.
    for (size_t i = 0; i < kdSorted.size(); ++i)
    {
      for (size_t j = 0; j < kdSorted[i].size(); ++j)
      {
        REQUIRE(kdSorted[i][j].second == ballSorted[i][j].second);
        REQUIRE(kdSorted[i][j].first ==
            Approx(ballSorted[i][j].first).epsilon(1e-7));
      }
      REQUIRE(kdSorted[i].size() == ballSorted[i].size());
    }
  }
}

/**
 * Ensure that dual tree range search with ball trees works by comparing
 * with the kd-tree implementation.
 */
TEST_CASE("DualBallTreeTest", "[RangeSearchTest]")
{
  arma::mat data;
  data.randu(8, 1000); // 1000 points in 8 dimensions.

  // Set up ball tree range search.
  RangeSearch<EuclideanDistance, arma::mat, BallTree> ballsearch(data);

  // Four trials with different ranges.
  for (size_t r = 0; r < 4; ++r)
  {
    // Set up kd-tree range search.
    RangeSearch<> kdsearch(data);

    Range range;
    switch (r)
    {
      case 0:
        // Includes zero distance.
        range = Range(0.0, 0.75);
        break;
      case 1:
        // A bounded range on both sides.
        range = Range(0.5, 1.5);
        break;
      case 2:
        // A range with no upper bound.
        range = Range(0.8, DBL_MAX);
        break;
      case 3:
        // A range which should have no results.
        range = Range(15.6, 15.7);
        break;
    }

    // Results for kd-tree search.
    vector<vector<size_t>> kdNeighbors;
    vector<vector<double>> kdDistances;

    // Results for ball tree search.
    vector<vector<size_t>> ballNeighbors;
    vector<vector<double>> ballDistances;

    // Clean the tree statistics.
    CleanTree(*ballsearch.ReferenceTree());

    // Run the searches.
    kdsearch.Search(range, kdNeighbors, kdDistances);
    ballsearch.Search(range, ballNeighbors, ballDistances);

    // Sort before comparison.
    vector<vector<pair<double, size_t>>> kdSorted;
    vector<vector<pair<double, size_t>>> ballSorted;
    SortResults(kdNeighbors, kdDistances, kdSorted);
    SortResults(ballNeighbors, ballDistances, ballSorted);

    // Now compare the results.
    for (size_t i = 0; i < kdSorted.size(); ++i)
    {
      for (size_t j = 0; j < kdSorted[i].size(); ++j)
      {
        REQUIRE(kdSorted[i][j].second == ballSorted[i][j].second);
        REQUIRE(kdSorted[i][j].first ==
            Approx(ballSorted[i][j].first).epsilon(1e-7));
      }
      REQUIRE(kdSorted[i].size() == ballSorted[i].size());
    }
  }
}

/**
 * Ensure that dual tree range search with ball trees works when using
 * two datasets.
 */
TEST_CASE("DualBallTreeTest2", "[RangeSearchTest]")
{
  arma::mat data;
  data.randu(8, 1000); // 1000 points in 8 dimensions.

  arma::mat queries;
  queries.randu(8, 350); // 350 points in 8 dimensions.

  // Set up ball tree range search.
  RangeSearch<EuclideanDistance, arma::mat, BallTree> ballsearch(data);

  // Four trials with different ranges.
  for (size_t r = 0; r < 4; ++r)
  {
    // Set up kd-tree range search.  We don't have an easy way to rebuild the
    // tree, so we'll just reinstantiate it here each loop time.
    RangeSearch<> kdsearch(data);

    Range range;
    switch (r)
    {
      case 0:
        // Includes zero distance.
        range = Range(0.0, 0.75);
        break;
      case 1:
        // A bounded range on both sides.
        range = Range(0.85, 1.05);
        break;
      case 2:
        // A range with no upper bound.
        range = Range(1.35, DBL_MAX);
        break;
      case 3:
        // A range which should have no results.
        range = Range(15.6, 15.7);
        break;
    }

    // Results for kd-tree search.
    vector<vector<size_t>> kdNeighbors;
    vector<vector<double>> kdDistances;

    // Results for ball tree search.
    vector<vector<size_t>> ballNeighbors;
    vector<vector<double>> ballDistances;

    // Clean the trees.
    CleanTree(*ballsearch.ReferenceTree());

    // Run the searches.
    ballsearch.Search(queries, range, ballNeighbors, ballDistances);
    kdsearch.Search(queries, range, kdNeighbors, kdDistances);

    // Sort before comparison.
    vector<vector<pair<double, size_t>>> kdSorted;
    vector<vector<pair<double, size_t>>> ballSorted;
    SortResults(kdNeighbors, kdDistances, kdSorted);
    SortResults(ballNeighbors, ballDistances, ballSorted);

    // Now compare the results.
    for (size_t i = 0; i < kdSorted.size(); ++i)
    {
      REQUIRE(kdSorted[i].size() == ballSorted[i].size());
      for (size_t j = 0; j < kdSorted[i].size(); ++j)
      {
        REQUIRE(kdSorted[i][j].second == ballSorted[i][j].second);
        REQUIRE(kdSorted[i][j].first ==
            Approx(ballSorted[i][j].first).epsilon(1e-7));
      }
    }
  }
}

/**
 * Make sure that no results are returned when we build a range search object
 * with no reference set.
 */
TEST_CASE("EmptySearchTest", "[RangeSearchTest]")
{
  RangeSearch<EuclideanDistance, arma::mat, KDTree> rs;

  vector<vector<size_t>> neighbors;
  vector<vector<double>> distances;

  rs.Search(Range(0.0, 10.0), neighbors, distances);

  REQUIRE(neighbors.size() == 0);
  REQUIRE(distances.size() == 0);

  // Now check with a query set.
  arma::mat querySet = arma::randu<arma::mat>(3, 100);

  REQUIRE_THROWS_AS(rs.Search(querySet, Range(0.0, 10.0), neighbors,
      distances), std::invalid_argument);
}

/**
 * Make sure things work right after Train() is called.
 */
TEST_CASE("RangeSearchTrainTest", "[RangeSearchTest]")
{
  RangeSearch<> empty;

  arma::mat dataset = arma::randu<arma::mat>(5, 100);
  RangeSearch<> baseline(dataset);

  vector<vector<size_t>> neighbors, baselineNeighbors;
  vector<vector<double>> distances, baselineDistances;

  empty.Train(dataset);

  empty.Search(Range(0.5, 0.7), neighbors, distances);
  baseline.Search(Range(0.5, 0.7), baselineNeighbors, baselineDistances);

  REQUIRE(neighbors.size() == baselineNeighbors.size());
  REQUIRE(distances.size() == baselineDistances.size());

  // Sort the results before comparing.
  vector<vector<pair<double, size_t>>> sorted;
  vector<vector<pair<double, size_t>>> baselineSorted;
  SortResults(neighbors, distances, sorted);
  SortResults(baselineNeighbors, baselineDistances, baselineSorted);

  for (size_t i = 0; i < sorted.size(); ++i)
  {
    REQUIRE(sorted[i].size() == baselineSorted[i].size());
    for (size_t j = 0; j < sorted[i].size(); ++j)
    {
      REQUIRE(sorted[i][j].second == baselineSorted[i][j].second);
      REQUIRE(sorted[i][j].first ==
          Approx(baselineSorted[i][j].first).epsilon(1e-7));
    }
  }
}

/**
 * Test training when a tree is given.
 */
TEST_CASE("TrainTreeTest", "[RangeSearchTest]")
{
  // Avoid mappings by using the cover tree.
  typedef RangeSearch<EuclideanDistance, arma::mat, StandardCoverTree> RSType;
  RSType empty;

  arma::mat dataset = arma::randu<arma::mat>(5, 100);
  RSType baseline(dataset);

  vector<vector<size_t>> neighbors, baselineNeighbors;
  vector<vector<double>> distances, baselineDistances;

  RSType::Tree tree(dataset);
  empty.Train(&tree);

  empty.Search(Range(0.5, 0.7), neighbors, distances);
  baseline.Search(Range(0.5, 0.7), baselineNeighbors, baselineDistances);

  REQUIRE(neighbors.size() == baselineNeighbors.size());
  REQUIRE(distances.size() == baselineDistances.size());

  // Sort the results before comparing.
  vector<vector<pair<double, size_t>>> sorted;
  vector<vector<pair<double, size_t>>> baselineSorted;
  SortResults(neighbors, distances, sorted);
  SortResults(baselineNeighbors, baselineDistances, baselineSorted);

  for (size_t i = 0; i < sorted.size(); ++i)
  {
    REQUIRE(sorted[i].size() == baselineSorted[i].size());
    for (size_t j = 0; j < sorted[i].size(); ++j)
    {
      REQUIRE(sorted[i][j].second == baselineSorted[i][j].second);
      REQUIRE(sorted[i][j].first ==
          Approx(baselineSorted[i][j].first).epsilon(1e-7));
    }
  }
}

/**
 * Test that training with a tree throws an exception when in naive mode.
 */
TEST_CASE("NaiveTrainTreeTest", "[RangeSearchTest]")
{
  RangeSearch<> empty(true);

  arma::mat dataset = arma::randu<arma::mat>(5, 100);
  RangeSearch<>::Tree tree(dataset);

  REQUIRE_THROWS_AS(empty.Train(&tree), std::invalid_argument);
}

/**
 * Test that the move constructor works.
 */
TEST_CASE("MoveConstructorMatrixTest", "[RangeSearchTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(3, 100);
  arma::mat copy(dataset);

  RangeSearch<> movers(std::move(copy));
  RangeSearch<> rs(dataset);

  REQUIRE(copy.n_elem == 0);
  REQUIRE(movers.ReferenceSet().n_rows == 3);
  REQUIRE(movers.ReferenceSet().n_cols == 100);

  vector<vector<size_t>> moveNeighbors, neighbors;
  vector<vector<double>> moveDistances, distances;

  movers.Search(Range(0.5, 0.7), moveNeighbors, moveDistances);
  rs.Search(Range(0.5, 0.7), neighbors, distances);

  REQUIRE(neighbors.size() == moveNeighbors.size());
  REQUIRE(distances.size() == moveDistances.size());

  // Sort the results before comparing.
  vector<vector<pair<double, size_t>>> sorted;
  vector<vector<pair<double, size_t>>> moveSorted;
  SortResults(neighbors, distances, sorted);
  SortResults(moveNeighbors, moveDistances, moveSorted);

  for (size_t i = 0; i < sorted.size(); ++i)
  {
    REQUIRE(sorted[i].size() == moveSorted[i].size());
    for (size_t j = 0; j < sorted[i].size(); ++j)
    {
      REQUIRE(sorted[i][j].second == moveSorted[i][j].second);
      REQUIRE(sorted[i][j].first ==
          Approx(moveSorted[i][j].first).epsilon(1e-7));
    }
  }
}

/**
 * Test that the std::move() Train() function works.
 */
TEST_CASE("MoveTrainTest", "[RangeSearchTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(3, 100);
  arma::mat copy(dataset);

  RangeSearch<> movers;
  movers.Train(std::move(copy));
  RangeSearch<> rs(dataset);

  REQUIRE(copy.n_elem == 0);
  REQUIRE(movers.ReferenceSet().n_rows == 3);
  REQUIRE(movers.ReferenceSet().n_cols == 100);

  vector<vector<size_t>> moveNeighbors, neighbors;
  vector<vector<double>> moveDistances, distances;

  movers.Search(Range(0.5, 0.7), moveNeighbors, moveDistances);
  rs.Search(Range(0.5, 0.7), neighbors, distances);

  REQUIRE(neighbors.size() == moveNeighbors.size());
  REQUIRE(distances.size() == moveDistances.size());

  // Sort the results before comparing.
  vector<vector<pair<double, size_t>>> sorted;
  vector<vector<pair<double, size_t>>> moveSorted;
  SortResults(neighbors, distances, sorted);
  SortResults(moveNeighbors, moveDistances, moveSorted);

  for (size_t i = 0; i < sorted.size(); ++i)
  {
    REQUIRE(sorted[i].size() == moveSorted[i].size());
    for (size_t j = 0; j < sorted[i].size(); ++j)
    {
      REQUIRE(sorted[i][j].second == moveSorted[i][j].second);
      REQUIRE(sorted[i][j].first ==
          Approx(moveSorted[i][j].first).epsilon(1e-7));
    }
  }
}

TEST_CASE("RSModelTest", "[RangeSearchTest]")
{
  // Ensure that we can build an RSModel and get correct results.
  arma::mat queryData = arma::randu<arma::mat>(10, 50);
  arma::mat referenceData = arma::randu<arma::mat>(10, 200);

  // Build all the possible models.
  RSModel models[28];
  models[0] = RSModel(RSModel::TreeTypes::KD_TREE, true);
  models[1] = RSModel(RSModel::TreeTypes::KD_TREE, false);
  models[2] = RSModel(RSModel::TreeTypes::COVER_TREE, true);
  models[3] = RSModel(RSModel::TreeTypes::COVER_TREE, false);
  models[4] = RSModel(RSModel::TreeTypes::R_TREE, true);
  models[5] = RSModel(RSModel::TreeTypes::R_TREE, false);
  models[6] = RSModel(RSModel::TreeTypes::R_STAR_TREE, true);
  models[7] = RSModel(RSModel::TreeTypes::R_STAR_TREE, false);
  models[8] = RSModel(RSModel::TreeTypes::X_TREE, true);
  models[9] = RSModel(RSModel::TreeTypes::X_TREE, false);
  models[10] = RSModel(RSModel::TreeTypes::BALL_TREE, true);
  models[11] = RSModel(RSModel::TreeTypes::BALL_TREE, false);
  models[12] = RSModel(RSModel::TreeTypes::HILBERT_R_TREE, true);
  models[13] = RSModel(RSModel::TreeTypes::HILBERT_R_TREE, false);
  models[14] = RSModel(RSModel::TreeTypes::R_PLUS_TREE, true);
  models[15] = RSModel(RSModel::TreeTypes::R_PLUS_TREE, false);
  models[16] = RSModel(RSModel::TreeTypes::R_PLUS_PLUS_TREE, true);
  models[17] = RSModel(RSModel::TreeTypes::R_PLUS_PLUS_TREE, false);
  models[18] = RSModel(RSModel::TreeTypes::VP_TREE, true);
  models[19] = RSModel(RSModel::TreeTypes::VP_TREE, false);
  models[20] = RSModel(RSModel::TreeTypes::RP_TREE, true);
  models[21] = RSModel(RSModel::TreeTypes::RP_TREE, false);
  models[22] = RSModel(RSModel::TreeTypes::MAX_RP_TREE, true);
  models[23] = RSModel(RSModel::TreeTypes::MAX_RP_TREE, false);
  models[24] = RSModel(RSModel::TreeTypes::UB_TREE, true);
  models[25] = RSModel(RSModel::TreeTypes::UB_TREE, false);
  models[26] = RSModel(RSModel::TreeTypes::OCTREE, true);
  models[27] = RSModel(RSModel::TreeTypes::OCTREE, false);

  util::Timers timers;

  for (size_t j = 0; j < 3; ++j)
  {
    // Get a baseline.
    RangeSearch<> rs(referenceData);
    vector<vector<size_t>> baselineNeighbors;
    vector<vector<double>> baselineDistances;
    rs.Search(queryData, Range(0.25, 0.75), baselineNeighbors,
        baselineDistances);

    vector<vector<pair<double, size_t>>> baselineSorted;
    SortResults(baselineNeighbors, baselineDistances, baselineSorted);

    for (size_t i = 0; i < 28; ++i)
    {
      // We only have std::move() constructors, so make a copy of our data.
      arma::mat referenceCopy(referenceData);
      arma::mat queryCopy(queryData);
      if (j == 0)
        models[i].BuildModel(timers, std::move(referenceCopy), 5, false, false);
      else if (j == 1)
        models[i].BuildModel(timers, std::move(referenceCopy), 5, false, true);
      else if (j == 2)
        models[i].BuildModel(timers, std::move(referenceCopy), 5, true, false);

      vector<vector<size_t>> neighbors;
      vector<vector<double>> distances;

      models[i].Search(timers, std::move(queryCopy), Range(0.25, 0.75),
          neighbors, distances);

      REQUIRE(neighbors.size() == baselineNeighbors.size());
      REQUIRE(distances.size() == baselineDistances.size());

      vector<vector<pair<double, size_t>>> sorted;
      SortResults(neighbors, distances, sorted);

      for (size_t k = 0; k < sorted.size(); ++k)
      {
        REQUIRE(sorted[k].size() == baselineSorted[k].size());
        for (size_t l = 0; l < sorted[k].size(); ++l)
        {
          REQUIRE(sorted[k][l].second == baselineSorted[k][l].second);
          REQUIRE(sorted[k][l].first ==
              Approx(baselineSorted[k][l].first).epsilon(1e-7));
        }
      }
    }
  }
}

TEST_CASE("RSModelMonochromaticTest", "[RangeSearchTest]")
{
  // Ensure that we can build an RSModel and get correct results.
  arma::mat referenceData = arma::randu<arma::mat>(10, 200);

  // Build all the possible models.
  RSModel models[28];
  models[0] = RSModel(RSModel::TreeTypes::KD_TREE, true);
  models[1] = RSModel(RSModel::TreeTypes::KD_TREE, false);
  models[2] = RSModel(RSModel::TreeTypes::COVER_TREE, true);
  models[3] = RSModel(RSModel::TreeTypes::COVER_TREE, false);
  models[4] = RSModel(RSModel::TreeTypes::R_TREE, true);
  models[5] = RSModel(RSModel::TreeTypes::R_TREE, false);
  models[6] = RSModel(RSModel::TreeTypes::R_STAR_TREE, true);
  models[7] = RSModel(RSModel::TreeTypes::R_STAR_TREE, false);
  models[8] = RSModel(RSModel::TreeTypes::X_TREE, true);
  models[9] = RSModel(RSModel::TreeTypes::X_TREE, false);
  models[10] = RSModel(RSModel::TreeTypes::BALL_TREE, true);
  models[11] = RSModel(RSModel::TreeTypes::BALL_TREE, false);
  models[12] = RSModel(RSModel::TreeTypes::HILBERT_R_TREE, true);
  models[13] = RSModel(RSModel::TreeTypes::HILBERT_R_TREE, false);
  models[14] = RSModel(RSModel::TreeTypes::R_PLUS_TREE, true);
  models[15] = RSModel(RSModel::TreeTypes::R_PLUS_TREE, false);
  models[16] = RSModel(RSModel::TreeTypes::R_PLUS_PLUS_TREE, true);
  models[17] = RSModel(RSModel::TreeTypes::R_PLUS_PLUS_TREE, false);
  models[18] = RSModel(RSModel::TreeTypes::VP_TREE, true);
  models[19] = RSModel(RSModel::TreeTypes::VP_TREE, false);
  models[20] = RSModel(RSModel::TreeTypes::RP_TREE, true);
  models[21] = RSModel(RSModel::TreeTypes::RP_TREE, false);
  models[22] = RSModel(RSModel::TreeTypes::MAX_RP_TREE, true);
  models[23] = RSModel(RSModel::TreeTypes::MAX_RP_TREE, false);
  models[24] = RSModel(RSModel::TreeTypes::MAX_RP_TREE, true);
  models[25] = RSModel(RSModel::TreeTypes::MAX_RP_TREE, false);
  models[26] = RSModel(RSModel::TreeTypes::OCTREE, true);
  models[27] = RSModel(RSModel::TreeTypes::OCTREE, false);

  util::Timers timers;

  for (size_t j = 0; j < 3; ++j)
  {
    // Get a baseline.
    RangeSearch<> rs(referenceData);
    vector<vector<size_t>> baselineNeighbors;
    vector<vector<double>> baselineDistances;
    rs.Search(Range(0.25, 0.5), baselineNeighbors, baselineDistances);

    vector<vector<pair<double, size_t>>> baselineSorted;
    SortResults(baselineNeighbors, baselineDistances, baselineSorted);

    for (size_t i = 0; i < 28; ++i)
    {
      // We only have std::move() cosntructors, so make a copy of our data.
      arma::mat referenceCopy(referenceData);
      if (j == 0)
        models[i].BuildModel(timers, std::move(referenceCopy), 5, false, false);
      else if (j == 1)
        models[i].BuildModel(timers, std::move(referenceCopy), 5, false, true);
      else if (j == 2)
        models[i].BuildModel(timers, std::move(referenceCopy), 5, true, false);

      vector<vector<size_t>> neighbors;
      vector<vector<double>> distances;

      models[i].Search(timers, Range(0.25, 0.5), neighbors, distances);

      REQUIRE(neighbors.size() == baselineNeighbors.size());
      REQUIRE(distances.size() == baselineDistances.size());

      vector<vector<pair<double, size_t>>> sorted;
      SortResults(neighbors, distances, sorted);

      for (size_t k = 0; k < sorted.size(); ++k)
      {
        REQUIRE(sorted[k].size() == baselineSorted[k].size());
        for (size_t l = 0; l < sorted[k].size(); ++l)
        {
          REQUIRE(sorted[k][l].second == baselineSorted[k][l].second);
          REQUIRE(sorted[k][l].first ==
              Approx(baselineSorted[k][l].first).epsilon(1e-7));
        }
      }
    }
  }
}

/**
 * Make sure that the neighborPtr matrix isn't accidentally deleted.
 * See issue #478.
 */
TEST_CASE("NeighborPtrDeleteTest", "[RangeSearchTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(5, 100);

  // Build the tree ourselves.
  vector<size_t> oldFromNewReferences;
  RangeSearch<>::Tree tree(dataset);
  RangeSearch<> ra(&tree);

  // Now make a query set.
  arma::mat queryset = arma::randu<arma::mat>(5, 50);
  vector<vector<double>> distances;
  vector<vector<size_t>> neighbors;
  ra.Search(queryset, Range(0.2, 0.5), neighbors, distances);

  // These will (hopefully) fail is either the neighbors or the distances matrix
  // has been accidentally deleted.
  REQUIRE(neighbors.size() == 50);
  REQUIRE(distances.size() == 50);
}

/**
 * Test copy constructor and copy operator.
 */
TEST_CASE("RangeSearchCopyConstructorAndOperatorTest", "[RangeSearchTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(5, 500);
  RangeSearch<> rs(std::move(dataset));

  // Copy constructor and operator.
  RangeSearch<> rs2(rs);
  RangeSearch<> rs3 = rs;

  // Get results.
  vector<vector<double>> distances, distances2, distances3;
  vector<vector<size_t>> neighbors, neighbors2, neighbors3;

  rs.Search(Range(0.2, 0.3), neighbors, distances);
  rs2.Search(Range(0.2, 0.3), neighbors2, distances2);
  rs3.Search(Range(0.2, 0.3), neighbors3, distances3);

  // Check results.
  REQUIRE(distances.size() == distances2.size());
  REQUIRE(distances.size() == distances3.size());
  REQUIRE(neighbors.size() == neighbors2.size());
  REQUIRE(neighbors.size() == neighbors3.size());

  for (size_t i = 0; i < neighbors.size(); ++i)
  {
    REQUIRE(distances[i].size() == distances2[i].size());
    REQUIRE(distances[i].size() == distances3[i].size());
    REQUIRE(neighbors[i].size() == neighbors2[i].size());
    REQUIRE(neighbors[i].size() == neighbors3[i].size());

    for (size_t j = 0; j < neighbors[i].size(); ++j)
    {
      REQUIRE(neighbors[i][j] == neighbors2[i][j]);
      REQUIRE(neighbors[i][j] == neighbors3[i][j]);

      // Distances will always be between 0.2 and 0.3.
      REQUIRE(distances[i][j] == Approx(distances2[i][j]).epsilon(1e-7));
      REQUIRE(distances[i][j] == Approx(distances3[i][j]).epsilon(1e-7));
    }
  }
}

/**
 * Test move constructor.
 */
TEST_CASE("RangeSearchMoveConstructorTest", "[RangeSearchTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(5, 500);
  RangeSearch<>* rs = new RangeSearch<>(std::move(dataset));

  // Get results.
  vector<vector<double>> distances, distances2;
  vector<vector<size_t>> neighbors, neighbors2;

  rs->Search(Range(0.2, 0.3), neighbors, distances);

  RangeSearch<> rs2(std::move(*rs));

  delete rs;

  rs2.Search(Range(0.2, 0.3), neighbors2, distances2);

  // Check results.
  REQUIRE(distances.size() == distances2.size());
  REQUIRE(neighbors.size() == neighbors2.size());

  for (size_t i = 0; i < neighbors.size(); ++i)
  {
    REQUIRE(distances[i].size() == distances2[i].size());
    REQUIRE(neighbors[i].size() == neighbors2[i].size());

    for (size_t j = 0; j < neighbors[i].size(); ++j)
    {
      REQUIRE(neighbors[i][j] == neighbors2[i][j]);

      // Distances will always be between 0.2 and 0.3.
      REQUIRE(distances[i][j] == Approx(distances2[i][j]).epsilon(1e-7));
    }
  }
}

/**
 * Test move operator.
 */
TEST_CASE("RangeSearchMoveOperatorTest", "[RangeSearchTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(5, 500);
  RangeSearch<>* rs = new RangeSearch<>(std::move(dataset));

  // Get results.
  vector<vector<double>> distances, distances2;
  vector<vector<size_t>> neighbors, neighbors2;

  rs->Search(Range(0.2, 0.3), neighbors, distances);

  RangeSearch<> rs2 = std::move(*rs);

  delete rs;

  rs2.Search(Range(0.2, 0.3), neighbors2, distances2);

  // Check results.
  REQUIRE(distances.size() == distances2.size());
  REQUIRE(neighbors.size() == neighbors2.size());

  for (size_t i = 0; i < neighbors.size(); ++i)
  {
    REQUIRE(distances[i].size() == distances2[i].size());
    REQUIRE(neighbors[i].size() == neighbors2[i].size());

    for (size_t j = 0; j < neighbors[i].size(); ++j)
    {
      REQUIRE(neighbors[i][j] == neighbors2[i][j]);

      // Distances will always be between 0.2 and 0.3.
      REQUIRE(distances[i][j] == Approx(distances2[i][j]).epsilon(1e-7));
    }
  }
}

/**
 * Test copy constructor and copy operator in naive mode (so there are no
 * trees).
 */
TEST_CASE("CopyConstructorAndOperatorNaiveTest", "[RangeSearchTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(5, 500);
  RangeSearch<> rs(std::move(dataset), true);

  // Copy constructor and operator.
  RangeSearch<> rs2(rs);
  RangeSearch<> rs3 = rs;

  REQUIRE(rs2.Naive() == true);
  REQUIRE(rs3.Naive() == true);

  // Get results.
  vector<vector<double>> distances, distances2, distances3;
  vector<vector<size_t>> neighbors, neighbors2, neighbors3;

  rs.Search(Range(0.2, 0.3), neighbors, distances);
  rs2.Search(Range(0.2, 0.3), neighbors2, distances2);
  rs3.Search(Range(0.2, 0.3), neighbors3, distances3);

  // Check results.
  REQUIRE(distances.size() == distances2.size());
  REQUIRE(distances.size() == distances3.size());
  REQUIRE(neighbors.size() == neighbors2.size());
  REQUIRE(neighbors.size() == neighbors3.size());

  for (size_t i = 0; i < neighbors.size(); ++i)
  {
    REQUIRE(distances[i].size() == distances2[i].size());
    REQUIRE(distances[i].size() == distances3[i].size());
    REQUIRE(neighbors[i].size() == neighbors2[i].size());
    REQUIRE(neighbors[i].size() == neighbors3[i].size());

    for (size_t j = 0; j < neighbors[i].size(); ++j)
    {
      REQUIRE(neighbors[i][j] == neighbors2[i][j]);
      REQUIRE(neighbors[i][j] == neighbors3[i][j]);

      // Distances will always be between 0.2 and 0.3.
      REQUIRE(distances[i][j] == Approx(distances2[i][j]).epsilon(1e-7));
      REQUIRE(distances[i][j] == Approx(distances3[i][j]).epsilon(1e-7));
    }
  }
}

/**
 * Test move constructor.
 */
TEST_CASE("MoveConstructorNaiveTest", "[RangeSearchTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(5, 500);
  RangeSearch<>* rs = new RangeSearch<>(std::move(dataset), true);

  // Get results.
  vector<vector<double>> distances, distances2;
  vector<vector<size_t>> neighbors, neighbors2;

  rs->Search(Range(0.2, 0.3), neighbors, distances);

  RangeSearch<> rs2(std::move(*rs));

  REQUIRE(rs2.Naive() == true);

  delete rs;

  rs2.Search(Range(0.2, 0.3), neighbors2, distances2);

  // Check results.
  REQUIRE(distances.size() == distances2.size());
  REQUIRE(neighbors.size() == neighbors2.size());

  for (size_t i = 0; i < neighbors.size(); ++i)
  {
    REQUIRE(distances[i].size() == distances2[i].size());
    REQUIRE(neighbors[i].size() == neighbors2[i].size());

    for (size_t j = 0; j < neighbors[i].size(); ++j)
    {
      REQUIRE(neighbors[i][j] == neighbors2[i][j]);

      // Distances will always be between 0.2 and 0.3.
      REQUIRE(distances[i][j] == Approx(distances2[i][j]).epsilon(1e-7));
    }
  }
}

/**
 * Test move operator.
 */
TEST_CASE("MoveOperatorNaiveTest", "[RangeSearchTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(5, 500);
  RangeSearch<>* rs = new RangeSearch<>(std::move(dataset), true);

  // Get results.
  vector<vector<double>> distances, distances2;
  vector<vector<size_t>> neighbors, neighbors2;

  rs->Search(Range(0.2, 0.3), neighbors, distances);

  RangeSearch<> rs2 = std::move(*rs);

  REQUIRE(rs2.Naive() == true);

  delete rs;

  rs2.Search(Range(0.2, 0.3), neighbors2, distances2);

  // Check results.
  REQUIRE(distances.size() == distances2.size());
  REQUIRE(neighbors.size() == neighbors2.size());

  for (size_t i = 0; i < neighbors.size(); ++i)
  {
    REQUIRE(distances[i].size() == distances2[i].size());
    REQUIRE(neighbors[i].size() == neighbors2[i].size());

    for (size_t j = 0; j < neighbors[i].size(); ++j)
    {
      REQUIRE(neighbors[i][j] == neighbors2[i][j]);

      // Distances will always be between 0.2 and 0.3.
      REQUIRE(distances[i][j] == Approx(distances2[i][j]).epsilon(1e-7));
    }
  }
}
