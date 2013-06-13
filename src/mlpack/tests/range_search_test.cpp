/**
 * @file range_search_test.cpp
 * @author Ryan Curtin
 *
 * Test file for RangeSearch<> class.
 *
 * This file is part of MLPACK 1.0.6.
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
#include <mlpack/methods/range_search/range_search.hpp>
#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::range;
using namespace mlpack::math;
using namespace std;

BOOST_AUTO_TEST_SUITE(RangeSearchTest);

// Get our results into a sorted format, so we can actually then test for
// correctness.
void SortResults(const vector<vector<size_t> >& neighbors,
                 const vector<vector<double> >& distances,
                 vector<vector<pair<double, size_t> > >& output)
{
  output.resize(neighbors.size());
  for (size_t i = 0; i < neighbors.size(); i++)
  {
    output[i].resize(neighbors[i].size());
    for (size_t j = 0; j < neighbors[i].size(); j++)
      output[i][j] = make_pair(distances[i][j], neighbors[i][j]);

    // Now that it's constructed, sort it.
    sort(output[i].begin(), output[i].end());
  }
}

/**
 * Simple range-search test with small, synthetic dataset.  This is an
 * exhaustive test, which checks that each method for performing the calculation
 * (dual-tree, single-tree, naive) produces the correct results.  An
 * eleven-point dataset and the points within three ranges are taken.  The
 * dataset is in one dimension for simplicity -- the correct functionality of
 * distance functions is not tested here.
 */
BOOST_AUTO_TEST_CASE(ExhaustiveSyntheticTest)
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

  // We will loop through three times, one for each method of performing the
  // calculation.
  for (int i = 0; i < 3; i++)
  {
    RangeSearch<>* rs;
    arma::mat dataMutable = data;
    switch (i)
    {
      case 0: // Use the dual-tree method.
        rs = new RangeSearch<>(dataMutable, false, false, 1);
        break;
      case 1: // Use the single-tree method.
        rs = new RangeSearch<>(dataMutable, false, true, 1);
        break;
      case 2: // Use the naive method.
        rs = new RangeSearch<>(dataMutable, true);
        break;
    }

    // Now perform the first calculation.  Points within 0.50.
    vector<vector<size_t> > neighbors;
    vector<vector<double> > distances;
    rs->Search(Range(0.0, 0.50), neighbors, distances);

    // Now the exhaustive check for correctness.  This will be long.  We must
    // also remember that the distances returned are squared distances.  As a
    // result, distance comparisons are written out as (distance * distance) for
    // readability.
    vector<vector<pair<double, size_t> > > sortedOutput;
    SortResults(neighbors, distances, sortedOutput);

    // Neighbors of point 0.
    BOOST_REQUIRE(sortedOutput[0].size() == 4);
    BOOST_REQUIRE(sortedOutput[0][0].second == 2);
    BOOST_REQUIRE_CLOSE(sortedOutput[0][0].first, (0.10 * 0.10), 1e-5);
    BOOST_REQUIRE(sortedOutput[0][1].second == 5);
    BOOST_REQUIRE_CLOSE(sortedOutput[0][1].first, (0.27 * 0.27), 1e-5);
    BOOST_REQUIRE(sortedOutput[0][2].second == 1);
    BOOST_REQUIRE_CLOSE(sortedOutput[0][2].first, (0.30 * 0.30), 1e-5);
    BOOST_REQUIRE(sortedOutput[0][3].second == 8);
    BOOST_REQUIRE_CLOSE(sortedOutput[0][3].first, (0.40 * 0.40), 1e-5);

    // Neighbors of point 1.
    BOOST_REQUIRE(sortedOutput[1].size() == 6);
    BOOST_REQUIRE(sortedOutput[1][0].second == 8);
    BOOST_REQUIRE_CLOSE(sortedOutput[1][0].first, (0.10 * 0.10), 1e-5);
    BOOST_REQUIRE(sortedOutput[1][1].second == 2);
    BOOST_REQUIRE_CLOSE(sortedOutput[1][1].first, (0.20 * 0.20), 1e-5);
    BOOST_REQUIRE(sortedOutput[1][2].second == 0);
    BOOST_REQUIRE_CLOSE(sortedOutput[1][2].first, (0.30 * 0.30), 1e-5);
    BOOST_REQUIRE(sortedOutput[1][3].second == 9);
    BOOST_REQUIRE_CLOSE(sortedOutput[1][3].first, (0.55 * 0.55), 1e-5);
    BOOST_REQUIRE(sortedOutput[1][4].second == 5);
    BOOST_REQUIRE_CLOSE(sortedOutput[1][4].first, (0.57 * 0.57), 1e-5);
    BOOST_REQUIRE(sortedOutput[1][5].second == 10);
    BOOST_REQUIRE_CLOSE(sortedOutput[1][5].first, (0.65 * 0.65), 1e-5);

    // Neighbors of point 2.
    BOOST_REQUIRE(sortedOutput[2].size() == 4);
    BOOST_REQUIRE(sortedOutput[2][0].second == 0);
    BOOST_REQUIRE_CLOSE(sortedOutput[2][0].first, (0.10 * 0.10), 1e-5);
    BOOST_REQUIRE(sortedOutput[2][1].second == 1);
    BOOST_REQUIRE_CLOSE(sortedOutput[2][1].first, (0.20 * 0.20), 1e-5);
    BOOST_REQUIRE(sortedOutput[2][2].second == 8);
    BOOST_REQUIRE_CLOSE(sortedOutput[2][2].first, (0.30 * 0.30), 1e-5);
    BOOST_REQUIRE(sortedOutput[2][3].second == 5);
    BOOST_REQUIRE_CLOSE(sortedOutput[2][3].first, (0.37 * 0.37), 1e-5);

    // Neighbors of point 3.
    BOOST_REQUIRE(sortedOutput[3].size() == 2);
    BOOST_REQUIRE(sortedOutput[3][0].second == 10);
    BOOST_REQUIRE_CLOSE(sortedOutput[3][0].first, (0.25 * 0.25), 1e-5);
    BOOST_REQUIRE(sortedOutput[3][1].second == 9);
    BOOST_REQUIRE_CLOSE(sortedOutput[3][1].first, (0.35 * 0.35), 1e-5);

    // Neighbors of point 4.
    BOOST_REQUIRE(sortedOutput[4].size() == 0);

    // Neighbors of point 5.
    BOOST_REQUIRE(sortedOutput[5].size() == 4);
    BOOST_REQUIRE(sortedOutput[5][0].second == 0);
    BOOST_REQUIRE_CLOSE(sortedOutput[5][0].first, (0.27 * 0.27), 1e-5);
    BOOST_REQUIRE(sortedOutput[5][1].second == 2);
    BOOST_REQUIRE_CLOSE(sortedOutput[5][1].first, (0.37 * 0.37), 1e-5);
    BOOST_REQUIRE(sortedOutput[5][2].second == 1);
    BOOST_REQUIRE_CLOSE(sortedOutput[5][2].first, (0.57 * 0.57), 1e-5);
    BOOST_REQUIRE(sortedOutput[5][3].second == 8);
    BOOST_REQUIRE_CLOSE(sortedOutput[5][3].first, (0.67 * 0.67), 1e-5);

    // Neighbors of point 6.
    BOOST_REQUIRE(sortedOutput[6].size() == 1);
    BOOST_REQUIRE(sortedOutput[6][0].second == 7);
    BOOST_REQUIRE_CLOSE(sortedOutput[6][0].first, (0.70 * 0.70), 1e-5);

    // Neighbors of point 7.
    BOOST_REQUIRE(sortedOutput[7].size() == 1);
    BOOST_REQUIRE(sortedOutput[7][0].second == 6);
    BOOST_REQUIRE_CLOSE(sortedOutput[7][0].first, (0.70 * 0.70), 1e-5);

    // Neighbors of point 8.
    BOOST_REQUIRE(sortedOutput[8].size() == 6);
    BOOST_REQUIRE(sortedOutput[8][0].second == 1);
    BOOST_REQUIRE_CLOSE(sortedOutput[8][0].first, (0.10 * 0.10), 1e-5);
    BOOST_REQUIRE(sortedOutput[8][1].second == 2);
    BOOST_REQUIRE_CLOSE(sortedOutput[8][1].first, (0.30 * 0.30), 1e-5);
    BOOST_REQUIRE(sortedOutput[8][2].second == 0);
    BOOST_REQUIRE_CLOSE(sortedOutput[8][2].first, (0.40 * 0.40), 1e-5);
    BOOST_REQUIRE(sortedOutput[8][3].second == 9);
    BOOST_REQUIRE_CLOSE(sortedOutput[8][3].first, (0.45 * 0.45), 1e-5);
    BOOST_REQUIRE(sortedOutput[8][4].second == 10);
    BOOST_REQUIRE_CLOSE(sortedOutput[8][4].first, (0.55 * 0.55), 1e-5);
    BOOST_REQUIRE(sortedOutput[8][5].second == 5);
    BOOST_REQUIRE_CLOSE(sortedOutput[8][5].first, (0.67 * 0.67), 1e-5);

    // Neighbors of point 9.
    BOOST_REQUIRE(sortedOutput[9].size() == 4);
    BOOST_REQUIRE(sortedOutput[9][0].second == 10);
    BOOST_REQUIRE_CLOSE(sortedOutput[9][0].first, (0.10 * 0.10), 1e-5);
    BOOST_REQUIRE(sortedOutput[9][1].second == 3);
    BOOST_REQUIRE_CLOSE(sortedOutput[9][1].first, (0.35 * 0.35), 1e-5);
    BOOST_REQUIRE(sortedOutput[9][2].second == 8);
    BOOST_REQUIRE_CLOSE(sortedOutput[9][2].first, (0.45 * 0.45), 1e-5);
    BOOST_REQUIRE(sortedOutput[9][3].second == 1);
    BOOST_REQUIRE_CLOSE(sortedOutput[9][3].first, (0.55 * 0.55), 1e-5);

    // Neighbors of point 10.
    BOOST_REQUIRE(sortedOutput[10].size() == 4);
    BOOST_REQUIRE(sortedOutput[10][0].second == 9);
    BOOST_REQUIRE_CLOSE(sortedOutput[10][0].first, (0.10 * 0.10), 1e-5);
    BOOST_REQUIRE(sortedOutput[10][1].second == 3);
    BOOST_REQUIRE_CLOSE(sortedOutput[10][1].first, (0.25 * 0.25), 1e-5);
    BOOST_REQUIRE(sortedOutput[10][2].second == 8);
    BOOST_REQUIRE_CLOSE(sortedOutput[10][2].first, (0.55 * 0.55), 1e-5);
    BOOST_REQUIRE(sortedOutput[10][3].second == 1);
    BOOST_REQUIRE_CLOSE(sortedOutput[10][3].first, (0.65 * 0.65), 1e-5);

    // Now do it again with a different range: [0.5 1.0].
    rs->Search(Range(0.5, 1.0), neighbors, distances);
    SortResults(neighbors, distances, sortedOutput);

    // Neighbors of point 0.
    BOOST_REQUIRE(sortedOutput[0].size() == 2);
    BOOST_REQUIRE(sortedOutput[0][0].second == 9);
    BOOST_REQUIRE_CLOSE(sortedOutput[0][0].first, (0.85 * 0.85), 1e-5);
    BOOST_REQUIRE(sortedOutput[0][1].second == 10);
    BOOST_REQUIRE_CLOSE(sortedOutput[0][1].first, (0.95 * 0.95), 1e-5);

    // Neighbors of point 1.
    BOOST_REQUIRE(sortedOutput[1].size() == 1);
    BOOST_REQUIRE(sortedOutput[1][0].second == 3);
    BOOST_REQUIRE_CLOSE(sortedOutput[1][0].first, (0.90 * 0.90), 1e-5);

    // Neighbors of point 2.
    BOOST_REQUIRE(sortedOutput[2].size() == 2);
    BOOST_REQUIRE(sortedOutput[2][0].second == 9);
    BOOST_REQUIRE_CLOSE(sortedOutput[2][0].first, (0.75 * 0.75), 1e-5);
    BOOST_REQUIRE(sortedOutput[2][1].second == 10);
    BOOST_REQUIRE_CLOSE(sortedOutput[2][1].first, (0.85 * 0.85), 1e-5);

    // Neighbors of point 3.
    BOOST_REQUIRE(sortedOutput[3].size() == 2);
    BOOST_REQUIRE(sortedOutput[3][0].second == 8);
    BOOST_REQUIRE_CLOSE(sortedOutput[3][0].first, (0.80 * 0.80), 1e-5);
    BOOST_REQUIRE(sortedOutput[3][1].second == 1);
    BOOST_REQUIRE_CLOSE(sortedOutput[3][1].first, (0.90 * 0.90), 1e-5);

    // Neighbors of point 4.
    BOOST_REQUIRE(sortedOutput[4].size() == 0);

    // Neighbors of point 5.
    BOOST_REQUIRE(sortedOutput[5].size() == 0);

    // Neighbors of point 6.
    BOOST_REQUIRE(sortedOutput[6].size() == 0);

    // Neighbors of point 7.
    BOOST_REQUIRE(sortedOutput[7].size() == 0);

    // Neighbors of point 8.
    BOOST_REQUIRE(sortedOutput[8].size() == 1);
    BOOST_REQUIRE(sortedOutput[8][0].second == 3);
    BOOST_REQUIRE_CLOSE(sortedOutput[8][0].first, (0.80 * 0.80), 1e-5);

    // Neighbors of point 9.
    BOOST_REQUIRE(sortedOutput[9].size() == 2);
    BOOST_REQUIRE(sortedOutput[9][0].second == 2);
    BOOST_REQUIRE_CLOSE(sortedOutput[9][0].first, (0.75 * 0.75), 1e-5);
    BOOST_REQUIRE(sortedOutput[9][1].second == 0);
    BOOST_REQUIRE_CLOSE(sortedOutput[9][1].first, (0.85 * 0.85), 1e-5);

    // Neighbors of point 10.
    BOOST_REQUIRE(sortedOutput[10].size() == 2);
    BOOST_REQUIRE(sortedOutput[10][0].second == 2);
    BOOST_REQUIRE_CLOSE(sortedOutput[10][0].first, (0.85 * 0.85), 1e-5);
    BOOST_REQUIRE(sortedOutput[10][1].second == 0);
    BOOST_REQUIRE_CLOSE(sortedOutput[10][1].first, (0.95 * 0.95), 1e-5);

    // Now do it again with a different range: [1.0 inf].
    rs->Search(Range(1.0, numeric_limits<double>::infinity()), neighbors,
        distances);
    SortResults(neighbors, distances, sortedOutput);

    // Neighbors of point 0.
    BOOST_REQUIRE(sortedOutput[0].size() == 4);
    BOOST_REQUIRE(sortedOutput[0][0].second == 3);
    BOOST_REQUIRE_CLOSE(sortedOutput[0][0].first, (1.20 * 1.20), 1e-5);
    BOOST_REQUIRE(sortedOutput[0][1].second == 7);
    BOOST_REQUIRE_CLOSE(sortedOutput[0][1].first, (1.35 * 1.35), 1e-5);
    BOOST_REQUIRE(sortedOutput[0][2].second == 6);
    BOOST_REQUIRE_CLOSE(sortedOutput[0][2].first, (2.05 * 2.05), 1e-5);
    BOOST_REQUIRE(sortedOutput[0][3].second == 4);
    BOOST_REQUIRE_CLOSE(sortedOutput[0][3].first, (5.00 * 5.00), 1e-5);

    // Neighbors of point 1.
    BOOST_REQUIRE(sortedOutput[1].size() == 3);
    BOOST_REQUIRE(sortedOutput[1][0].second == 7);
    BOOST_REQUIRE_CLOSE(sortedOutput[1][0].first, (1.65 * 1.65), 1e-5);
    BOOST_REQUIRE(sortedOutput[1][1].second == 6);
    BOOST_REQUIRE_CLOSE(sortedOutput[1][1].first, (2.35 * 2.35), 1e-5);
    BOOST_REQUIRE(sortedOutput[1][2].second == 4);
    BOOST_REQUIRE_CLOSE(sortedOutput[1][2].first, (4.70 * 4.70), 1e-5);

    // Neighbors of point 2.
    BOOST_REQUIRE(sortedOutput[2].size() == 4);
    BOOST_REQUIRE(sortedOutput[2][0].second == 3);
    BOOST_REQUIRE_CLOSE(sortedOutput[2][0].first, (1.10 * 1.10), 1e-5);
    BOOST_REQUIRE(sortedOutput[2][1].second == 7);
    BOOST_REQUIRE_CLOSE(sortedOutput[2][1].first, (1.45 * 1.45), 1e-5);
    BOOST_REQUIRE(sortedOutput[2][2].second == 6);
    BOOST_REQUIRE_CLOSE(sortedOutput[2][2].first, (2.15 * 2.15), 1e-5);
    BOOST_REQUIRE(sortedOutput[2][3].second == 4);
    BOOST_REQUIRE_CLOSE(sortedOutput[2][3].first, (4.90 * 4.90), 1e-5);

    // Neighbors of point 3.
    BOOST_REQUIRE(sortedOutput[3].size() == 6);
    BOOST_REQUIRE(sortedOutput[3][0].second == 2);
    BOOST_REQUIRE_CLOSE(sortedOutput[3][0].first, (1.10 * 1.10), 1e-5);
    BOOST_REQUIRE(sortedOutput[3][1].second == 0);
    BOOST_REQUIRE_CLOSE(sortedOutput[3][1].first, (1.20 * 1.20), 1e-5);
    BOOST_REQUIRE(sortedOutput[3][2].second == 5);
    BOOST_REQUIRE_CLOSE(sortedOutput[3][2].first, (1.47 * 1.47), 1e-5);
    BOOST_REQUIRE(sortedOutput[3][3].second == 7);
    BOOST_REQUIRE_CLOSE(sortedOutput[3][3].first, (2.55 * 2.55), 1e-5);
    BOOST_REQUIRE(sortedOutput[3][4].second == 6);
    BOOST_REQUIRE_CLOSE(sortedOutput[3][4].first, (3.25 * 3.25), 1e-5);
    BOOST_REQUIRE(sortedOutput[3][5].second == 4);
    BOOST_REQUIRE_CLOSE(sortedOutput[3][5].first, (3.80 * 3.80), 1e-5);

    // Neighbors of point 4.
    BOOST_REQUIRE(sortedOutput[4].size() == 10);
    BOOST_REQUIRE(sortedOutput[4][0].second == 3);
    BOOST_REQUIRE_CLOSE(sortedOutput[4][0].first, (3.80 * 3.80), 1e-5);
    BOOST_REQUIRE(sortedOutput[4][1].second == 10);
    BOOST_REQUIRE_CLOSE(sortedOutput[4][1].first, (4.05 * 4.05), 1e-5);
    BOOST_REQUIRE(sortedOutput[4][2].second == 9);
    BOOST_REQUIRE_CLOSE(sortedOutput[4][2].first, (4.15 * 4.15), 1e-5);
    BOOST_REQUIRE(sortedOutput[4][3].second == 8);
    BOOST_REQUIRE_CLOSE(sortedOutput[4][3].first, (4.60 * 4.60), 1e-5);
    BOOST_REQUIRE(sortedOutput[4][4].second == 1);
    BOOST_REQUIRE_CLOSE(sortedOutput[4][4].first, (4.70 * 4.70), 1e-5);
    BOOST_REQUIRE(sortedOutput[4][5].second == 2);
    BOOST_REQUIRE_CLOSE(sortedOutput[4][5].first, (4.90 * 4.90), 1e-5);
    BOOST_REQUIRE(sortedOutput[4][6].second == 0);
    BOOST_REQUIRE_CLOSE(sortedOutput[4][6].first, (5.00 * 5.00), 1e-5);
    BOOST_REQUIRE(sortedOutput[4][7].second == 5);
    BOOST_REQUIRE_CLOSE(sortedOutput[4][7].first, (5.27 * 5.27), 1e-5);
    BOOST_REQUIRE(sortedOutput[4][8].second == 7);
    BOOST_REQUIRE_CLOSE(sortedOutput[4][8].first, (6.35 * 6.35), 1e-5);
    BOOST_REQUIRE(sortedOutput[4][9].second == 6);
    BOOST_REQUIRE_CLOSE(sortedOutput[4][9].first, (7.05 * 7.05), 1e-5);

    // Neighbors of point 5.
    BOOST_REQUIRE(sortedOutput[5].size() == 6);
    BOOST_REQUIRE(sortedOutput[5][0].second == 7);
    BOOST_REQUIRE_CLOSE(sortedOutput[5][0].first, (1.08 * 1.08), 1e-5);
    BOOST_REQUIRE(sortedOutput[5][1].second == 9);
    BOOST_REQUIRE_CLOSE(sortedOutput[5][1].first, (1.12 * 1.12), 1e-5);
    BOOST_REQUIRE(sortedOutput[5][2].second == 10);
    BOOST_REQUIRE_CLOSE(sortedOutput[5][2].first, (1.22 * 1.22), 1e-5);
    BOOST_REQUIRE(sortedOutput[5][3].second == 3);
    BOOST_REQUIRE_CLOSE(sortedOutput[5][3].first, (1.47 * 1.47), 1e-5);
    BOOST_REQUIRE(sortedOutput[5][4].second == 6);
    BOOST_REQUIRE_CLOSE(sortedOutput[5][4].first, (1.78 * 1.78), 1e-5);
    BOOST_REQUIRE(sortedOutput[5][5].second == 4);
    BOOST_REQUIRE_CLOSE(sortedOutput[5][5].first, (5.27 * 5.27), 1e-5);

    // Neighbors of point 6.
    BOOST_REQUIRE(sortedOutput[6].size() == 9);
    BOOST_REQUIRE(sortedOutput[6][0].second == 5);
    BOOST_REQUIRE_CLOSE(sortedOutput[6][0].first, (1.78 * 1.78), 1e-5);
    BOOST_REQUIRE(sortedOutput[6][1].second == 0);
    BOOST_REQUIRE_CLOSE(sortedOutput[6][1].first, (2.05 * 2.05), 1e-5);
    BOOST_REQUIRE(sortedOutput[6][2].second == 2);
    BOOST_REQUIRE_CLOSE(sortedOutput[6][2].first, (2.15 * 2.15), 1e-5);
    BOOST_REQUIRE(sortedOutput[6][3].second == 1);
    BOOST_REQUIRE_CLOSE(sortedOutput[6][3].first, (2.35 * 2.35), 1e-5);
    BOOST_REQUIRE(sortedOutput[6][4].second == 8);
    BOOST_REQUIRE_CLOSE(sortedOutput[6][4].first, (2.45 * 2.45), 1e-5);
    BOOST_REQUIRE(sortedOutput[6][5].second == 9);
    BOOST_REQUIRE_CLOSE(sortedOutput[6][5].first, (2.90 * 2.90), 1e-5);
    BOOST_REQUIRE(sortedOutput[6][6].second == 10);
    BOOST_REQUIRE_CLOSE(sortedOutput[6][6].first, (3.00 * 3.00), 1e-5);
    BOOST_REQUIRE(sortedOutput[6][7].second == 3);
    BOOST_REQUIRE_CLOSE(sortedOutput[6][7].first, (3.25 * 3.25), 1e-5);
    BOOST_REQUIRE(sortedOutput[6][8].second == 4);
    BOOST_REQUIRE_CLOSE(sortedOutput[6][8].first, (7.05 * 7.05), 1e-5);

    // Neighbors of point 7.
    BOOST_REQUIRE(sortedOutput[7].size() == 9);
    BOOST_REQUIRE(sortedOutput[7][0].second == 5);
    BOOST_REQUIRE_CLOSE(sortedOutput[7][0].first, (1.08 * 1.08), 1e-5);
    BOOST_REQUIRE(sortedOutput[7][1].second == 0);
    BOOST_REQUIRE_CLOSE(sortedOutput[7][1].first, (1.35 * 1.35), 1e-5);
    BOOST_REQUIRE(sortedOutput[7][2].second == 2);
    BOOST_REQUIRE_CLOSE(sortedOutput[7][2].first, (1.45 * 1.45), 1e-5);
    BOOST_REQUIRE(sortedOutput[7][3].second == 1);
    BOOST_REQUIRE_CLOSE(sortedOutput[7][3].first, (1.65 * 1.65), 1e-5);
    BOOST_REQUIRE(sortedOutput[7][4].second == 8);
    BOOST_REQUIRE_CLOSE(sortedOutput[7][4].first, (1.75 * 1.75), 1e-5);
    BOOST_REQUIRE(sortedOutput[7][5].second == 9);
    BOOST_REQUIRE_CLOSE(sortedOutput[7][5].first, (2.20 * 2.20), 1e-5);
    BOOST_REQUIRE(sortedOutput[7][6].second == 10);
    BOOST_REQUIRE_CLOSE(sortedOutput[7][6].first, (2.30 * 2.30), 1e-5);
    BOOST_REQUIRE(sortedOutput[7][7].second == 3);
    BOOST_REQUIRE_CLOSE(sortedOutput[7][7].first, (2.55 * 2.55), 1e-5);
    BOOST_REQUIRE(sortedOutput[7][8].second == 4);
    BOOST_REQUIRE_CLOSE(sortedOutput[7][8].first, (6.35 * 6.35), 1e-5);

    // Neighbors of point 8.
    BOOST_REQUIRE(sortedOutput[8].size() == 3);
    BOOST_REQUIRE(sortedOutput[8][0].second == 7);
    BOOST_REQUIRE_CLOSE(sortedOutput[8][0].first, (1.75 * 1.75), 1e-5);
    BOOST_REQUIRE(sortedOutput[8][1].second == 6);
    BOOST_REQUIRE_CLOSE(sortedOutput[8][1].first, (2.45 * 2.45), 1e-5);
    BOOST_REQUIRE(sortedOutput[8][2].second == 4);
    BOOST_REQUIRE_CLOSE(sortedOutput[8][2].first, (4.60 * 4.60), 1e-5);

    // Neighbors of point 9.
    BOOST_REQUIRE(sortedOutput[9].size() == 4);
    BOOST_REQUIRE(sortedOutput[9][0].second == 5);
    BOOST_REQUIRE_CLOSE(sortedOutput[9][0].first, (1.12 * 1.12), 1e-5);
    BOOST_REQUIRE(sortedOutput[9][1].second == 7);
    BOOST_REQUIRE_CLOSE(sortedOutput[9][1].first, (2.20 * 2.20), 1e-5);
    BOOST_REQUIRE(sortedOutput[9][2].second == 6);
    BOOST_REQUIRE_CLOSE(sortedOutput[9][2].first, (2.90 * 2.90), 1e-5);
    BOOST_REQUIRE(sortedOutput[9][3].second == 4);
    BOOST_REQUIRE_CLOSE(sortedOutput[9][3].first, (4.15 * 4.15), 1e-5);

    // Neighbors of point 10.
    BOOST_REQUIRE(sortedOutput[10].size() == 4);
    BOOST_REQUIRE(sortedOutput[10][0].second == 5);
    BOOST_REQUIRE_CLOSE(sortedOutput[10][0].first, (1.22 * 1.22), 1e-5);
    BOOST_REQUIRE(sortedOutput[10][1].second == 7);
    BOOST_REQUIRE_CLOSE(sortedOutput[10][1].first, (2.30 * 2.30), 1e-5);
    BOOST_REQUIRE(sortedOutput[10][2].second == 6);
    BOOST_REQUIRE_CLOSE(sortedOutput[10][2].first, (3.00 * 3.00), 1e-5);
    BOOST_REQUIRE(sortedOutput[10][3].second == 4);
    BOOST_REQUIRE_CLOSE(sortedOutput[10][3].first, (4.05 * 4.05), 1e-5);

    // Clean the memory.
    delete rs;
  }
}

/**
 * Test the dual-tree nearest-neighbors method with the naive method.  This
 * uses both a query and reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(DualTreeVsNaive1)
{
  arma::mat dataForTree;

  // Hard-coded filename: bad!
  if (!data::Load("test_data_3_1000.csv", dataForTree))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Set up matrices to work with.
  arma::mat dualQuery(dataForTree);
  arma::mat dualReferences(dataForTree);
  arma::mat naiveQuery(dataForTree);
  arma::mat naiveReferences(dataForTree);

  RangeSearch<> rs(dualQuery, dualReferences);

  RangeSearch<> naive(naiveQuery, naiveReferences, true);

  vector<vector<size_t> > neighborsTree;
  vector<vector<double> > distancesTree;
  rs.Search(Range(0.25, 1.05), neighborsTree, distancesTree);
  vector<vector<pair<double, size_t> > > sortedTree;
  SortResults(neighborsTree, distancesTree, sortedTree);

  vector<vector<size_t> > neighborsNaive;
  vector<vector<double> > distancesNaive;
  naive.Search(Range(0.25, 1.05), neighborsNaive, distancesNaive);
  vector<vector<pair<double, size_t> > > sortedNaive;
  SortResults(neighborsNaive, distancesNaive, sortedNaive);

  for (size_t i = 0; i < sortedTree.size(); i++)
  {
    BOOST_REQUIRE(sortedTree[i].size() == sortedNaive[i].size());

    for (size_t j = 0; j < sortedTree[i].size(); j++)
    {
      BOOST_REQUIRE(sortedTree[i][j].second == sortedNaive[i][j].second);
      BOOST_REQUIRE_CLOSE(sortedTree[i][j].first, sortedNaive[i][j].first,
          1e-5);
    }
  }
}

/**
 * Test the dual-tree nearest-neighbors method with the naive method.  This uses
 * only a reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(DualTreeVsNaive2)
{
  arma::mat dataForTree;

  // Hard-coded filename: bad!
  // Code duplication: also bad!
  if (!data::Load("test_data_3_1000.csv", dataForTree))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Set up matrices to work with (may not be necessary with no ALIAS_MATRIX?).
  arma::mat dualQuery(dataForTree);
  arma::mat naiveQuery(dataForTree);

  RangeSearch<> rs(dualQuery);

  // Set naive mode.
  RangeSearch<> naive(naiveQuery, true);

  vector<vector<size_t> > neighborsTree;
  vector<vector<double> > distancesTree;
  rs.Search(Range(0.25, 1.05), neighborsTree, distancesTree);
  vector<vector<pair<double, size_t> > > sortedTree;
  SortResults(neighborsTree, distancesTree, sortedTree);

  vector<vector<size_t> > neighborsNaive;
  vector<vector<double> > distancesNaive;
  naive.Search(Range(0.25, 1.05), neighborsNaive, distancesNaive);
  vector<vector<pair<double, size_t> > > sortedNaive;
  SortResults(neighborsNaive, distancesNaive, sortedNaive);

  for (size_t i = 0; i < sortedTree.size(); i++)
  {
    BOOST_REQUIRE(sortedTree[i].size() == sortedNaive[i].size());

    for (size_t j = 0; j < sortedTree[i].size(); j++)
    {
      BOOST_REQUIRE(sortedTree[i][j].second == sortedNaive[i][j].second);
      BOOST_REQUIRE_CLOSE(sortedTree[i][j].first, sortedNaive[i][j].first,
          1e-5);
    }
  }
}

/**
 * Test the single-tree nearest-neighbors method with the naive method.  This
 * uses only a reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(SingleTreeVsNaive)
{
  arma::mat dataForTree;

  // Hard-coded filename: bad!
  // Code duplication: also bad!
  if (!data::Load("test_data_3_1000.csv", dataForTree))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Set up matrices to work with (may not be necessary with no ALIAS_MATRIX?).
  arma::mat singleQuery(dataForTree);
  arma::mat naiveQuery(dataForTree);

  RangeSearch<> single(singleQuery, false, true);

  // Set up computation for naive mode.
  RangeSearch<> naive(naiveQuery, true);

  vector<vector<size_t> > neighborsSingle;
  vector<vector<double> > distancesSingle;
  single.Search(Range(0.25, 1.05), neighborsSingle, distancesSingle);
  vector<vector<pair<double, size_t> > > sortedTree;
  SortResults(neighborsSingle, distancesSingle, sortedTree);

  vector<vector<size_t> > neighborsNaive;
  vector<vector<double> > distancesNaive;
  naive.Search(Range(0.25, 1.05), neighborsNaive, distancesNaive);
  vector<vector<pair<double, size_t> > > sortedNaive;
  SortResults(neighborsNaive, distancesNaive, sortedNaive);

  for (size_t i = 0; i < sortedTree.size(); i++)
  {
    BOOST_REQUIRE(sortedTree[i].size() == sortedNaive[i].size());

    for (size_t j = 0; j < sortedTree[i].size(); j++)
    {
      BOOST_REQUIRE(sortedTree[i][j].second == sortedNaive[i][j].second);
      BOOST_REQUIRE_CLOSE(sortedTree[i][j].first, sortedNaive[i][j].first,
          1e-5);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
