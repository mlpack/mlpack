/**
 * @file range_search_test.cpp
 * @author Ryan Curtin
 *
 * Test file for RangeSearch<> class.
 *
 * This file is part of MLPACK 1.0.11.
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
#include <mlpack/core/tree/cover_tree.hpp>
#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::range;
using namespace mlpack::math;
using namespace mlpack::tree;
using namespace mlpack::bound;
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

// Clean a tree's statistics.
template<typename TreeType>
void CleanTree(TreeType& node)
{
  node.Stat().LastDistanceNode() = NULL;
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

  typedef BinarySpaceTree<HRectBound<2>, RangeSearchStat> TreeType;

  // We will loop through three times, one for each method of performing the
  // calculation.
  arma::mat dataMutable = data;
  std::vector<size_t> oldFromNew;
  std::vector<size_t> newFromOld;
  TreeType* tree = new TreeType(dataMutable, oldFromNew, newFromOld, 1);
  for (int i = 0; i < 3; i++)
  {
    RangeSearch<>* rs;

    switch (i)
    {
      case 0: // Use the naive method.
        rs = new RangeSearch<>(dataMutable, true);
        break;
      case 1: // Use the single-tree method.
        rs = new RangeSearch<>(tree, dataMutable, true);
        break;
      case 2: // Use the dual-tree method.
        rs = new RangeSearch<>(tree, dataMutable);
        break;
    }

    // Now perform the first calculation.  Points within 0.50.
    vector<vector<size_t> > neighbors;
    vector<vector<double> > distances;
    rs->Search(Range(0.0, sqrt(0.5)), neighbors, distances);

    // Now the exhaustive check for correctness.  This will be long.
    vector<vector<pair<double, size_t> > > sortedOutput;
    SortResults(neighbors, distances, sortedOutput);

    BOOST_REQUIRE(sortedOutput[newFromOld[0]].size() == 4);
    BOOST_REQUIRE(sortedOutput[newFromOld[0]][0].second == newFromOld[2]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[0]][0].first, 0.10, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[0]][1].second == newFromOld[5]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[0]][1].first, 0.27, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[0]][2].second == newFromOld[1]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[0]][2].first, 0.30, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[0]][3].second == newFromOld[8]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[0]][3].first, 0.40, 1e-5);

    // Neighbors of point 1.
    BOOST_REQUIRE(sortedOutput[newFromOld[1]].size() == 6);
    BOOST_REQUIRE(sortedOutput[newFromOld[1]][0].second == newFromOld[8]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[1]][0].first, 0.10, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[1]][1].second == newFromOld[2]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[1]][1].first, 0.20, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[1]][2].second == newFromOld[0]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[1]][2].first, 0.30, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[1]][3].second == newFromOld[9]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[1]][3].first, 0.55, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[1]][4].second == newFromOld[5]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[1]][4].first, 0.57, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[1]][5].second == newFromOld[10]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[1]][5].first, 0.65, 1e-5);

    // Neighbors of point 2.
    BOOST_REQUIRE(sortedOutput[newFromOld[2]].size() == 4);
    BOOST_REQUIRE(sortedOutput[newFromOld[2]][0].second == newFromOld[0]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[2]][0].first, 0.10, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[2]][1].second == newFromOld[1]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[2]][1].first, 0.20, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[2]][2].second == newFromOld[8]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[2]][2].first, 0.30, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[2]][3].second == newFromOld[5]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[2]][3].first, 0.37, 1e-5);

    // Neighbors of point 3.
    BOOST_REQUIRE(sortedOutput[newFromOld[3]].size() == 2);
    BOOST_REQUIRE(sortedOutput[newFromOld[3]][0].second == newFromOld[10]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[3]][0].first, 0.25, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[3]][1].second == newFromOld[9]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[3]][1].first, 0.35, 1e-5);

    // Neighbors of point 4.
    BOOST_REQUIRE(sortedOutput[newFromOld[4]].size() == 0);

    // Neighbors of point 5.
    BOOST_REQUIRE(sortedOutput[newFromOld[5]].size() == 4);
    BOOST_REQUIRE(sortedOutput[newFromOld[5]][0].second == newFromOld[0]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[5]][0].first, 0.27, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[5]][1].second == newFromOld[2]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[5]][1].first, 0.37, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[5]][2].second == newFromOld[1]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[5]][2].first, 0.57, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[5]][3].second == newFromOld[8]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[5]][3].first, 0.67, 1e-5);

    // Neighbors of point 6.
    BOOST_REQUIRE(sortedOutput[newFromOld[6]].size() == 1);
    BOOST_REQUIRE(sortedOutput[newFromOld[6]][0].second == newFromOld[7]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[6]][0].first, 0.70, 1e-5);

    // Neighbors of point 7.
    BOOST_REQUIRE(sortedOutput[newFromOld[7]].size() == 1);
    BOOST_REQUIRE(sortedOutput[newFromOld[7]][0].second == newFromOld[6]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[7]][0].first, 0.70, 1e-5);

    // Neighbors of point 8.
    BOOST_REQUIRE(sortedOutput[newFromOld[8]].size() == 6);
    BOOST_REQUIRE(sortedOutput[newFromOld[8]][0].second == newFromOld[1]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[8]][0].first, 0.10, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[8]][1].second == newFromOld[2]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[8]][1].first, 0.30, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[8]][2].second == newFromOld[0]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[8]][2].first, 0.40, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[8]][3].second == newFromOld[9]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[8]][3].first, 0.45, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[8]][4].second == newFromOld[10]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[8]][4].first, 0.55, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[8]][5].second == newFromOld[5]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[8]][5].first, 0.67, 1e-5);

    // Neighbors of point 9.
    BOOST_REQUIRE(sortedOutput[newFromOld[9]].size() == 4);
    BOOST_REQUIRE(sortedOutput[newFromOld[9]][0].second == newFromOld[10]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[9]][0].first, 0.10, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[9]][1].second == newFromOld[3]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[9]][1].first, 0.35, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[9]][2].second == newFromOld[8]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[9]][2].first, 0.45, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[9]][3].second == newFromOld[1]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[9]][3].first, 0.55, 1e-5);

    // Neighbors of point 10.
    BOOST_REQUIRE(sortedOutput[newFromOld[10]].size() == 4);
    BOOST_REQUIRE(sortedOutput[newFromOld[10]][0].second == newFromOld[9]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[10]][0].first, 0.10, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[10]][1].second == newFromOld[3]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[10]][1].first, 0.25, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[10]][2].second == newFromOld[8]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[10]][2].first, 0.55, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[10]][3].second == newFromOld[1]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[10]][3].first, 0.65, 1e-5);

    // Now do it again with a different range: [sqrt(0.5) 1.0].
    rs->Search(Range(sqrt(0.5), 1.0), neighbors, distances);
    SortResults(neighbors, distances, sortedOutput);

    // Neighbors of point 0.
    BOOST_REQUIRE(sortedOutput[newFromOld[0]].size() == 2);
    BOOST_REQUIRE(sortedOutput[newFromOld[0]][0].second == newFromOld[9]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[0]][0].first, 0.85, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[0]][1].second == newFromOld[10]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[0]][1].first, 0.95, 1e-5);

    // Neighbors of point 1.
    BOOST_REQUIRE(sortedOutput[newFromOld[1]].size() == 1);
    BOOST_REQUIRE(sortedOutput[newFromOld[1]][0].second == newFromOld[3]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[1]][0].first, 0.90, 1e-5);

    // Neighbors of point 2.
    BOOST_REQUIRE(sortedOutput[newFromOld[2]].size() == 2);
    BOOST_REQUIRE(sortedOutput[newFromOld[2]][0].second == newFromOld[9]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[2]][0].first, 0.75, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[2]][1].second == newFromOld[10]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[2]][1].first, 0.85, 1e-5);

    // Neighbors of point 3.
    BOOST_REQUIRE(sortedOutput[newFromOld[3]].size() == 2);
    BOOST_REQUIRE(sortedOutput[newFromOld[3]][0].second == newFromOld[8]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[3]][0].first, 0.80, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[3]][1].second == newFromOld[1]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[3]][1].first, 0.90, 1e-5);

    // Neighbors of point 4.
    BOOST_REQUIRE(sortedOutput[newFromOld[4]].size() == 0);

    // Neighbors of point 5.
    BOOST_REQUIRE(sortedOutput[newFromOld[5]].size() == 0);

    // Neighbors of point 6.
    BOOST_REQUIRE(sortedOutput[newFromOld[6]].size() == 0);

    // Neighbors of point 7.
    BOOST_REQUIRE(sortedOutput[newFromOld[7]].size() == 0);

    // Neighbors of point 8.
    BOOST_REQUIRE(sortedOutput[newFromOld[8]].size() == 1);
    BOOST_REQUIRE(sortedOutput[newFromOld[8]][0].second == newFromOld[3]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[8]][0].first, 0.80, 1e-5);

    // Neighbors of point 9.
    BOOST_REQUIRE(sortedOutput[newFromOld[9]].size() == 2);
    BOOST_REQUIRE(sortedOutput[newFromOld[9]][0].second == newFromOld[2]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[9]][0].first, 0.75, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[9]][1].second == newFromOld[0]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[9]][1].first, 0.85, 1e-5);

    // Neighbors of point 10.
    BOOST_REQUIRE(sortedOutput[newFromOld[10]].size() == 2);
    BOOST_REQUIRE(sortedOutput[newFromOld[10]][0].second == newFromOld[2]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[10]][0].first, 0.85, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[10]][1].second == newFromOld[0]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[10]][1].first, 0.95, 1e-5);

    // Now do it again with a different range: [1.0 inf].
    rs->Search(Range(1.0, numeric_limits<double>::infinity()), neighbors,
        distances);
    SortResults(neighbors, distances, sortedOutput);

    // Neighbors of point 0.
    BOOST_REQUIRE(sortedOutput[newFromOld[0]].size() == 4);
    BOOST_REQUIRE(sortedOutput[newFromOld[0]][0].second == newFromOld[3]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[0]][0].first, 1.20, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[0]][1].second == newFromOld[7]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[0]][1].first, 1.35, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[0]][2].second == newFromOld[6]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[0]][2].first, 2.05, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[0]][3].second == newFromOld[4]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[0]][3].first, 5.00, 1e-5);

    // Neighbors of point 1.
    BOOST_REQUIRE(sortedOutput[newFromOld[1]].size() == 3);
    BOOST_REQUIRE(sortedOutput[newFromOld[1]][0].second == newFromOld[7]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[1]][0].first, 1.65, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[1]][1].second == newFromOld[6]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[1]][1].first, 2.35, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[1]][2].second == newFromOld[4]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[1]][2].first, 4.70, 1e-5);

    // Neighbors of point 2.
    BOOST_REQUIRE(sortedOutput[newFromOld[2]].size() == 4);
    BOOST_REQUIRE(sortedOutput[newFromOld[2]][0].second == newFromOld[3]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[2]][0].first, 1.10, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[2]][1].second == newFromOld[7]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[2]][1].first, 1.45, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[2]][2].second == newFromOld[6]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[2]][2].first, 2.15, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[2]][3].second == newFromOld[4]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[2]][3].first, 4.90, 1e-5);

    // Neighbors of point 3.
    BOOST_REQUIRE(sortedOutput[newFromOld[3]].size() == 6);
    BOOST_REQUIRE(sortedOutput[newFromOld[3]][0].second == newFromOld[2]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[3]][0].first, 1.10, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[3]][1].second == newFromOld[0]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[3]][1].first, 1.20, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[3]][2].second == newFromOld[5]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[3]][2].first, 1.47, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[3]][3].second == newFromOld[7]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[3]][3].first, 2.55, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[3]][4].second == newFromOld[6]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[3]][4].first, 3.25, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[3]][5].second == newFromOld[4]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[3]][5].first, 3.80, 1e-5);

    // Neighbors of point 4.
    BOOST_REQUIRE(sortedOutput[newFromOld[4]].size() == 10);
    BOOST_REQUIRE(sortedOutput[newFromOld[4]][0].second == newFromOld[3]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[4]][0].first, 3.80, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[4]][1].second == newFromOld[10]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[4]][1].first, 4.05, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[4]][2].second == newFromOld[9]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[4]][2].first, 4.15, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[4]][3].second == newFromOld[8]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[4]][3].first, 4.60, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[4]][4].second == newFromOld[1]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[4]][4].first, 4.70, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[4]][5].second == newFromOld[2]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[4]][5].first, 4.90, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[4]][6].second == newFromOld[0]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[4]][6].first, 5.00, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[4]][7].second == newFromOld[5]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[4]][7].first, 5.27, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[4]][8].second == newFromOld[7]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[4]][8].first, 6.35, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[4]][9].second == newFromOld[6]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[4]][9].first, 7.05, 1e-5);

    // Neighbors of point 5.
    BOOST_REQUIRE(sortedOutput[newFromOld[5]].size() == 6);
    BOOST_REQUIRE(sortedOutput[newFromOld[5]][0].second == newFromOld[7]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[5]][0].first, 1.08, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[5]][1].second == newFromOld[9]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[5]][1].first, 1.12, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[5]][2].second == newFromOld[10]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[5]][2].first, 1.22, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[5]][3].second == newFromOld[3]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[5]][3].first, 1.47, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[5]][4].second == newFromOld[6]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[5]][4].first, 1.78, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[5]][5].second == newFromOld[4]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[5]][5].first, 5.27, 1e-5);

    // Neighbors of point 6.
    BOOST_REQUIRE(sortedOutput[newFromOld[6]].size() == 9);
    BOOST_REQUIRE(sortedOutput[newFromOld[6]][0].second == newFromOld[5]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[6]][0].first, 1.78, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[6]][1].second == newFromOld[0]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[6]][1].first, 2.05, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[6]][2].second == newFromOld[2]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[6]][2].first, 2.15, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[6]][3].second == newFromOld[1]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[6]][3].first, 2.35, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[6]][4].second == newFromOld[8]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[6]][4].first, 2.45, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[6]][5].second == newFromOld[9]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[6]][5].first, 2.90, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[6]][6].second == newFromOld[10]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[6]][6].first, 3.00, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[6]][7].second == newFromOld[3]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[6]][7].first, 3.25, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[6]][8].second == newFromOld[4]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[6]][8].first, 7.05, 1e-5);

    // Neighbors of point 7.
    BOOST_REQUIRE(sortedOutput[newFromOld[7]].size() == 9);
    BOOST_REQUIRE(sortedOutput[newFromOld[7]][0].second == newFromOld[5]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[7]][0].first, 1.08, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[7]][1].second == newFromOld[0]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[7]][1].first, 1.35, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[7]][2].second == newFromOld[2]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[7]][2].first, 1.45, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[7]][3].second == newFromOld[1]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[7]][3].first, 1.65, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[7]][4].second == newFromOld[8]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[7]][4].first, 1.75, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[7]][5].second == newFromOld[9]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[7]][5].first, 2.20, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[7]][6].second == newFromOld[10]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[7]][6].first, 2.30, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[7]][7].second == newFromOld[3]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[7]][7].first, 2.55, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[7]][8].second == newFromOld[4]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[7]][8].first, 6.35, 1e-5);

    // Neighbors of point 8.
    BOOST_REQUIRE(sortedOutput[newFromOld[8]].size() == 3);
    BOOST_REQUIRE(sortedOutput[newFromOld[8]][0].second == newFromOld[7]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[8]][0].first, 1.75, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[8]][1].second == newFromOld[6]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[8]][1].first, 2.45, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[8]][2].second == newFromOld[4]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[8]][2].first, 4.60, 1e-5);

    // Neighbors of point 9.
    BOOST_REQUIRE(sortedOutput[newFromOld[9]].size() == 4);
    BOOST_REQUIRE(sortedOutput[newFromOld[9]][0].second == newFromOld[5]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[9]][0].first, 1.12, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[9]][1].second == newFromOld[7]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[9]][1].first, 2.20, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[9]][2].second == newFromOld[6]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[9]][2].first, 2.90, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[9]][3].second == newFromOld[4]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[9]][3].first, 4.15, 1e-5);

    // Neighbors of point 10.
    BOOST_REQUIRE(sortedOutput[newFromOld[10]].size() == 4);
    BOOST_REQUIRE(sortedOutput[newFromOld[10]][0].second == newFromOld[5]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[10]][0].first, 1.22, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[10]][1].second == newFromOld[7]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[10]][1].first, 2.30, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[10]][2].second == newFromOld[6]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[10]][2].first, 3.00, 1e-5);
    BOOST_REQUIRE(sortedOutput[newFromOld[10]][3].second == newFromOld[4]);
    BOOST_REQUIRE_CLOSE(sortedOutput[newFromOld[10]][3].first, 4.05, 1e-5);

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
 * Test the dual-tree range search method with the naive method.  This uses
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

  // Set up matrices to work with.
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
 * Test the single-tree range search method with the naive method.  This
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

/**
 * Ensure that dual tree range search with cover trees works by comparing 
 * with the kd-tree implementation.
 */
BOOST_AUTO_TEST_CASE(CoverTreeTest)
{
  arma::mat data;
  data.randu(8, 1000); // 1000 points in 8 dimensions.

  // Set up cover tree range search.
  typedef tree::CoverTree<metric::EuclideanDistance, tree::FirstPointIsRoot,
      RangeSearchStat> CoverTreeType;
  CoverTreeType tree(data);
  RangeSearch<metric::EuclideanDistance, CoverTreeType> coversearch(&tree,
      data);

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
    vector<vector<size_t> > kdNeighbors;
    vector<vector<double> > kdDistances;

    // Results for cover tree search.
    vector<vector<size_t> > coverNeighbors;
    vector<vector<double> > coverDistances;

    // Clean the tree statistics.
    CleanTree(tree);

    // Run the searches.
    kdsearch.Search(range, kdNeighbors, kdDistances);
    coversearch.Search(range, coverNeighbors, coverDistances);

    // Sort before comparison.
    vector<vector<pair<double, size_t> > > kdSorted;
    vector<vector<pair<double, size_t> > > coverSorted;
    SortResults(kdNeighbors, kdDistances, kdSorted);
    SortResults(coverNeighbors, coverDistances, coverSorted);

    // Now compare the results.
    for (size_t i = 0; i < kdSorted.size(); ++i)
    {
      for (size_t j = 0; j < kdSorted[i].size(); ++j)
      {
        BOOST_REQUIRE_EQUAL(kdSorted[i][j].second, coverSorted[i][j].second);
        BOOST_REQUIRE_CLOSE(kdSorted[i][j].first, coverSorted[i][j].first,
            1e-5);
      }
      BOOST_REQUIRE_EQUAL(kdSorted[i].size(), coverSorted[i].size());
    }
  }
}

/**
 * Ensure that dual tree range search with cover trees works when using
 * two datasets.
 */
BOOST_AUTO_TEST_CASE(CoverTreeTwoDatasetsTest)
{
  arma::mat data;
  data.randu(8, 1000); // 1000 points in 8 dimensions.
  arma::mat queries;
  queries.randu(8, 350); // 350 points in 8 dimensions.

  // Set up cover tree range search.
  typedef tree::CoverTree<metric::EuclideanDistance, tree::FirstPointIsRoot,
      RangeSearchStat> CoverTreeType;
  CoverTreeType tree(data);
  CoverTreeType queryTree(queries);
  RangeSearch<metric::EuclideanDistance, CoverTreeType>
      coversearch(&tree, &queryTree, data, queries);

  // Four trials with different ranges.
  for (size_t r = 0; r < 4; ++r)
  {
    // Set up kd-tree range search.  We don't have an easy way to rebuild the
    // tree, so we'll just reinstantiate it here each loop time.
    RangeSearch<> kdsearch(data, queries);

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
    vector<vector<size_t> > kdNeighbors;
    vector<vector<double> > kdDistances;

    // Results for cover tree search.
    vector<vector<size_t> > coverNeighbors;
    vector<vector<double> > coverDistances;

    // Clean the trees.
    CleanTree(tree);
    CleanTree(queryTree);

    // Run the searches.
    coversearch.Search(range, coverNeighbors, coverDistances);
    kdsearch.Search(range, kdNeighbors, kdDistances);

    // Sort before comparison.
    vector<vector<pair<double, size_t> > > kdSorted;
    vector<vector<pair<double, size_t> > > coverSorted;
    SortResults(kdNeighbors, kdDistances, kdSorted);
    SortResults(coverNeighbors, coverDistances, coverSorted);

    // Now compare the results.
    for (size_t i = 0; i < kdSorted.size(); ++i)
    {
      for (size_t j = 0; j < kdSorted[i].size(); ++j)
      {
        BOOST_REQUIRE_EQUAL(kdSorted[i][j].second, coverSorted[i][j].second);
        BOOST_REQUIRE_CLOSE(kdSorted[i][j].first, coverSorted[i][j].first,
            1e-5);
      }
      BOOST_REQUIRE_EQUAL(kdSorted[i].size(), coverSorted[i].size());
    }
  }
}

/**
 * Ensure that single-tree cover tree range search works.
 */
BOOST_AUTO_TEST_CASE(CoverTreeSingleTreeTest)
{
  arma::mat data;
  data.randu(8, 1000); // 1000 points in 8 dimensions.

  // Set up cover tree range search.
  typedef tree::CoverTree<metric::EuclideanDistance, tree::FirstPointIsRoot,
      RangeSearchStat> CoverTreeType;
  CoverTreeType tree(data);
  RangeSearch<metric::EuclideanDistance, CoverTreeType>
      coversearch(&tree, data, true);

  // Four trials with different ranges.
  for (size_t r = 0; r < 4; ++r)
  {
    // Set up kd-tree range search.
    RangeSearch<> kdsearch(data, true);

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
    vector<vector<size_t> > kdNeighbors;
    vector<vector<double> > kdDistances;

    // Results for cover tree search.
    vector<vector<size_t> > coverNeighbors;
    vector<vector<double> > coverDistances;

    // Clean the tree statistics.
    CleanTree(tree);

    // Run the searches.
    kdsearch.Search(range, kdNeighbors, kdDistances);
    coversearch.Search(range, coverNeighbors, coverDistances);

    // Sort before comparison.
    vector<vector<pair<double, size_t> > > kdSorted;
    vector<vector<pair<double, size_t> > > coverSorted;
    SortResults(kdNeighbors, kdDistances, kdSorted);
    SortResults(coverNeighbors, coverDistances, coverSorted);

    // Now compare the results.
    for (size_t i = 0; i < kdSorted.size(); ++i)
    {
      for (size_t j = 0; j < kdSorted[i].size(); ++j)
      {
        BOOST_REQUIRE_EQUAL(kdSorted[i][j].second, coverSorted[i][j].second);
        BOOST_REQUIRE_CLOSE(kdSorted[i][j].first, coverSorted[i][j].first,
            1e-5);
      }
      BOOST_REQUIRE_EQUAL(kdSorted[i].size(), coverSorted[i].size());
    }
  }
}

/**
 * Ensure that single-tree ball tree range search works.
 */
BOOST_AUTO_TEST_CASE(SingleBallTreeTest)
{
  arma::mat data;
  data.randu(8, 1000); // 1000 points in 8 dimensions.

  // Set up ball tree range search.
  typedef BinarySpaceTree<BallBound<>, RangeSearchStat> TreeType;
  TreeType tree(data);
  RangeSearch<metric::EuclideanDistance, TreeType>
      ballsearch(&tree, data, true);

  // Four trials with different ranges.
  for (size_t r = 0; r < 4; ++r)
  {
    // Set up kd-tree range search.
    RangeSearch<> kdsearch(data, true);

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
    vector<vector<size_t> > kdNeighbors;
    vector<vector<double> > kdDistances;

    // Results for ball tree search.
    vector<vector<size_t> > ballNeighbors;
    vector<vector<double> > ballDistances;

    // Clean the tree statistics.
    CleanTree(tree);

    // Run the searches.
    kdsearch.Search(range, kdNeighbors, kdDistances);
    ballsearch.Search(range, ballNeighbors, ballDistances);

    // Sort before comparison.
    vector<vector<pair<double, size_t> > > kdSorted;
    vector<vector<pair<double, size_t> > > ballSorted;
    SortResults(kdNeighbors, kdDistances, kdSorted);
    SortResults(ballNeighbors, ballDistances, ballSorted);

    // Now compare the results.
    for (size_t i = 0; i < kdSorted.size(); ++i)
    {
      for (size_t j = 0; j < kdSorted[i].size(); ++j)
      {
        BOOST_REQUIRE_EQUAL(kdSorted[i][j].second, ballSorted[i][j].second);
        BOOST_REQUIRE_CLOSE(kdSorted[i][j].first, ballSorted[i][j].first,
            1e-5);
      }
      BOOST_REQUIRE_EQUAL(kdSorted[i].size(), ballSorted[i].size());
    }
  }
}

/**
 * Ensure that dual tree range search with ball trees works by comparing 
 * with the kd-tree implementation.
 */
BOOST_AUTO_TEST_CASE(DualBallTreeTest)
{
  arma::mat data;
  data.randu(8, 1000); // 1000 points in 8 dimensions.

  // Set up ball tree range search.
  typedef BinarySpaceTree<BallBound<>, RangeSearchStat> TreeType;
  TreeType tree(data);
  RangeSearch<metric::EuclideanDistance, TreeType> ballsearch(&tree, data);

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
    vector<vector<size_t> > kdNeighbors;
    vector<vector<double> > kdDistances;

    // Results for ball tree search.
    vector<vector<size_t> > ballNeighbors;
    vector<vector<double> > ballDistances;

    // Clean the tree statistics.
    CleanTree(tree);

    // Run the searches.
    kdsearch.Search(range, kdNeighbors, kdDistances);
    ballsearch.Search(range, ballNeighbors, ballDistances);

    // Sort before comparison.
    vector<vector<pair<double, size_t> > > kdSorted;
    vector<vector<pair<double, size_t> > > ballSorted;
    SortResults(kdNeighbors, kdDistances, kdSorted);
    SortResults(ballNeighbors, ballDistances, ballSorted);

    // Now compare the results.
    for (size_t i = 0; i < kdSorted.size(); ++i)
    {
      for (size_t j = 0; j < kdSorted[i].size(); ++j)
      {
        BOOST_REQUIRE_EQUAL(kdSorted[i][j].second, ballSorted[i][j].second);
        BOOST_REQUIRE_CLOSE(kdSorted[i][j].first, ballSorted[i][j].first,
            1e-5);
      }
      BOOST_REQUIRE_EQUAL(kdSorted[i].size(), ballSorted[i].size());
    }
  }
}

/**
 * Ensure that dual tree range search with ball trees works when using
 * two datasets.
 */
BOOST_AUTO_TEST_CASE(DualBallTreeTest2)
{
  arma::mat data;
  data.randu(8, 1000); // 1000 points in 8 dimensions.

  arma::mat queries;
  queries.randu(8, 350); // 350 points in 8 dimensions.

  // Set up ball tree range search.
  typedef BinarySpaceTree<BallBound<>, RangeSearchStat> TreeType;
  TreeType tree(data);
  TreeType queryTree(queries);
  RangeSearch<metric::EuclideanDistance, TreeType>
      ballsearch(&tree, &queryTree, data, queries);

  // Four trials with different ranges.
  for (size_t r = 0; r < 4; ++r)
  {
    // Set up kd-tree range search.  We don't have an easy way to rebuild the
    // tree, so we'll just reinstantiate it here each loop time.
    RangeSearch<> kdsearch(data, queries);

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
    vector<vector<size_t> > kdNeighbors;
    vector<vector<double> > kdDistances;

    // Results for ball tree search.
    vector<vector<size_t> > ballNeighbors;
    vector<vector<double> > ballDistances;

    // Clean the trees.
    CleanTree(tree);
    CleanTree(queryTree);

    // Run the searches.
    ballsearch.Search(range, ballNeighbors, ballDistances);
    kdsearch.Search(range, kdNeighbors, kdDistances);

    // Sort before comparison.
    vector<vector<pair<double, size_t> > > kdSorted;
    vector<vector<pair<double, size_t> > > ballSorted;
    SortResults(kdNeighbors, kdDistances, kdSorted);
    SortResults(ballNeighbors, ballDistances, ballSorted);

    // Now compare the results.
    for (size_t i = 0; i < kdSorted.size(); ++i)
    {
      for (size_t j = 0; j < kdSorted[i].size(); ++j)
      {
        BOOST_REQUIRE_EQUAL(kdSorted[i][j].second, ballSorted[i][j].second);
        BOOST_REQUIRE_CLOSE(kdSorted[i][j].first, ballSorted[i][j].first,
            1e-5);
      }
      BOOST_REQUIRE_EQUAL(kdSorted[i].size(), ballSorted[i].size());
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
