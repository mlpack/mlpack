/**
 * @file emst_main.cpp
 * @author Bill March (march@gatech.edu)
 *
 * Calls the DualTreeBoruvka algorithm from dtb.hpp.
 * Can optionally call naive Boruvka's method.
 *
 * For algorithm details, see:
 *
 * @code
 * @inproceedings{
 *   author = {March, W.B., Ram, P., and Gray, A.G.},
 *   title = {{Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis,
 *      Applications.}},
 *   booktitle = {Proceedings of the 16th ACM SIGKDD International Conference
 *      on Knowledge Discovery and Data Mining}
 *   series = {KDD 2010},
 *   year = {2010}
 * }
 * @endcode
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

#include "dtb.hpp"

#include <mlpack/core.hpp>

PROGRAM_INFO("Fast Euclidean Minimum Spanning Tree", "This program can compute "
    "the Euclidean minimum spanning tree of a set of input points using the "
    "dual-tree Boruvka algorithm."
    "\n\n"
    "The output is saved in a three-column matrix, where each row indicates an "
    "edge.  The first column corresponds to the lesser index of the edge; the "
    "second column corresponds to the greater index of the edge; and the third "
    "column corresponds to the distance between the two points.");

PARAM_STRING_REQ("input_file", "Data input file.", "i");
PARAM_STRING("output_file", "Data output file.  Stored as an edge list.", "o",
    "emst_output.csv");
PARAM_FLAG("naive", "Compute the MST using O(n^2) naive algorithm.", "n");
PARAM_INT("leaf_size", "Leaf size in the kd-tree.  One-element leaves give the "
    "empirically best performance, but at the cost of greater memory "
    "requirements.", "l", 1);

using namespace mlpack;
using namespace mlpack::emst;
using namespace mlpack::tree;
using namespace std;

int main(int argc, char* argv[])
{
  CLI::ParseCommandLine(argc, argv);

  const string dataFilename = CLI::GetParam<string>("input_file");

  arma::mat dataPoints;
  data::Load(dataFilename, dataPoints, true);

  // Do naive computation if necessary.
  if (CLI::GetParam<bool>("naive"))
  {
    Log::Info << "Running naive algorithm." << endl;

    DualTreeBoruvka<> naive(dataPoints, true);

    arma::mat naiveResults;
    naive.ComputeMST(naiveResults);

    const string outputFilename = CLI::GetParam<string>("output_file");

    data::Save(outputFilename, naiveResults, true);
  }
  else
  {
    Log::Info << "Building tree.\n";

    // Check that the leaf size is reasonable.
    if (CLI::GetParam<int>("leaf_size") <= 0)
    {
      Log::Fatal << "Invalid leaf size (" << CLI::GetParam<int>("leaf_size")
          << ")!  Must be greater than or equal to 1." << std::endl;
    }

    // Initialize the tree and get ready to compute the MST.
    const size_t leafSize = (size_t) CLI::GetParam<int>("leaf_size");
    DualTreeBoruvka<> dtb(dataPoints, false, leafSize);

    // Run the DTB algorithm.
    Log::Info << "Calculating minimum spanning tree." << endl;
    arma::mat results;
    dtb.ComputeMST(results);

    // Output the results.
    const string outputFilename = CLI::GetParam<string>("output_file");

    data::Save(outputFilename, results, true);
  }
}
