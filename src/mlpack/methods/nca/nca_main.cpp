/**
 * @file nca_main.cpp
 * @author Ryan Curtin
 *
 * Executable for Neighborhood Components Analysis.
 *
 * This file is part of MLPACK 1.0.3.
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
#include <mlpack/core/metrics/lmetric.hpp>

#include "nca.hpp"

// Define parameters.
PROGRAM_INFO("Neighborhood Components Analysis (NCA)",
    "This program implements Neighborhood Components Analysis, both a linear "
    "dimensionality reduction technique and a distance learning technique.  The"
    " method seeks to improve k-nearest-neighbor classification on a dataset "
    "by scaling the dimensions.  The method is nonparametric, and does not "
    "require a value of k.  It works by using stochastic (\"soft\") neighbor "
    "assignments and using optimization techniques over the gradient of the "
    "accuracy of the neighbor assignments.\n"
    "\n"
    "To work, this algorithm needs labeled data.  It can be given as the last "
    "row of the input dataset (--input_file), or alternatively in a separate "
    "file (--labels_file).");

PARAM_STRING_REQ("input_file", "Input dataset to run NCA on.", "i");
PARAM_STRING_REQ("output_file", "Output file for learned distance matrix.",
    "o");
PARAM_STRING("labels_file", "File of labels for input dataset.", "l", "");

using namespace mlpack;
using namespace mlpack::nca;
using namespace mlpack::metric;
using namespace std;
using namespace arma;

int main(int argc, char* argv[])
{
  // Parse command line.
  CLI::ParseCommandLine(argc, argv);

  const string inputFile = CLI::GetParam<string>("input_file");
  const string labelsFile = CLI::GetParam<string>("labels_file");
  const string outputFile = CLI::GetParam<string>("output_file");

  // Load data.
  mat data;
  data::Load(inputFile.c_str(), data, true);

  // Do we want to load labels separately?
  umat labels(data.n_cols, 1);
  if (labelsFile != "")
  {
    data::Load(labelsFile.c_str(), labels, true);

    if (labels.n_rows == 1)
      labels = trans(labels);

    if (labels.n_cols > 1)
      Log::Fatal << "Labels must have only one column or row!" << endl;
  }
  else
  {
    for (size_t i = 0; i < data.n_cols; i++)
      labels[i] = (int) data(data.n_rows - 1, i);

    data.shed_row(data.n_rows - 1);
  }

  // Now create the NCA object and run the optimization.
  NCA<LMetric<2> > nca(data, labels.unsafe_col(0));

  mat distance;
  nca.LearnDistance(distance);

  // Save the output.
  data::Save(CLI::GetParam<string>("output_file").c_str(), distance, true);
}
