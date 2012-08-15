/**
 * @file kmeans_main.cpp
 * @author Ryan Curtin
 *
 * Executable for running K-Means.
 *
 * This file is part of MLPACK 1.0.2.
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

#include "kmeans.hpp"
#include "allow_empty_clusters.hpp"

using namespace mlpack;
using namespace mlpack::kmeans;
using namespace std;

// Define parameters for the executable.
PROGRAM_INFO("K-Means Clustering", "This program performs K-Means clustering "
    "on the given dataset, storing the learned cluster assignments either as "
    "a column of labels in the file containing the input dataset or in a "
    "separate file.  Empty clusters are not allowed by default; when a cluster "
    "becomes empty, the point furthest from the centroid of the cluster with "
    "maximum variance is taken to fill that cluster.");

PARAM_STRING_REQ("inputFile", "Input dataset to perform clustering on.", "i");
PARAM_INT_REQ("clusters", "Number of clusters to find.", "c");

PARAM_FLAG("in_place", "If specified, a column of the learned cluster "
    "assignments will be added to the input dataset file.  In this case, "
    "--outputFile is not necessary.", "p");
PARAM_STRING("outputFile", "File to write output labels or labeled data to.",
    "o", "output.csv");
PARAM_FLAG("allow_empty_clusters", "Allow empty clusters to be created.", "e");
PARAM_FLAG("labels_only", "Only output labels into output file.", "l");
PARAM_DOUBLE("overclustering", "Finds (overclustering * clusters) clusters, "
    "then merges them together until only the desired number of clusters are "
    "left.", "O", 1.0);
PARAM_INT("max_iterations", "Maximum number of iterations before K-Means "
    "terminates.", "m", 1000);
PARAM_INT("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);
PARAM_FLAG("fast_kmeans", "Use the experimental fast k-means algorithm by Pelleg and Moore", "f")

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  // Initialize random seed.
  if (CLI::GetParam<int>("seed") != 0)
    math::RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    math::RandomSeed((size_t) std::time(NULL));

  // Now do validation of options.
  string inputFile = CLI::GetParam<string>("inputFile");
  int clusters = CLI::GetParam<int>("clusters");
  if (clusters < 1)
  {
    Log::Fatal << "Invalid number of clusters requested (" << clusters << ")! "
        << "Must be greater than or equal to 1." << std::endl;
  }

  int maxIterations = CLI::GetParam<int>("max_iterations");
  if (maxIterations < 0)
  {
    Log::Fatal << "Invalid value for maximum iterations (" << maxIterations <<
        ")! Must be greater than or equal to 0." << std::endl;
  }

  double overclustering = CLI::GetParam<double>("overclustering");
  if (overclustering < 1)
  {
    Log::Fatal << "Invalid value for overclustering (" << overclustering <<
        ")! Must be greater than or equal to 1." << std::endl;
  }

  // Make sure we have an output file if we're not doing the work in-place.
  if (!CLI::HasParam("in_place") && !CLI::HasParam("outputFile"))
  {
    Log::Fatal << "--outputFile not specified (and --in_place not set)."
        << std::endl;
  }

  // Load our dataset.
  arma::mat dataset;
  data::Load(inputFile.c_str(), dataset);

  // Now create the KMeans object.  Because we could be using different types,
  // it gets a little weird...
  arma::Col<size_t> assignments;

  if (CLI::HasParam("allow_empty_clusters"))
  {
    KMeans<metric::SquaredEuclideanDistance, RandomPartition,
        AllowEmptyClusters> k(maxIterations, overclustering);

    Timer::Start("clustering");
		if(CLI::HasParam("fast_kmeans"))
			k.FastCluster(dataset, clusters, assignments);
		else
			k.Cluster(dataset, clusters, assignments);
    Timer::Stop("clustering");
  }
  else
  {
    KMeans<> k(maxIterations, overclustering);

    Timer::Start("clustering");
		if(CLI::HasParam("fast_kmeans"))
			k.FastCluster(dataset, clusters, assignments);
		else
			k.Cluster(dataset, clusters, assignments);
    Timer::Stop("clustering");
  }

  // Now figure out what to do with our results.
  if (CLI::HasParam("in_place"))
  {
    // Add the column of assignments to the dataset; but we have to convert them
    // to type double first.
    arma::vec converted(assignments.n_elem);
    for (size_t i = 0; i < assignments.n_elem; i++)
      converted(i) = (double) assignments(i);

    dataset.insert_rows(dataset.n_rows, trans(converted));

    // Save the dataset.
    data::Save(inputFile.c_str(), dataset);
  }
  else
  {
    if (CLI::HasParam("labels_only"))
    {
      // Save only the labels.
      string outputFile = CLI::GetParam<string>("outputFile");
      arma::Mat<size_t> output = trans(assignments);
      data::Save(outputFile.c_str(), output);
    }
    else
    {
      // Convert the assignments to doubles.
      arma::vec converted(assignments.n_elem);
      for (size_t i = 0; i < assignments.n_elem; i++)
        converted(i) = (double) assignments(i);

      dataset.insert_rows(dataset.n_rows, trans(converted));

      // Now save, in the different file.
      string outputFile = CLI::GetParam<string>("outputFile");
      data::Save(outputFile.c_str(), dataset);
    }
  }
}

