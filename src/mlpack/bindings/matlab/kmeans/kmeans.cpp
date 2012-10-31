#include "mex.h"

#include <mlpack/core.hpp>

#include "kmeans.hpp"
#include "allow_empty_clusters.hpp"

using namespace mlpack;
using namespace mlpack::kmeans;
using namespace std;

/*
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
*/

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  // argument checks
  if (nrhs != 7) 
  {
    mexErrMsgTxt("Expecting seven arguments.");
  }

  if (nlhs != 1) 
  {
    mexErrMsgTxt("Output required.");
  }

  size_t seed = (size_t) mxGetScalar(prhs[6]);

  // Initialize random seed.
  //if (CLI::GetParam<int>("seed") != 0)
    //math::RandomSeed((size_t) CLI::GetParam<int>("seed"));
  if (seed != 0)
    math::RandomSeed(seed);
  else
    math::RandomSeed((size_t) std::time(NULL));

  // Now do validation of options.
  //string inputFile = CLI::GetParam<string>("inputFile");
  //int clusters = CLI::GetParam<int>("clusters");
  int clusters = (int) mxGetScalar(prhs[1]);
  if (clusters < 1)
  {
    stringstream ss;
    ss << "Invalid number of clusters requested (" << clusters << ")! "
        << "Must be greater than or equal to 1.";
    mexErrMsgTxt(ss.str().c_str());
  }

  //int maxIterations = CLI::GetParam<int>("max_iterations");
  int maxIterations = (int) mxGetScalar(prhs[2]);
  if (maxIterations < 0)
  {
    stringstream ss;
    ss << "Invalid value for maximum iterations (" << maxIterations <<
        ")! Must be greater than or equal to 0.";
    mexErrMsgTxt(ss.str().c_str());
  }

  //double overclustering = CLI::GetParam<double>("overclustering");
  double overclustering = mxGetScalar(prhs[3]);
  if (overclustering < 1)
  {
    stringstream ss;
    ss << "Invalid value for overclustering (" << overclustering <<
        ")! Must be greater than or equal to 1.";
    mexErrMsgTxt(ss.str().c_str());
  }

  const bool allow_empty_clusters = (mxGetScalar(prhs[4]) == 1.0);
  const bool fast_kmeans = (mxGetScalar(prhs[5]) == 1.0);

  /*
  // Make sure we have an output file if we're not doing the work in-place.
  if (!CLI::HasParam("in_place") && !CLI::HasParam("outputFile"))
  {
    Log::Fatal << "--outputFile not specified (and --in_place not set)."
        << std::endl;
  }
  */

  // Load our dataset.
  const size_t numPoints = mxGetN(prhs[0]);
  const size_t numDimensions = mxGetM(prhs[0]);
  arma::mat dataset(numDimensions, numPoints);
  
  // setting the values. 
  double * mexDataPoints = mxGetPr(prhs[0]);
  for (int i = 0, n = numPoints * numDimensions; i < n; ++i) 
  {
    dataset(i) = mexDataPoints[i];
  }

  // Now create the KMeans object.  Because we could be using different types,
  // it gets a little weird...
  arma::Col<size_t> assignments;

  //if (CLI::HasParam("allow_empty_clusters"))
  if (allow_empty_clusters)
  {
    KMeans<metric::SquaredEuclideanDistance, RandomPartition,
        AllowEmptyClusters> k(maxIterations, overclustering);

    //if (CLI::HasParam("fast_kmeans"))
    if (fast_kmeans)
      k.FastCluster(dataset, clusters, assignments);
    else
      k.Cluster(dataset, clusters, assignments);
  }
  else
  {
    KMeans<> k(maxIterations, overclustering);

    //if (CLI::HasParam("fast_kmeans"))
    if (fast_kmeans)
      k.FastCluster(dataset, clusters, assignments);
    else
      k.Cluster(dataset, clusters, assignments);
  }

  /*
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
  */

  // constructing matrix to return to matlab
  plhs[0] = mxCreateDoubleMatrix(assignments.n_elem, 1, mxREAL);

  // setting the values
  double * out = mxGetPr(plhs[0]);
  for (int i = 0, n = assignments.n_elem; i < n; ++i) 
  {
    out[i] = assignments(i);
  }

}

