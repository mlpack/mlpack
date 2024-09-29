/**
 * @file methods/kmeans/kmeans_main.cpp
 * @author Ryan Curtin
 *
 * Executable for running K-Means.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME kmeans

#include <mlpack/core/util/mlpack_main.hpp>

#include "kmeans.hpp"
#include "allow_empty_clusters.hpp"
#include "kill_empty_clusters.hpp"
#include "refined_start.hpp"
#include "kmeans_plus_plus_initialization.hpp"
#include "elkan_kmeans.hpp"
#include "hamerly_kmeans.hpp"
#include "pelleg_moore_kmeans.hpp"
#include "dual_tree_kmeans.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_USER_NAME("K-Means Clustering");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of several strategies for efficient k-means clustering. "
    "Given a dataset and a value of k, this computes and returns a k-means "
    "clustering on that data.");

// Long description.
BINDING_LONG_DESC(
    "This program performs K-Means clustering on the given dataset.  It can "
    "return the learned cluster assignments, and the centroids of the clusters."
    "  Empty clusters are not allowed by default; when a cluster becomes empty,"
    " the point furthest from the centroid of the cluster with maximum variance"
    " is taken to fill that cluster."
    "\n\n"
    "Optionally, the strategy to choose initial centroids can be specified.  "
    "The k-means++ algorithm can be used to choose initial centroids with "
    "the " + PRINT_PARAM_STRING("kmeans_plus_plus") + " parameter.  The "
    "Bradley and Fayyad approach (\"Refining initial points for k-means "
    "clustering\", 1998) can be used to select initial points by specifying "
    "the " + PRINT_PARAM_STRING("refined_start") + " parameter.  This approach "
    "works by taking random samplings of the dataset; to specify the number of "
    "samplings, the " + PRINT_PARAM_STRING("samplings") + " parameter is used, "
    "and to specify the percentage of the dataset to be used in each sample, "
    "the " + PRINT_PARAM_STRING("percentage") + " parameter is used (it should "
    "be a value between 0.0 and 1.0)."
    "\n\n"
    "There are several options available for the algorithm used for each Lloyd "
    "iteration, specified with the " + PRINT_PARAM_STRING("algorithm") + " "
    " option.  The standard O(kN) approach can be used ('naive').  Other "
    "options include the Pelleg-Moore tree-based algorithm ('pelleg-moore'), "
    "Elkan's triangle-inequality based algorithm ('elkan'), Hamerly's "
    "modification to Elkan's algorithm ('hamerly'), the dual-tree k-means "
    "algorithm ('dualtree'), and the dual-tree k-means algorithm using the "
    "cover tree ('dualtree-covertree')."
    "\n\n"
    "The behavior for when an empty cluster is encountered can be modified with"
    " the " + PRINT_PARAM_STRING("allow_empty_clusters") + " option.  When "
    "this option is specified and there is a cluster owning no points at the "
    "end of an iteration, that cluster's centroid will simply remain in its "
    "position from the previous iteration. If the " +
    PRINT_PARAM_STRING("kill_empty_clusters") + " option is specified, then "
    "when a cluster owns no points at the end of an iteration, the cluster "
    "centroid is simply filled with DBL_MAX, killing it and effectively "
    "reducing k for the rest of the computation.  Note that the default option "
    "when neither empty cluster option is specified can be time-consuming to "
    "calculate; therefore, specifying either of these parameters will often "
    "accelerate runtime."
    "\n\n"
    "Initial clustering assignments may be specified using the " +
    PRINT_PARAM_STRING("initial_centroids") + " parameter, and the maximum "
    "number of iterations may be specified with the " +
    PRINT_PARAM_STRING("max_iterations") + " parameter.");

// Example.
BINDING_EXAMPLE(
    "As an example, to use Hamerly's algorithm to perform k-means clustering "
    "with k=10 on the dataset " + PRINT_DATASET("data") + ", saving the "
    "centroids to " + PRINT_DATASET("centroids") + " and the assignments for "
    "each point to " + PRINT_DATASET("assignments") + ", the following "
    "command could be used:"
    "\n\n" +
    PRINT_CALL("kmeans", "input", "data", "clusters", 10, "output",
        "assignments", "centroid", "centroids") +
    "\n\n"
    "To run k-means on that same dataset with initial centroids specified in " +
    PRINT_DATASET("initial") + " with a maximum of 500 iterations, "
    "storing the output centroids in " + PRINT_DATASET("final") + " the "
    "following command may be used:"
    "\n\n" +
    PRINT_CALL("kmeans", "input", "data", "initial_centroids", "initial",
        "clusters", 10, "max_iterations", 500, "centroid", "final"));

// See also...
BINDING_SEE_ALSO("@dbscan", "#dbscan");
BINDING_SEE_ALSO("k-means++", "https://en.wikipedia.org/wiki/K-means%2B%2B");
BINDING_SEE_ALSO("Using the triangle inequality to accelerate k-means (pdf)",
    "https://cdn.aaai.org/ICML/2003/ICML03-022.pdf");
BINDING_SEE_ALSO("Making k-means even faster (pdf)",
    "https://www.cse.iitd.ac.in/~rjaiswal/2015/col870/Project/Faster-k-means/"
    "Hamerly.pdf");
BINDING_SEE_ALSO("Accelerating exact k-means algorithms with geometric"
    " reasoning (pdf)", "http://reports-archive.adm.cs.cmu.edu/anon/anon/usr/"
    "ftp/usr0/ftp/2000/CMU-CS-00-105.pdf");
BINDING_SEE_ALSO("A dual-tree algorithm for fast k-means clustering with large "
    "k (pdf)", "http://www.ratml.org/pub/pdf/2017dual.pdf");
BINDING_SEE_ALSO("KMeans class documentation",
    "@src/mlpack/methods/kmeans/kmeans.hpp");

// Required options.
PARAM_MATRIX_IN_REQ("input", "Input dataset to perform clustering on.", "i");
PARAM_INT_IN_REQ("clusters", "Number of clusters to find (0 autodetects from "
    "initial centroids).", "c");

// Output options.
PARAM_FLAG("in_place", "If specified, a column containing the learned cluster "
    "assignments will be added to the input dataset file.  In this case, "
    "--output_file is overridden. (Do not use in Python.)", "P");
PARAM_MATRIX_OUT("output", "Matrix to store output labels or labeled data to.",
    "o");
PARAM_MATRIX_OUT("centroid", "If specified, the centroids of each cluster will "
    " be written to the given file.", "C");

// k-means configuration options.
PARAM_FLAG("allow_empty_clusters", "Allow empty clusters to be persist.", "e");
PARAM_FLAG("kill_empty_clusters", "Remove empty clusters when they occur.",
    "E");
PARAM_FLAG("labels_only", "Only output labels into output file.", "l");
PARAM_INT_IN("max_iterations", "Maximum number of iterations before k-means "
    "terminates.", "m", 1000);
PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);
PARAM_MATRIX_IN("initial_centroids", "Start with the specified initial "
    "centroids.", "I");

// Parameters for "refined start" k-means.
PARAM_FLAG("refined_start", "Use the refined initial point strategy by Bradley "
    "and Fayyad to choose initial points.", "r");
PARAM_INT_IN("samplings", "Number of samplings to perform for refined start "
    "(use when --refined_start is specified).", "S", 100);
PARAM_DOUBLE_IN("percentage", "Percentage of dataset to use for each refined "
    "start sampling (use when --refined_start is specified).", "p", 0.02);
PARAM_FLAG("kmeans_plus_plus", "Use the k-means++ initialization strategy to "
    "choose initial points.", "K");

PARAM_STRING_IN("algorithm", "Algorithm to use for the Lloyd iteration "
    "('naive', 'pelleg-moore', 'elkan', 'hamerly', 'dualtree', or "
    "'dualtree-covertree').", "a", "naive");

// Given the type of initial partition policy, figure out the empty cluster
// policy and run k-means.
template<typename InitialPartitionPolicy>
void FindEmptyClusterPolicy(util::Params& params,
                            util::Timers& timers,
                            const InitialPartitionPolicy& ipp);

// Given the initial partitionining policy and empty cluster policy, figure out
// the Lloyd iteration step type and run k-means.
template<typename InitialPartitionPolicy, typename EmptyClusterPolicy>
void FindLloydStepType(util::Params& params,
                       util::Timers& timers,
                       const InitialPartitionPolicy& ipp);

// Given the template parameters, sanitize/load input and run k-means.
template<typename InitialPartitionPolicy,
         typename EmptyClusterPolicy,
         template<class, class> class LloydStepType>
void RunKMeans(util::Params& params,
               util::Timers& timers,
               const InitialPartitionPolicy& ipp);

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // Initialize random seed.
  if (params.Get<int>("seed") != 0)
    RandomSeed((size_t) params.Get<int>("seed"));
  else
    RandomSeed((size_t) std::time(NULL));

  RequireOnlyOnePassed(params, { "refined_start", "kmeans_plus_plus" }, true,
      "Only one initialization strategy can be specified!", true);

  // Now, start building the KMeans type that we'll be using.  Start with the
  // initial partition policy.  The call to FindEmptyClusterPolicy<> results in
  // a call to RunKMeans<> and the algorithm is completed.
  if (params.Has("refined_start"))
  {
    RequireParamValue<int>(params, "samplings", [](int x) { return x > 0; },
        true, "number of samplings must be positive");
    const int samplings = params.Get<int>("samplings");
    RequireParamValue<double>(params, "percentage",
        [](double x) { return x > 0.0 && x <= 1.0; }, true, "percentage to "
        "sample must be greater than 0.0 and less than or equal to 1.0");
    const double percentage = params.Get<double>("percentage");

    FindEmptyClusterPolicy<RefinedStart>(params, timers,
        RefinedStart(samplings, percentage));
  }
  else if (params.Has("kmeans_plus_plus"))
  {
    FindEmptyClusterPolicy<KMeansPlusPlusInitialization>(params, timers,
        KMeansPlusPlusInitialization());
  }
  else
  {
    FindEmptyClusterPolicy<SampleInitialization>(params, timers,
        SampleInitialization());
  }
}

// Given the type of initial partition policy, figure out the empty cluster
// policy and run k-means.
template<typename InitialPartitionPolicy>
void FindEmptyClusterPolicy(util::Params& params,
                            util::Timers& timers,
                            const InitialPartitionPolicy& ipp)
{
  if (params.Has("allow_empty_clusters") ||
      params.Has("kill_empty_clusters"))
  {
    RequireOnlyOnePassed(params, { "allow_empty_clusters",
        "kill_empty_clusters" }, true);
  }

  if (params.Has("allow_empty_clusters"))
  {
    FindLloydStepType<InitialPartitionPolicy, AllowEmptyClusters>(params,
        timers, ipp);
  }
  else if (params.Has("kill_empty_clusters"))
  {
    FindLloydStepType<InitialPartitionPolicy, KillEmptyClusters>(params, timers,
        ipp);
  }
  else
  {
    FindLloydStepType<InitialPartitionPolicy, MaxVarianceNewCluster>(params,
        timers, ipp);
  }
}

// Given the initial partitionining policy and empty cluster policy, figure out
// the Lloyd iteration step type and run k-means.
template<typename InitialPartitionPolicy, typename EmptyClusterPolicy>
void FindLloydStepType(util::Params& params,
                       util::Timers& timers,
                       const InitialPartitionPolicy& ipp)
{
  RequireParamInSet<string>(params, "algorithm", { "elkan", "hamerly",
      "pelleg-moore", "dualtree", "dualtree-covertree", "naive" }, true,
      "unknown k-means algorithm");

  const string algorithm = params.Get<string>("algorithm");
  if (algorithm == "elkan")
  {
    RunKMeans<InitialPartitionPolicy, EmptyClusterPolicy, ElkanKMeans>(params,
        timers, ipp);
  }
  else if (algorithm == "hamerly")
  {
    RunKMeans<InitialPartitionPolicy, EmptyClusterPolicy, HamerlyKMeans>(
        params, timers, ipp);
  }
  else if (algorithm == "pelleg-moore")
  {
    RunKMeans<InitialPartitionPolicy, EmptyClusterPolicy,
        PellegMooreKMeans>(params, timers, ipp);
  }
  else if (algorithm == "dualtree")
  {
    RunKMeans<InitialPartitionPolicy, EmptyClusterPolicy,
        DefaultDualTreeKMeans>(params, timers, ipp);
  }
  else if (algorithm == "dualtree-covertree")
  {
    RunKMeans<InitialPartitionPolicy, EmptyClusterPolicy,
        CoverTreeDualTreeKMeans>(params, timers, ipp);
  }
  else if (algorithm == "naive")
  {
    RunKMeans<InitialPartitionPolicy, EmptyClusterPolicy, NaiveKMeans>(params,
        timers, ipp);
  }
}

// Given the template parameters, sanitize/load input and run k-means.
template<typename InitialPartitionPolicy,
         typename EmptyClusterPolicy,
         template<class, class> class LloydStepType>
void RunKMeans(util::Params& params,
               util::Timers& timers,
               const InitialPartitionPolicy& ipp)
{
  // Now, do validation of input options.
  if (!params.Has("initial_centroids"))
  {
    RequireParamValue<int>(params, "clusters", [](int x) { return x > 0; },
        true, "number of clusters must be positive");
  }
  else
  {
    ReportIgnoredParam(params, {{ "initial_centroids", true }}, "clusters");
  }

  int clusters = params.Get<int>("clusters");
  if (clusters == 0 && params.Has("initial_centroids"))
  {
    Log::Info << "Detecting number of clusters automatically from input "
        << "centroids." << endl;
  }

  RequireParamValue<int>(params, "max_iterations", [](int x) { return x >= 0; },
      true, "maximum iterations must be positive or 0 (for no limit)");
  const int maxIterations = params.Get<int>("max_iterations");

  // Make sure we have an output file if we're not doing the work in-place.
  RequireOnlyOnePassed(params, { "in_place", "output", "centroid" }, false,
      "no results will be saved");

  arma::mat dataset = params.Get<arma::mat>("input");  // Load our dataset.
  arma::mat centroids;

  const bool initialCentroidGuess = params.Has("initial_centroids");
  // Load initial centroids if the user asked for it.
  if (initialCentroidGuess)
  {
    centroids = std::move(params.Get<arma::mat>("initial_centroids"));
    if (clusters == 0)
      clusters = centroids.n_cols;

    ReportIgnoredParam(params, {{ "refined_start", true }},
        "initial_centroids");

    if (!params.Has("refined_start"))
      Log::Info << "Using initial centroid guesses." << endl;
  }

  timers.Start("clustering");
  KMeans<EuclideanDistance,
         InitialPartitionPolicy,
         EmptyClusterPolicy,
         LloydStepType> kmeans(maxIterations, EuclideanDistance(), ipp);

  if (params.Has("output") || params.Has("in_place"))
  {
    // We need to get the assignments.
    arma::Row<size_t> assignments;
    kmeans.Cluster(dataset, clusters, assignments, centroids,
        false, initialCentroidGuess);
    timers.Stop("clustering");

    // Now figure out what to do with our results.
    if (params.Has("in_place"))
    {
      // Add the column of assignments to the dataset; but we have to convert
      // them to type double first.
      arma::rowvec converted(assignments.n_elem);
      for (size_t i = 0; i < assignments.n_elem; ++i)
        converted(i) = (double) assignments(i);

      dataset.insert_rows(dataset.n_rows, converted);

      // Save the dataset.
      params.MakeInPlaceCopy("output", "input");
      params.Get<arma::mat>("output") = std::move(dataset);
    }
    else
    {
      if (params.Has("labels_only"))
      {
        // Save only the labels.  TODO: figure out how to get this to output an
        // arma::Mat<size_t> instead of an arma::mat.
        params.Get<arma::mat>("output") =
            ConvTo<arma::mat>::From(assignments);
      }
      else
      {
        // Convert the assignments to doubles.
        arma::rowvec converted(assignments.n_elem);
        for (size_t i = 0; i < assignments.n_elem; ++i)
          converted(i) = (double) assignments(i);

        dataset.insert_rows(dataset.n_rows, converted);

        // Now save, in the different file.
        params.Get<arma::mat>("output") = std::move(dataset);
      }
    }
  }
  else
  {
    // Just save the centroids.
    kmeans.Cluster(dataset, clusters, centroids, initialCentroidGuess);
    timers.Stop("clustering");
  }

  // Should we write the centroids to a file?
  if (params.Has("centroid"))
    params.Get<arma::mat>("centroid") = std::move(centroids);
}
