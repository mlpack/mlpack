/**
 * @file methods/lsh/lsh_main.cpp
 * @author Parikshit Ram
 *
 * This file computes the approximate nearest-neighbors using 2-stable
 * Locality-sensitive Hashing.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME lsh

#include <mlpack/core/util/mlpack_main.hpp>

#include "lsh_search.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("K-Approximate-Nearest-Neighbor Search with LSH");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of approximate k-nearest-neighbor search with "
    "locality-sensitive hashing (LSH).  Given a set of reference points and a "
    "set of query points, this will compute the k approximate nearest neighbors"
    " of each query point in the reference set; models can be saved for future "
    "use.");

// Long description.
BINDING_LONG_DESC(
    "This program will calculate the k approximate-nearest-neighbors of a set "
    "of points using locality-sensitive hashing. You may specify a separate set"
    " of reference points and query points, or just a reference set which will "
    "be used as both the reference and query set. ");

// Example.
BINDING_EXAMPLE(
    "For example, the following will return 5 neighbors from the data for each "
    "point in " + PRINT_DATASET("input") + " and store the distances in " +
    PRINT_DATASET("distances") + " and the neighbors in " +
    PRINT_DATASET("neighbors") + ":"
    "\n\n" +
    PRINT_CALL("lsh", "k", 5, "reference", "input", "distances", "distances",
        "neighbors", "neighbors") +
    "\n\n"
    "The output is organized such that row i and column j in the neighbors "
    "output corresponds to the index of the point in the reference set which "
    "is the j'th nearest neighbor from the point in the query set with index "
    "i.  Row j and column i in the distances output file corresponds to the "
    "distance between those two points."
    "\n\n"
    "Because this is approximate-nearest-neighbors search, results may be "
    "different from run to run.  Thus, the " + PRINT_PARAM_STRING("seed") +
    " parameter can be specified to set the random seed."
    "\n\n"
    "This program also has many other parameters to control its functionality;"
    " see the parameter-specific documentation for more information.");

// See also...
BINDING_SEE_ALSO("@knn", "#knn");
BINDING_SEE_ALSO("@krann", "#krann");
BINDING_SEE_ALSO("Locality-sensitive hashing on Wikipedia",
    "https://en.wikipedia.org/wiki/Locality-sensitive_hashing");
BINDING_SEE_ALSO("Locality-sensitive hashing scheme based on p-stable"
    "  distributions(pdf)", "https://www.mlpack.org/papers/lsh.pdf");
BINDING_SEE_ALSO("LSHSearch C++ class documentation",
    "@src/mlpack/methods/lsh/lsh.hpp");

// Define our input parameters that this program will take.
PARAM_MATRIX_IN("reference", "Matrix containing the reference dataset.", "r");
PARAM_MATRIX_OUT("distances", "Matrix to output distances into.", "d");
PARAM_UMATRIX_OUT("neighbors", "Matrix to output neighbors into.", "n");

// We can load or save models.
PARAM_MODEL_IN(LSHSearch<>, "input_model", "Input LSH model.", "m");
PARAM_MODEL_OUT(LSHSearch<>, "output_model", "Output for trained LSH model.",
    "M");

// For testing recall.
PARAM_UMATRIX_IN("true_neighbors", "Matrix of true neighbors to compute "
    "recall with (the recall is printed when -v is specified).", "t");

PARAM_INT_IN("k", "Number of nearest neighbors to find.", "k", 0);
PARAM_MATRIX_IN("query", "Matrix containing query points (optional).", "q");

PARAM_INT_IN("projections", "The number of hash functions for each table", "K",
    10);
PARAM_INT_IN("tables", "The number of hash tables to be used.", "L", 30);
PARAM_DOUBLE_IN("hash_width", "The hash width for the first-level hashing in "
    "the LSH preprocessing. By default, the LSH class automatically estimates "
    "a hash width for its use.", "H", 0.0);
PARAM_INT_IN("num_probes", "Number of additional probes for multiprobe LSH; if "
    "0, traditional LSH is used.", "T", 0);
PARAM_INT_IN("second_hash_size", "The size of the second level hash table.",
    "S", 99901);
PARAM_INT_IN("bucket_size", "The size of a bucket in the second level hash.",
    "B", 500);
PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  if (params.Get<int>("seed") != 0)
    RandomSeed((size_t) params.Get<int>("seed"));
  else
    RandomSeed((size_t) time(NULL));

  // Get all the parameters after checking them.
  if (params.Has("k"))
  {
    RequireParamValue<int>(params, "k", [](int x) { return x > 0; }, true,
        "k must be greater than 0");
  }
  RequireParamValue<int>(params, "second_hash_size",
      [](int x) { return x > 0; }, true,
      "second hash size must be greater than 0");
  RequireParamValue<int>(params, "bucket_size", [](int x) { return x > 0; },
      true, "bucket size must be greater than 0");

  size_t k = params.Get<int>("k");
  size_t secondHashSize = params.Get<int>("second_hash_size");
  size_t bucketSize = params.Get<int>("bucket_size");

  RequireOnlyOnePassed(params, { "input_model", "reference" }, true);
  RequireAtLeastOnePassed(params, { "neighbors", "distances", "output_model" },
      false, "no results will be saved");

  if (params.Has("k"))
  {
    RequireAtLeastOnePassed(params, { "query", "reference", "input_model" },
        true, "must pass set to search");
  }

  if (params.Has("input_model") && params.Has("k") &&
      !params.Has("query"))
  {
    Log::Info << "Performing LSH-based approximate nearest neighbor search on "
        << "the reference dataset in the model stored in '"
        << params.GetPrintable<LSHSearch<>>("input_model") << "'." << endl;
  }

  ReportIgnoredParam(params, {{ "k", false }}, "neighbors");
  ReportIgnoredParam(params, {{ "k", false }}, "distances");

  ReportIgnoredParam(params, {{ "reference", false }}, "bucket_size");
  ReportIgnoredParam(params, {{ "reference", false }}, "second_hash_size");
  ReportIgnoredParam(params, {{ "reference", false }}, "hash_width");

  if (params.Has("input_model") && !params.Has("k"))
  {
    Log::Warn << PRINT_PARAM_STRING("k") << " not passed; no search will be "
        << "performed!" << std::endl;
  }

  // These declarations are here so that the matrices don't go out of scope.
  arma::mat referenceData;
  arma::mat queryData;

  // Pick up the LSH-specific parameters.
  const size_t numProj = params.Get<int>("projections");
  const size_t numTables = params.Get<int>("tables");
  const double hashWidth = params.Get<double>("hash_width");
  const size_t numProbes = (size_t) params.Get<int>("num_probes");

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  if (hashWidth == 0.0)
    Log::Info << "Using LSH with " << numProj << " projections (K) and " <<
        numTables << " tables (L) with default hash width." << endl;
  else
    Log::Info << "Using LSH with " << numProj << " projections (K) and " <<
        numTables << " tables (L) with hash width (r): " << hashWidth << endl;

  LSHSearch<>* allkann;
  if (params.Has("reference"))
  {
    // Workaround: this avoids printing load information twice for the CLI
    // bindings, where GetPrintable() will trigger a call to data::Load(),
    // which prints loading information in the middle of the Log::Info
    // message.
    (void) params.Get<arma::mat>("reference");

    allkann = new LSHSearch<>();
    Log::Info << "Using reference data from "
        << params.GetPrintable<arma::mat>("reference") << "." << endl;
    referenceData = std::move(params.Get<arma::mat>("reference"));

    timers.Start("hash_building");
    allkann->Train(std::move(referenceData), numProj, numTables, hashWidth,
        secondHashSize, bucketSize);
    timers.Stop("hash_building");
  }
  else // We must have an input model.
  {
    allkann = params.Get<LSHSearch<>*>("input_model");
  }

  if (params.Has("k"))
  {
    Log::Info << "Computing " << k << " distance approximate nearest neighbors."
        << endl;
    if (params.Has("query"))
    {
      // Workaround: avoid printing load information twice for CLI bindings.
      (void) params.Get<arma::mat>("query");

      Log::Info << "Loaded query data from "
          << params.GetPrintable<arma::mat>("query") << "." << endl;
      queryData = std::move(params.Get<arma::mat>("query"));

      timers.Start("computing_neighbors");
      allkann->Search(queryData, k, neighbors, distances, 0, numProbes);
      timers.Stop("computing_neighbors");
    }
    else
    {
      timers.Start("computing_neighbors");
      allkann->Search(k, neighbors, distances, 0, numProbes);
      timers.Stop("computing_neighbors");
    }

    Log::Info << "Neighbors computed." << endl;
  }

  // Compute recall, if desired.
  if (params.Has("true_neighbors"))
  {
    Log::Info << "Using true neighbor indices from '"
        << params.GetPrintable<arma::Mat<size_t>>("true_neighbors") << "'."
        << endl;

    // Load the true neighbors.
    arma::Mat<size_t> trueNeighbors =
        std::move(params.Get<arma::Mat<size_t>>("true_neighbors"));

    if (trueNeighbors.n_rows != neighbors.n_rows ||
        trueNeighbors.n_cols != neighbors.n_cols)
    {
      // Delete the model if needed.
      if (params.Has("reference"))
        delete allkann;
      Log::Fatal << "The true neighbors file must have the same number of "
          << "values as the set of neighbors being queried!" << endl;
    }

    // Compute recall and print it.
    double recallPercentage = 100 * allkann->ComputeRecall(neighbors,
        trueNeighbors);

    Log::Info << "Recall: " << recallPercentage << endl;
  }

  // Save output, if we did a search..
  if (params.Has("k"))
  {
    params.Get<arma::mat>("distances") = std::move(distances);
    params.Get<arma::Mat<size_t>>("neighbors") = std::move(neighbors);
  }
  params.Get<LSHSearch<>*>("output_model") = allkann;
}
