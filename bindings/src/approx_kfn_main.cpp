/**
 * @file methods/approx_kfn/approx_kfn_main.cpp
 * @author Ryan Curtin
 *
 * Command-line program for various furthest neighbor search algorithms.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME approx_kfn

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

#include "drusilla_select.hpp"
#include "qdafn.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_USER_NAME("Approximate furthest neighbor search");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of two strategies for furthest neighbor search.  This "
    "can be used to compute the furthest neighbor of query point(s) from a set "
    "of points; furthest neighbor models can be saved and reused with future "
    "query point(s).");

// Long description.
BINDING_LONG_DESC(
    "This program implements two strategies for furthest neighbor search. "
    "These strategies are:"
    "\n\n"
    " - The 'qdafn' algorithm from \"Approximate Furthest Neighbor in High "
    "Dimensions\" by R. Pagh, F. Silvestri, J. Sivertsen, and M. Skala, in "
    "Similarity Search and Applications 2015 (SISAP)."
    "\n"
    " - The 'DrusillaSelect' algorithm from \"Fast approximate furthest "
    "neighbors with data-dependent candidate selection\", by R.R. Curtin and "
    "A.B. Gardner, in Similarity Search and Applications 2016 (SISAP)."
    "\n\n"
    "These two strategies give approximate results for the furthest neighbor "
    "search problem and can be used as fast replacements for other furthest "
    "neighbor techniques such as those found in the mlpack_kfn program.  Note "
    "that typically, the 'ds' algorithm requires far fewer tables and "
    "projections than the 'qdafn' algorithm."
    "\n\n"
    "Specify a reference set (set to search in) with " +
    PRINT_PARAM_STRING("reference") + ", specify a query set with " +
    PRINT_PARAM_STRING("query") + ", and specify algorithm parameters with " +
    PRINT_PARAM_STRING("num_tables") + " and " +
    PRINT_PARAM_STRING("num_projections") + " (or don't and defaults will be "
    "used).  The algorithm to be used (either 'ds'---the default---or 'qdafn') "
    " may be specified with " + PRINT_PARAM_STRING("algorithm") + ".  Also "
    "specify the number of neighbors to search for with " +
    PRINT_PARAM_STRING("k") + "."
    "\n\n"
    "Note that for 'qdafn' in lower dimensions, " +
    PRINT_PARAM_STRING("num_projections") + " may need to be set to a high "
    "value in order to return results for each query point."
    "\n\n"
    "If no query set is specified, the reference set will be used as the "
    "query set.  The " + PRINT_PARAM_STRING("output_model") + " output "
    "parameter may be used to store the built model, and an input model may be "
    "loaded instead of specifying a reference set with the " +
    PRINT_PARAM_STRING("input_model") + " option."
    "\n\n"
    "Results for each query point can be stored with the " +
    PRINT_PARAM_STRING("neighbors") + " and " +
    PRINT_PARAM_STRING("distances") + " output parameters.  Each row of these "
    "output matrices holds the k distances or neighbor indices for each query "
    "point.");

// Example.
BINDING_EXAMPLE(
    "For example, to find the 5 approximate furthest neighbors with " +
    PRINT_DATASET("reference_set") + " as the reference set and " +
    PRINT_DATASET("query_set") + " as the query set using DrusillaSelect, "
    "storing the furthest neighbor indices to " + PRINT_DATASET("neighbors") +
    " and the furthest neighbor distances to " + PRINT_DATASET("distances") +
    ", one could call"
    "\n\n" +
    PRINT_CALL("approx_kfn", "query", "query_set", "reference", "reference_set",
        "k", 5, "algorithm", "ds", "neighbors", "neighbors", "distances",
        "distances") +
    "\n\n"
    "and to perform approximate all-furthest-neighbors search with k=1 on the "
    "set " + PRINT_DATASET("data") + " storing only the furthest neighbor "
    "distances to " + PRINT_DATASET("distances") + ", one could call"
    "\n\n" +
    PRINT_CALL("approx_kfn", "reference", "reference_set", "k", 1, "distances",
        "distances") +
    "\n\n"
    "A trained model can be re-used.  If a model has been previously saved to "
    + PRINT_MODEL("model") + ", then we may find 3 approximate furthest "
    "neighbors on a query set " + PRINT_DATASET("new_query_set") + " using "
    "that model and store the furthest neighbor indices into " +
    PRINT_DATASET("neighbors") + " by calling"
    "\n\n" +
    PRINT_CALL("approx_kfn", "input_model", "model", "query", "new_query_set",
        "k", 3, "neighbors", "neighbors"));

// See also...
BINDING_SEE_ALSO("k-furthest-neighbor search", "#kfn");
BINDING_SEE_ALSO("k-nearest-neighbor search", "#knn");
BINDING_SEE_ALSO("Fast approximate furthest neighbors with data-dependent"
    " candidate selection (pdf)", "http://ratml.org/pub/pdf/2016fast.pdf");
BINDING_SEE_ALSO("Approximate furthest neighbor in high dimensions (pdf)",
    "https://www.rasmuspagh.net/papers/approx-furthest-neighbor-SISAP15.pdf");
BINDING_SEE_ALSO("QDAFN class documentation",
    "@src/mlpack/methods/approx_kfn/qdafn.hpp");
BINDING_SEE_ALSO("DrusillaSelect class documentation",
    "@src/mlpack/methods/approx_kfn/drusilla_select.hpp");

PARAM_MATRIX_IN("reference", "Matrix containing the reference dataset.", "r");
PARAM_MATRIX_IN("query", "Matrix containing query points.", "q");

PARAM_INT_IN("k", "Number of furthest neighbors to search for.", "k", 0);

PARAM_INT_IN("num_tables", "Number of hash tables to use.", "t", 5);
PARAM_INT_IN("num_projections", "Number of projections to use in each hash "
    "table.", "p", 5);
PARAM_STRING_IN("algorithm", "Algorithm to use: 'ds' or 'qdafn'.", "a", "ds");

PARAM_UMATRIX_OUT("neighbors", "Matrix to save neighbor indices to.", "n");
PARAM_MATRIX_OUT("distances", "Matrix to save furthest neighbor distances to.",
    "d");

PARAM_FLAG("calculate_error", "If set, calculate the average distance error for"
    " the first furthest neighbor only.", "e");
PARAM_MATRIX_IN("exact_distances", "Matrix containing exact distances to "
    "furthest neighbors; this can be used to avoid explicit calculation when "
    "--calculate_error is set.", "x");

// If we save a model we must also save what type it is.
class ApproxKFNModel
{
 public:
  int type;
  DrusillaSelect<> ds;
  QDAFN<> qdafn;

  //! Constructor, which does nothing.
  ApproxKFNModel() : type(0), ds(1, 1), qdafn(1, 1) { }

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(type));
    if (type == 0)
    {
      ar(CEREAL_NVP(ds));
    }
    else
    {
      ar(CEREAL_NVP(qdafn));
    }
  }
};

// Model loading and saving.
PARAM_MODEL_IN(ApproxKFNModel, "input_model", "File containing input model.",
    "m");
PARAM_MODEL_OUT(ApproxKFNModel, "output_model", "File to save output model to.",
    "M");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // We have to pass either a reference set or an input model.
  RequireOnlyOnePassed(params, { "reference", "input_model" });

  // Warn if no task will be performed.
  RequireAtLeastOnePassed(params, { "reference", "k" }, false,
      "no task will be performed");

  // Warn if no output is going to be saved.
  RequireAtLeastOnePassed(params, { "neighbors", "distances", "output_model" },
      false, "no output will be saved");

  // Check that the user specified a valid algorithm.
  RequireParamInSet<string>(params, "algorithm", { "ds", "qdafn" }, true,
      "unknown algorithm");

  // If we are searching, we need a set to search in.
  if (params.Has("k"))
  {
    RequireAtLeastOnePassed(params, { "reference", "query" }, true,
        "if search is being performed, at least one set must be specified");
  }

  // Validate parameters.
  if (params.Has("k"))
  {
    RequireParamValue<int>(params, "k", [](int x) { return x > 0; }, true,
        "number of neighbors to search for must be positive");
  }
  RequireParamValue<int>(params, "num_tables", [](int x) { return x > 0; },
      true, "number of tables must be positive");
  RequireParamValue<int>(params, "num_projections", [](int x) { return x > 0; },
      true, "number of projections must be positive");

  ReportIgnoredParam(params, {{ "input_model", true }}, "algorithm");
  ReportIgnoredParam(params, {{ "input_model", true }}, "num_tables");
  ReportIgnoredParam(params, {{ "input_model", true }}, "num_projections");
  ReportIgnoredParam(params, {{ "k", false }}, "calculate_error");
  ReportIgnoredParam(params, {{ "calculate_error", false }}, "exact_distances");

  if (params.Has("calculate_error"))
  {
    RequireAtLeastOnePassed(params, { "exact_distances", "reference" }, true,
        "if error is to be calculated, either precalculated exact distances or "
        "the reference set must be passed");
  }

  if (params.Has("k") && params.Has("reference") &&
      ((size_t) params.Get<int>("k")) >
          params.Get<arma::mat>("reference").n_cols)
  {
    Log::Fatal << "Number of neighbors to search for ("
        << params.Get<int>("k") << ") must be less than the number of "
        << "reference points ("
        << params.Get<arma::mat>("reference").n_cols << ")." << std::endl;
  }

  // Do the building of a model, if necessary.
  ApproxKFNModel* m;
  arma::mat referenceSet; // This may be used at query time.
  if (params.Has("reference"))
  {
    referenceSet = std::move(params.Get<arma::mat>("reference"));
    m = new ApproxKFNModel();

    const size_t numTables = (size_t) params.Get<int>("num_tables");
    const size_t numProjections =
        (size_t) params.Get<int>("num_projections");
    const string algorithm = params.Get<string>("algorithm");

    if (algorithm == "ds")
    {
      timers.Start("drusilla_select_construct");
      Log::Info << "Building DrusillaSelect model..." << endl;
      m->type = 0;
      m->ds = DrusillaSelect<>(referenceSet, numTables, numProjections);
      timers.Stop("drusilla_select_construct");
    }
    else
    {
      timers.Start("qdafn_construct");
      Log::Info << "Building QDAFN model..." << endl;
      m->type = 1;
      m->qdafn = QDAFN<>(referenceSet, numTables, numProjections);
      timers.Stop("qdafn_construct");
    }
    Log::Info << "Model built." << endl;
  }
  else
  {
    // We must load the model from what was passed.
    m = params.Get<ApproxKFNModel*>("input_model");
  }

  // Now, do we need to do any queries?
  if (params.Has("k"))
  {
    arma::mat querySet; // This may or may not be used.
    const size_t k = (size_t) params.Get<int>("k");

    arma::Mat<size_t> neighbors;
    arma::mat distances;

    arma::mat& set = params.Has("query") ? querySet : referenceSet;
    if (params.Has("query"))
      querySet = std::move(params.Get<arma::mat>("query"));

    if (m->type == 0)
    {
      timers.Start("drusilla_select_search");
      Log::Info << "Searching for " << k << " furthest neighbors with "
          << "DrusillaSelect..." << endl;
      m->ds.Search(set, k, neighbors, distances);
      timers.Stop("drusilla_select_search");
    }
    else
    {
      timers.Start("qdafn_search");
      Log::Info << "Searching for " << k << " furthest neighbors with "
          << "QDAFN..." << endl;
      m->qdafn.Search(set, k, neighbors, distances);
      timers.Stop("qdafn_search");
    }
    Log::Info << "Search complete." << endl;

    // Should we calculate error?
    if (params.Has("calculate_error"))
    {
      arma::mat exactDistances;
      if (params.Has("exact_distances"))
      {
        // Check the exact distances matrix has the right dimensions.
        exactDistances = std::move(params.Get<arma::mat>("exact_distances"));

        if (exactDistances.n_rows != k)
        {
          delete m;
          Log::Fatal << "The number of rows in the exact distances matrix ("
              << exactDistances.n_rows << " must be equal to k (" << k << ")."
              << std::endl;
        }
        else if (exactDistances.n_cols != referenceSet.n_cols)
        {
          delete m;
          Log::Fatal << "The number of columns in the exact distances matrix ("
              << exactDistances.n_cols << ") must be equal to the number of "
              << "columns in the reference set (" << referenceSet.n_cols << ")."
              << std::endl;
        }
      }
      else
      {
        // Calculate exact distances.  We are guaranteed the reference set is
        // available.
        Log::Info << "Calculating exact distances..." << endl;
        KFN kfn(referenceSet);
        arma::Mat<size_t> exactNeighbors;
        kfn.Search(set, 1, exactNeighbors, exactDistances);
        Log::Info << "Calculation complete." << endl;
      }

      const double averageError = sum(exactDistances.row(0) /
          distances.row(0)) / distances.n_cols;
      const double minError = min(exactDistances.row(0) /
          distances.row(0));
      const double maxError = max(exactDistances.row(0) /
          distances.row(0));

      Log::Info << "Average error: " << averageError << "." << endl;
      Log::Info << "Maximum error: " << maxError << "." << endl;
      Log::Info << "Minimum error: " << minError << "." << endl;
    }

    // Save results, if desired.
    params.Get<arma::Mat<size_t>>("neighbors") = std::move(neighbors);
    params.Get<arma::mat>("distances") = std::move(distances);
  }

  params.Get<ApproxKFNModel*>("output_model") = m;
}
