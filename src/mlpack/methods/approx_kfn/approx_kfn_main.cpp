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
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "drusilla_select.hpp"
#include "qdafn.hpp"

using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_NAME("Approximate furthest neighbor search");

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
        "https://pdfs.semanticscholar.org/a4b5/7b9cbf37201fb1d9a56c0f4eefad0466"
        "9c20.pdf");
BINDING_SEE_ALSO("mlpack::neighbor::QDAFN class documentation",
        "@doxygen/classmlpack_1_1neighbor_1_1QDAFN.html");
BINDING_SEE_ALSO("mlpack::neighbor::DrusillaSelect class documentation",
        "@doxygen/classmlpack_1_1neighbor_1_1DrusillaSelect.html");

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
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(type);
    if (type == 0)
    {
      ar & BOOST_SERIALIZATION_NVP(ds);
    }
    else
    {
      ar & BOOST_SERIALIZATION_NVP(qdafn);
    }
  }
};

// Model loading and saving.
PARAM_MODEL_IN(ApproxKFNModel, "input_model", "File containing input model.",
    "m");
PARAM_MODEL_OUT(ApproxKFNModel, "output_model", "File to save output model to.",
    "M");

static void mlpackMain()
{
  // We have to pass either a reference set or an input model.
  RequireOnlyOnePassed({ "reference", "input_model" });

  // Warn if no task will be performed.
  RequireAtLeastOnePassed({ "reference", "k" }, false,
      "no task will be performed");

  // Warn if no output is going to be saved.
  RequireAtLeastOnePassed({ "neighbors", "distances", "output_model" }, false,
      "no output will be saved");

  // Check that the user specified a valid algorithm.
  RequireParamInSet<string>("algorithm", { "ds", "qdafn" }, true,
      "unknown algorithm");

  // If we are searching, we need a set to search in.
  if (IO::HasParam("k"))
  {
    RequireAtLeastOnePassed({ "reference", "query" }, true,
        "if search is being performed, at least one set must be specified");
  }

  // Validate parameters.
  if (IO::HasParam("k"))
  {
    RequireParamValue<int>("k", [](int x) { return x > 0; }, true,
        "number of neighbors to search for must be positive");
  }
  RequireParamValue<int>("num_tables", [](int x) { return x > 0; }, true,
      "number of tables must be positive");
  RequireParamValue<int>("num_projections", [](int x) { return x > 0; }, true,
      "number of projections must be positive");

  ReportIgnoredParam({{ "input_model", true }}, "algorithm");
  ReportIgnoredParam({{ "input_model", true }}, "num_tables");
  ReportIgnoredParam({{ "input_model", true }}, "num_projections");
  ReportIgnoredParam({{ "k", false }}, "calculate_error");
  ReportIgnoredParam({{ "calculate_error", false }}, "exact_distances");

  if (IO::HasParam("calculate_error"))
  {
    RequireAtLeastOnePassed({ "exact_distances", "reference" }, true,
        "if error is to be calculated, either precalculated exact distances or "
        "the reference set must be passed");
  }

  if (IO::HasParam("k") && IO::HasParam("reference") &&
      ((size_t) IO::GetParam<int>("k")) >
          IO::GetParam<arma::mat>("reference").n_cols)
  {
    Log::Fatal << "Number of neighbors to search for ("
        << IO::GetParam<int>("k") << ") must be less than the number of "
        << "reference points ("
        << IO::GetParam<arma::mat>("reference").n_cols << ")." << std::endl;
  }

  // Do the building of a model, if necessary.
  ApproxKFNModel* m;
  arma::mat referenceSet; // This may be used at query time.
  if (IO::HasParam("reference"))
  {
    referenceSet = std::move(IO::GetParam<arma::mat>("reference"));
    m = new ApproxKFNModel();

    const size_t numTables = (size_t) IO::GetParam<int>("num_tables");
    const size_t numProjections =
        (size_t) IO::GetParam<int>("num_projections");
    const string algorithm = IO::GetParam<string>("algorithm");

    if (algorithm == "ds")
    {
      Timer::Start("drusilla_select_construct");
      Log::Info << "Building DrusillaSelect model..." << endl;
      m->type = 0;
      m->ds = DrusillaSelect<>(referenceSet, numTables, numProjections);
      Timer::Stop("drusilla_select_construct");
    }
    else
    {
      Timer::Start("qdafn_construct");
      Log::Info << "Building QDAFN model..." << endl;
      m->type = 1;
      m->qdafn = QDAFN<>(referenceSet, numTables, numProjections);
      Timer::Stop("qdafn_construct");
    }
    Log::Info << "Model built." << endl;
  }
  else
  {
    // We must load the model from what was passed.
    m = IO::GetParam<ApproxKFNModel*>("input_model");
  }

  // Now, do we need to do any queries?
  if (IO::HasParam("k"))
  {
    arma::mat querySet; // This may or may not be used.
    const size_t k = (size_t) IO::GetParam<int>("k");

    arma::Mat<size_t> neighbors;
    arma::mat distances;

    arma::mat& set = IO::HasParam("query") ? querySet : referenceSet;
    if (IO::HasParam("query"))
      querySet = std::move(IO::GetParam<arma::mat>("query"));

    if (m->type == 0)
    {
      Timer::Start("drusilla_select_search");
      Log::Info << "Searching for " << k << " furthest neighbors with "
          << "DrusillaSelect..." << endl;
      m->ds.Search(set, k, neighbors, distances);
      Timer::Stop("drusilla_select_search");
    }
    else
    {
      Timer::Start("qdafn_search");
      Log::Info << "Searching for " << k << " furthest neighbors with "
          << "QDAFN..." << endl;
      m->qdafn.Search(set, k, neighbors, distances);
      Timer::Stop("qdafn_search");
    }
    Log::Info << "Search complete." << endl;

    // Should we calculate error?
    if (IO::HasParam("calculate_error"))
    {
      arma::mat exactDistances;
      if (IO::HasParam("exact_distances"))
      {
        // Check the exact distances matrix has the right dimensions.
        exactDistances = std::move(IO::GetParam<arma::mat>("exact_distances"));

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

      const double averageError = arma::sum(exactDistances.row(0) /
          distances.row(0)) / distances.n_cols;
      const double minError = arma::min(exactDistances.row(0) /
          distances.row(0));
      const double maxError = arma::max(exactDistances.row(0) /
          distances.row(0));

      Log::Info << "Average error: " << averageError << "." << endl;
      Log::Info << "Maximum error: " << maxError << "." << endl;
      Log::Info << "Minimum error: " << minError << "." << endl;
    }

    // Save results, if desired.
    IO::GetParam<arma::Mat<size_t>>("neighbors") = std::move(neighbors);
    IO::GetParam<arma::mat>("distances") = std::move(distances);
  }

  IO::GetParam<ApproxKFNModel*>("output_model") = m;
}
