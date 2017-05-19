/**
 * @file approx_kfn_main.cpp
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
#include <mlpack/core/util/cli.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "drusilla_select.hpp"
#include "qdafn.hpp"

using namespace mlpack;
using namespace mlpack::neighbor;
using namespace std;

PROGRAM_INFO("Approximate furthest neighbor search",
    "This program implements two strategies for furthest neighbor search. "
    "These strategies are:"
    "\n\n"
    " - The 'qdafn' algorithm from 'Approximate Furthest Neighbor in High "
    "Dimensions' by R. Pagh, F. Silvestri, J. Sivertsen, and M. Skala, in "
    "Similarity Search and Applications 2015 (SISAP)."
    "\n"
    " - The 'DrusillaSelect' algorithm from 'Fast approximate furthest "
    "neighbors with data-dependent candidate selection, by R.R. Curtin and A.B."
    " Gardner, in Similarity Search and Applications 2016 (SISAP)."
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
    "point."
    "\n\n"
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
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(type, "type");
    if (type == 0)
    {
      ar & data::CreateNVP(ds, "model");
    }
    else
    {
      ar & data::CreateNVP(qdafn, "model");
    }
  }
};

// Model loading and saving.
PARAM_MODEL_IN(ApproxKFNModel, "input_model", "File containing input model.",
    "m");
PARAM_MODEL_OUT(ApproxKFNModel, "output_model", "File to save output model to.",
    "M");

void mlpackMain()
{
  if (!CLI::HasParam("reference") && !CLI::HasParam("input_model"))
    Log::Fatal << "Either --reference_file (-r) or --input_model_file (-m) must"
        << " be specified!" << endl;
  if (CLI::HasParam("reference") && CLI::HasParam("input_model"))
    Log::Fatal << "Only one of --reference_file (-r) or --input_model_file (-m)"
        << " can be specified!" << endl;
  if (!CLI::HasParam("output_model") && !CLI::HasParam("k"))
    Log::Warn << "Neither --output_model_file (-M) nor --k (-k) are specified;"
        << " no task will be performed." << endl;
  if (!CLI::HasParam("neighbors") && !CLI::HasParam("distances") &&
      !CLI::HasParam("output_model"))
    Log::Warn << "None of --output_model_file (-M), --neighbors_file (-n), or "
        << "--distances_file (-d) are specified; no output will be saved!"
        << endl;
  if (CLI::GetParam<string>("algorithm") != "ds" &&
      CLI::GetParam<string>("algorithm") != "qdafn")
    Log::Fatal << "Unknown algorithm '" << CLI::GetParam<string>("algorithm")
        << "'; must be 'ds' or 'qdafn'!" << endl;
  if (CLI::HasParam("k") && !(CLI::HasParam("reference") ||
                              CLI::HasParam("query")))
    Log::Fatal << "If search is being performed, then either --query_file "
        << "or --reference_file must be specified!" << endl;

  if (CLI::GetParam<int>("num_tables") <= 0)
    Log::Fatal << "Invalid --num_tables value ("
        << CLI::GetParam<int>("num_tables") << "); must be greater than 0!"
        << endl;
  if (CLI::GetParam<int>("num_projections") <= 0)
    Log::Fatal << "Invalid --num_projections value ("
        << CLI::GetParam<int>("num_projections") << "); must be greater than 0!"
        << endl;

  if (CLI::HasParam("calculate_error") && !CLI::HasParam("k"))
    Log::Warn << "--calculate_error ignored because --k is not specified."
        << endl;
  if (CLI::HasParam("exact_distances") && !CLI::HasParam("calculate_error"))
    Log::Warn << "--exact_distances_file ignored beceause --calculate_error is "
        << "not specified." << endl;
  if (CLI::HasParam("calculate_error") &&
      !CLI::HasParam("exact_distances") &&
      !CLI::HasParam("reference"))
    Log::Fatal << "Cannot calculate error without either --exact_distances_file"
        << " or --reference_file specified!" << endl;

  // Do the building of a model, if necessary.
  ApproxKFNModel m;
  arma::mat referenceSet; // This may be used at query time.
  if (CLI::HasParam("reference"))
  {
    referenceSet = std::move(CLI::GetParam<arma::mat>("reference"));

    const size_t numTables = (size_t) CLI::GetParam<int>("num_tables");
    const size_t numProjections =
        (size_t) CLI::GetParam<int>("num_projections");
    const string algorithm = CLI::GetParam<string>("algorithm");

    if (algorithm == "ds")
    {
      Timer::Start("drusilla_select_construct");
      Log::Info << "Building DrusillaSelect model..." << endl;
      m.type = 0;
      m.ds = DrusillaSelect<>(referenceSet, numTables, numProjections);
      Timer::Stop("drusilla_select_construct");
    }
    else
    {
      Timer::Start("qdafn_construct");
      Log::Info << "Building QDAFN model..." << endl;
      m.type = 1;
      m.qdafn = QDAFN<>(referenceSet, numTables, numProjections);
      Timer::Stop("qdafn_construct");
    }
    Log::Info << "Model built." << endl;
  }
  else
  {
    // We must load the model from file.
    m = std::move(CLI::GetParam<ApproxKFNModel>("input_model"));
  }

  // Now, do we need to do any queries?
  if (CLI::HasParam("k"))
  {
    arma::mat querySet; // This may or may not be used.
    const size_t k = (size_t) CLI::GetParam<int>("k");

    arma::Mat<size_t> neighbors;
    arma::mat distances;

    arma::mat& set = CLI::HasParam("query") ? querySet : referenceSet;
    if (CLI::HasParam("query"))
      querySet = std::move(CLI::GetParam<arma::mat>("query"));

    if (m.type == 0)
    {
      Timer::Start("drusilla_select_search");
      Log::Info << "Searching for " << k << " furthest neighbors with "
          << "DrusillaSelect..." << endl;
      m.ds.Search(set, k, neighbors, distances);
      Timer::Stop("drusilla_select_search");
    }
    else
    {
      Timer::Start("qdafn_search");
      Log::Info << "Searching for " << k << " furthest neighbors with "
          << "QDAFN..." << endl;
      m.qdafn.Search(set, k, neighbors, distances);
      Timer::Stop("qdafn_search");
    }
    Log::Info << "Search complete." << endl;

    // Should we calculate error?
    if (CLI::HasParam("calculate_error"))
    {
      arma::mat exactDistances;
      if (CLI::HasParam("exact_distances"))
      {
        exactDistances = std::move(CLI::GetParam<arma::mat>("exact_distances"));
      }
      else
      {
        // Calculate exact distances.  We are guaranteed the reference set is
        // available.
        Log::Info << "Calculating exact distances..." << endl;
        AllkFN kfn(referenceSet);
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
    if (CLI::HasParam("neighbors"))
      CLI::GetParam<arma::Mat<size_t>>("neighbors") = std::move(neighbors);
    if (CLI::HasParam("distances"))
      CLI::GetParam<arma::mat>("distances") = std::move(distances);
  }

  // Should we save the model?
  if (CLI::HasParam("output_model"))
    CLI::GetParam<ApproxKFNModel>("output_model") = std::move(m);
}
