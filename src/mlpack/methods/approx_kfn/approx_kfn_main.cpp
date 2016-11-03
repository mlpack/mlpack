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
#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
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
    "Specify a reference set (set to search in) with --reference_file, "
    "specify a query set with --query_file, and specify algorithm parameters "
    "with --num_tables (-t) and --num_projections (-p) (or don't and defaults "
    "will be used).  The algorithm to be used (either 'ds'---the default---or "
    "'qdafn') may be specified with --algorithm.  Also specify the number of "
    "neighbors to search for with --k.  Each of those options also has short "
    "names; see the detailed parameter documentation below."
    "\n\n"
    "If no query file is specified, the reference set will be used as the "
    "query set.  A model may be saved with --output_model_file (-M), and an "
    "input model may be loaded instead of specifying a reference set with "
    "--input_model_file (-m)."
    "\n\n"
    "Results for each query point are stored in the files specified by "
    "--neighbors_file and --distances_file.  This is in the same format as the "
    "mlpack_kfn and mlpack_knn programs: each row holds the k distances or "
    "neighbor indices for each query point.");

PARAM_MATRIX_IN("reference", "Matrix containing the reference dataset.", "r");
PARAM_MATRIX_IN("query", "Matrix containing query points.", "q");

// Model loading and saving.
PARAM_STRING_IN("input_model_file", "File containing input model.", "m", "");
PARAM_STRING_OUT("output_model_file", "File to save output model to.", "M");

PARAM_INT_IN("k", "Number of furthest neighbors to search for.", "k", 0);

PARAM_INT_IN("num_tables", "Number of hash tables to use.", "t", 5);
PARAM_INT_IN("num_projections", "Number of projections to use in each hash "
    "table.", "p", 5);
PARAM_STRING_IN("algorithm", "Algorithm to use: 'ds' or 'qdafn'.", "a", "ds");

PARAM_MATRIX_OUT("neighbors", "Matrix to save neighbor indices to.", "n");
PARAM_UMATRIX_OUT("distances", "Matrix to save furthest neighbor distances to.",
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

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  if (!CLI::HasParam("reference") && !CLI::HasParam("input_model_file"))
    Log::Fatal << "Either --reference_file (-r) or --input_model_file (-m) must"
        << " be specified!" << endl;
  if (CLI::HasParam("reference") && CLI::HasParam("input_model_file"))
    Log::Fatal << "Only one of --reference_file (-r) or --input_model_file (-m)"
        << " can be specified!" << endl;
  if (!CLI::HasParam("output_model_file") && !CLI::HasParam("k"))
    Log::Warn << "Neither --output_model_file (-M) nor --k (-k) are specified;"
        << " no task will be performed." << endl;
  if (!CLI::HasParam("neighbors") && !CLI::HasParam("distances") &&
      !CLI::HasParam("output_model_file"))
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
    const string inputModelFile = CLI::GetParam<string>("input_model_file");
    data::Load(inputModelFile, "approx_kfn", m);
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
        exactDistances = std::move(CLI::GetParam<string>("exact_distances"));
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
    if (CLI::HasParam("distances_file"))
      CLI::GetParam<arma::mat>("distances") = std::move(distances);
  }

  // Should we save the model?
  if (CLI::HasParam("output_model_file"))
    data::Save(CLI::GetParam<string>("output_model_file"), "approx_kfn", m);
}
