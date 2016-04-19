/**
 * @file qdafn_main.cpp
 * @author Ryan Curtin
 *
 * Command-line program for the QDAFN algorithm.
 */
#include <mlpack/core.hpp>
#include "qdafn.hpp"

using namespace qdafn;
using namespace mlpack;
using namespace std;

PROGRAM_INFO("Query-dependent approximate furthest neighbor search",
    "This program implements the algorithm from the SISAP 2015 paper titled "
    "'Approximate Furthest Neighbor in High Dimensions' by R. Pagh, F. "
    "Silvestri, J. Sivertsen, and M. Skala.  Specify a reference set (set to "
    "search in) with --reference_file, specify a query set (set to search for) "
    "with --query_file, and specify algorithm parameters with --num_tables and "
    "--num_projections (or don't, and defaults will be used).  Also specify "
    "the number of points to search for with --k.  Each of those options has "
    "short names too; see the detailed parameter documentation below."
    "\n\n"
    "Results for each query point are stored in the files specified by "
    "--neighbors_file and --distances_file.  This is in the same format as the "
    "mlpack KFN and KNN programs: each row holds the k distances or neighbor "
    "indices for each query point.");

PARAM_STRING_REQ("reference_file", "File containing reference points.", "r");
PARAM_STRING_REQ("query_file", "File containing query points.", "q");

PARAM_INT_REQ("k", "Number of furthest neighbors to search for.", "k");

PARAM_INT("num_tables", "Number of hash tables to use.", "t", 10);
PARAM_INT("num_projections", "Number of projections to use in each hash table.",
    "p", 30);

PARAM_STRING("neighbors_file", "File to save furthest neighbor indices to.",
    "n", "");
PARAM_STRING("distances_file", "File to save furthest neighbor distances to.",
    "d", "");

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  const string referenceFile = CLI::GetParam<string>("reference_file");
  const string queryFile = CLI::GetParam<string>("query_file");
  const size_t k = (size_t) CLI::GetParam<int>("k");
  const size_t numTables = (size_t) CLI::GetParam<int>("num_tables");
  const size_t numProjections = (size_t) CLI::GetParam<int>("num_projections");

  // Load the data.
  arma::mat referenceData, queryData;
  data::Load(referenceFile, referenceData, true);
  data::Load(queryFile, queryData, true);

  // Construct the object.
  Timer::Start("qdafn_construct");
  QDAFN<> q(referenceData, numTables, numProjections);
  Timer::Stop("qdafn_construct");

  // Do the search.
  arma::Mat<size_t> neighbors;
  arma::mat distances;
  Timer::Start("qdafn_search");
  q.Search(queryData, k, neighbors, distances);
  Timer::Stop("qdafn_search");

  // Save the results.
  if (CLI::HasParam("neighbors_file"))
    data::Save(CLI::GetParam<string>("neighbors_file"), neighbors);
  if (CLI::HasParam("distances_file"))
    data::Save(CLI::GetParam<string>("distances_file"), distances);
}
