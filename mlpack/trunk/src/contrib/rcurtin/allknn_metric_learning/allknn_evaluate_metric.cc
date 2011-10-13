/***
 * @file allknn_evaluate_metric.cc
 * @author Ryan Curtin
 *
 * When we have a weighting vector generated with allknn_learn_metric, we can
 * evaluate it with this utility.
 */

#include <fastlib/fastlib.h>
#include <mlpack/neighbor_search/neighbor_search.h>

#include <armadillo>
#include <stdlib.h>
#include <ctime>

#include "allknn_metric_utils.h"

// Program documentation.
PROGRAM_INFO("AllkNN Learned Metric Evaluation",
    "Use this utility to evaluate a metric learned with the allknn_learn_metric"
    " program.  Pass in a reference and query file, and the learned weights, "
    "and the AllkNN classifier will be run and the outputs reported to the "
    "given output file.", "allknn_evaluate_metric");

// Define options.
PARAM_MODULE("allknn_evaluate_metric", "");
PARAM_STRING_REQ("reference_file", "Input CSV file.", "allknn_evaluate_metric");
PARAM_STRING_REQ("query_file", "Query CSV file.", "allknn_evaluate_metric");
PARAM_STRING_REQ("weights_file", "CSV file containing weights to evaluate.",
    "allknn_evaluate_metric");
PARAM_STRING_REQ("output_file", "Output correct counts file.", 
    "allknn_evaluate_metric");

using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::neighbor::metric;

using namespace std;

//
// Main method.
//
int main(int argc, char* argv[]) {
  IO::ParseCommandLine(argc, argv);

  string reference_file =
      IO::GetParam<string>("allknn_evaluate_metric/reference_file");
  string query_file =
      IO::GetParam<string>("allknn_evaluate_metric/query_file");
  string weights_file =
      IO::GetParam<string>("allknn_evaluate_metric/weights_file");
  string output_file =
      IO::GetParam<string>("allknn_evaluate_metric/output_file");

  arma::mat references;
  data::Load(reference_file.c_str(), references);

  // But we will ignore the last column (numeric class labels) for now
  // (save in independent object).
  arma::vec ref_labels = trans(references.row(references.n_rows - 1));
  
  // Now remove that row so we can ignore it.
  references.shed_row(references.n_rows - 1);

  arma::mat queries;
  data::Load(query_file.c_str(), queries);

  // Again, ignore the last column for now.
  arma::vec query_labels = trans(queries.row(queries.n_rows - 1));

  // And remove it.
  queries.shed_row(queries.n_rows - 1);

  // Load the weights which we will weight both matrices with.
  arma::mat weights; // This will end up being a vector.
  data::Load(weights_file.c_str(), weights);
  arma::mat diag_weights(weights.n_rows, weights.n_rows);
  diag_weights.zeros();
  diag_weights.diag() = weights.col(0);

  Log::Info << "Applying weights..." << endl;

  // Apply weights to query and reference sets.
  references = diag_weights * references;
  queries = diag_weights * queries;

  Log::Info << "Building trees..." << endl;

  // Now we need to actually run the k-nearest neighbors computation.
  AllkNN allknn(queries, references);
  
  arma::Mat<index_t> neighbors;
  arma::mat distances;

  Log::Info << "Computing neighbors..." << endl;

  allknn.ComputeNeighbors(neighbors, distances);

  // Now we want the classification score breakdown for each phoneme class.  I
  // happen to already know that we have 61 classes total.  (this will need to
  // be rewritten for generality)
  arma::Col<int> class_counts(61);
  arma::Col<int> class_scores(61);

  Log::Info << "Evaluating nearest neighbor class guesses..." << endl;

  index_t knns = IO::GetParam<int>("neighbor_search/k");

  int correct = EvaluateClassCorrect(neighbors, knns, ref_labels, query_labels,
      class_counts, class_scores);

  Log::Info << "Query set total: " << correct << " correct out of " << 
      (neighbors.n_elem / knns) << "; " << ((double) correct / (double)
      (neighbors.n_elem / knns) * 100) << "%% classified correctly." << endl;

  for (int i = 0; i < class_counts.n_elem; i++) {
    Log::Info << "Class " << i << ": " << class_scores[i] << " correct out of "
        << class_counts[i] << ": " << (((double) class_scores[i] / (double)
        class_counts[i]) * 100) << "%% classified correctly." << endl;
  }

  // Assemble our results into a matrix for exporting.
  arma::mat counts_scores(2, 62);
  for (int i = 0; i < 62; i++) {
    counts_scores(0, i) = class_counts[i];
    counts_scores(1, i) = class_scores[i];
  }

  data::Save(output_file.c_str(), trans(counts_scores));
}
