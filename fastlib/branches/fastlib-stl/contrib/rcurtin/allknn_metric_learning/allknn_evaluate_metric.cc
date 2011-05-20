/***
 * @file allknn_evaluate_metric.cc
 * @author Ryan Curtin
 *
 * When we have a weighting vector generated with allknn_learn_metric, we can
 * evaluate it with this utility.
 */

#include "allknn_evaluate_metric.h"

#include <armadillo>
#include <stdlib.h>
#include <ctime>

using namespace mlpack;
using namespace mlpack::allknn;
using namespace mlpack::allknn::metric;

//
// Main method.
//
int main(int argc, char* argv[]) {
  fx_module* root = fx_init(argc, argv, &allnn_timit_doc);

  const char* reference_file = fx_param_str_req(NULL, "reference_file");
  const char* query_file = fx_param_str_req(NULL, "query_file");
  const char* weights_file = fx_param_str_req(NULL, "weights_file");
  const char* output_file = fx_param_str_req(NULL, "output_file");
  index_t knns = fx_param_int_req(root, "k");

  arma::mat references;
  data::Load(reference_file, references);

  // But we will ignore the last column (numeric class labels) for now
  // (save in independent object).
  arma::vec ref_labels = trans(references.row(references.n_rows - 1));
  
  // Now remove that row so we can ignore it.
  references.shed_row(references.n_rows - 1);

  arma::mat queries;
  data::Load(query_file, queries);

  // Again, ignore the last column for now.
  arma::vec query_labels = trans(queries.row(queries.n_rows - 1));

  // And remove it.
  queries.shed_row(queries.n_rows - 1);

  // Load the weights which we will weight both matrices with.
  arma::mat weights; // This will end up being a vector.
  data::Load(weights_file, weights);
  arma::mat diag_weights(weights.n_rows, weights.n_rows);
  diag_weights.zeros();
  diag_weights.diag() = weights.col(0);

  NOTIFY("Applying weights...");

  // Apply weights to query and reference sets.
  references = diag_weights * references;
  queries = diag_weights * queries;

  NOTIFY("Building trees...");

  // Now we need to actually run the k-nearest neighbors computation.
  AllkNN allknn(queries, references, 20, knns); // Don't alias the matrix.
  
  arma::Col<index_t> neighbors;
  arma::vec distances;

  NOTIFY("Computing neighbors...");

  allknn.ComputeNeighbors(neighbors, distances);

  // Now we want the classification score breakdown for each phoneme class.  I
  // happen to already know that we have 61 classes total.
  arma::Col<int> class_counts(61);
  arma::Col<int> class_scores(61);

  NOTIFY("Evaluating nearest neighbor class guesses...");

  int correct = EvaluateClassCorrect(neighbors, knns, ref_labels, query_labels,
      class_counts, class_scores);

  NOTIFY("Query set total: %d correct out of %d; %f%% classified correctly",
      correct, (neighbors.n_elem / knns), ((double) correct / (double)
      (neighbors.n_elem / knns) * 100));

  for (int i = 0; i < class_counts.n_elem; i++) {
    NOTIFY("Class %d: %d correct out of %d: %f%% classified correctly",
        i, class_scores[i], class_counts[i], ((double) class_scores[i] /
        (double) class_counts[i]) * 100);
  }

  // Assemble our results into a matrix for exporting.
  arma::mat counts_scores(2, 62);
  for (int i = 0; i < 62; i++) {
    counts_scores[0, i] = class_counts[i];
    counts_scores[1, i] = class_scores[i];
  }

  data::Save(output_file, trans(counts_scores));

  fx_done(root);
}
