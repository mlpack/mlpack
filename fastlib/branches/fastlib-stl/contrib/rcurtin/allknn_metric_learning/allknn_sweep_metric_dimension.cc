/***
 * @file allknn_sweep_metric_dimension.cc
 * @author Ryan Curtin
 *
 * This utility sweeps a single dimension of an AllkNN metric and performs the
 * computation for each value in the specified range of values.  So, for
 * instance, we could evaluate AllkNN on a 30-dimensional dataset where we are
 * testing all weights on dimension 12 from 0:0.02:5 (we can also specify step
 * size).
 */

#include "allknn_sweep_metric_dimension.h"

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

  const char* inputfile = fx_param_str_req(NULL, "input_file");
  index_t knns = fx_param_int_req(root, "k");
  int dim = fx_param_int_req(root, "dim");
  double step = fx_param_double(root, "step", 0.01);
  double spread = fx_param_double(root, "spread", 0.5);
  const char* output_file = fx_param_str_req(NULL, "output_file");

  arma::mat input;
  data::Load(inputfile, input);

  // But we will ignore the last column (numeric class labels) for now
  // (save in independent object).
  arma::vec labels = trans(input.row(input.n_rows - 1)); // casts a subview
  
  // Now remove that row so we can ignore it.
  input.shed_row(input.n_rows - 1);
  arma::mat input_preserved = input;

  arma::vec weights;
  weights.ones(input_preserved.n_rows);
  arma::mat diag;
  diag.zeros(weights.n_elem, weights.n_elem);

  int iterations = (int) std::ceil((2 * spread) / step) + 1;
  arma::vec results(iterations);

  for(int it = 0; it < iterations; it++) {
    double value = (1 - spread) + (it * step);

    // update input matrix
    weights(dim) = value;
    diag.diag() = weights;
    input = diag * input_preserved;

    // initialize AllkNN object
    arma::Col<index_t> neighbors;
    arma::vec distances;
    
    // get five times as many neighbors and we will re-evaluate them after
    // applying new weighting schemes
    AllkNN allknn(input, 20, knns);
  
    // run
    allknn.ComputeNeighbors(neighbors, distances);

    // check classifications
    int correct = EvaluateCorrect(neighbors, knns, knns, labels);

    NOTIFY("Dimension %d with value %lf: %d correct out of %d; %f%% classified correctly",
      dim, value, correct, (neighbors.n_elem / knns), ((double) correct / (double)
      (neighbors.n_elem / knns) * 100));

    results(it) = correct;
  }

  data::Save(output_file, results);

  fx_done(root);
}
