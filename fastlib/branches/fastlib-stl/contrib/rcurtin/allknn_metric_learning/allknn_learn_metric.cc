/***
 * @file allknn_learn_metric.cc
 * @author Ryan Curtin
 *
 * Our objective is to learn a metric to optimize the performance of
 * all-k-nearest-neighbors.  The specific thing we are seeking is the matrix A
 * in the distance calculation d(x_i, x_j) = x_i^T * A * x_j; in this method we
 * have restricted A to being diagonal.
 *
 * A user passes in the dataset which is to be analyzed with AllkNN, an output
 * file to store the weights, and optionally a file of initial weights to start
 * from (if, for instance, a better set of weights is already known).  The
 * number of iterations performed, the number of neighbors k, and the learning
 * rate alpha can all be specified.
 *
 * This code uses the old-style FASTLIB FX system (which will need to be
 * changed).
 */

#include "allknn_learn_metric.h"

#include <armadillo>
#include <stdlib.h>
#include <ctime>
#include <map>
#include <vector>

using namespace mlpack;
using namespace mlpack::allknn;
using namespace mlpack::allknn::metric;



//
// Main method.
//
int main(int argc, char* argv[]) {
  fx_module* root = fx_init(argc, argv, &allnn_timit_doc);

  const char* inputfile = fx_param_str_req(root, "input_file");
  const char* outputfile = fx_param_str_req(root, "output_file");

  index_t knns = fx_param_int_req(root, "k");
  double alpha = fx_param_double(root, "alpha", 0.7);
  double beta = fx_param_double(root, "beta", 0.95);
  double perturbation = fx_param_double(root, "perturbation", 0.05);
  int max_iterations = fx_param_int(root, "max_iterations", 100000);
  double initial_step = fx_param_double(root, "initial_step", 1.0);

  // Initialize random numbers.
  std::srand(std::time(0));

  arma::mat input;

  data::Load(inputfile, input);

  // But we will ignore the last column (numeric class labels) for now
  // (save in independent object).
  arma::vec labels = trans(input.row(input.n_rows - 1)); // casts a subview
  
  // Now remove that row so we can ignore it.
  input.shed_row(input.n_rows - 1);
  arma::mat input_preserved = input;

  // Load the weight vector (or start from ones).  If the user specified the
  // weights file, we have faith that 
  arma::vec weights;
  if (fx_param_exists(root, "input_weights")) {
    arma::mat weights_tmp;
    const char* weights_file = fx_param_str(root, "input_weights", "");
    data::Load(weights_file, weights_tmp);

    // Sanity check to make sure our user did not pass in bad data.
    if (weights_tmp.n_cols > 1)
      NOTIFY("More than one column in %s; ignoring all but the first column",
          weights_file);
  
    weights = weights_tmp.col(0); // Take only first column.

  } else {
    // Start with a vector of ones (no weighting).
    weights.ones(input_preserved.n_rows);
  }

  // We will use this diagonal matrix for multiplication with the input.
  arma::mat diag;
  diag.zeros(weights.n_elem, weights.n_elem);

  bool new_random_vec = true; // Specifies if we should choose a new direction.

  arma::vec random_weights;

  // Loop infinitely until we cannot find an improvement (termination is later
  // in loop).
  int n = 0; // Iteration number.
  while(true) {
    // Update input matrix with the current weights.
    diag.diag() = weights;
    input = diag * input_preserved;

    // initialize AllkNN object
    arma::Col<index_t> neighbors;
    arma::vec distances;

    // We will get five times as many neighbors and we will re-evaluate them
    // after applying new weighting schemes.
    AllkNN allknn(input, 20, 5 * knns);

    // Run the computation.
    allknn.ComputeNeighbors(neighbors, distances);

    // Check classifications of the calculated neighbors (compare with the
    // labels we saved from earlier).
    int correct = EvaluateCorrect(neighbors, 5 * knns, knns, labels);

    NOTIFY("Iteration %d: %d correct out of %d; %f%% classified correctly",
        n, correct, (neighbors.n_elem / (5 * knns)),
        ((double) correct / (double) (neighbors.n_elem / (5 * knns)) * 100));

    bool improvement = false;

    // We use an FDSA gradient estimate (perturb in each dimension and estimate
    // the gradient in that direction).
    arma::vec gradient_estimate(weights.n_elem);
    arma::vec gradient_weights = weights;
    
    for (int i = 0; i < weights.n_elem; i++) {
      arma::mat diag_weights;
      diag_weights.zeros(weights.n_elem, weights.n_elem);
      diag_weights.diag() = gradient_weights;
      
      // First add the perturbation.
      diag_weights(i, i) += perturbation;
      input = diag_weights * input_preserved;

      int cor_pos = EstimateScores(input, neighbors, labels, 5 * knns, knns);

      // Now subtract the perturbation.
      diag_weights(i, i) -= 2 * perturbation;
      input = diag_weights * input_preserved;

      int cor_neg = EstimateScores(input, neighbors, labels, 5 * knns, knns);

      // The actual gradient estimate for each dimension is
      //   (cor_pos - cor_neg) / perturbation(i).
      gradient_estimate(i) = (cor_pos - cor_neg) * perturbation;
    }

    // Now, with the gradient estimate, attempt to search in this direction.
    double delta = 0.005;
    double factor = 0;
    int attempts = 1;
    arma::vec new_weights;
    while (!improvement) {
      // We will choose a search direction related to the gradient, and if it
      // doesn't work we will add a little bit of a perturbation.
      arma::vec perturbation_vec = arma::randu(weights.n_elem);
      perturbation_vec -= 0.5;
      perturbation_vec /= (norm(perturbation_vec, 2) / initial_step);

      // Calculate new weights.
      new_weights = weights + alpha * (((1 - factor) * perturbation *
          gradient_estimate) + (factor * perturbation_vec));
      arma::mat diag_new_weights;
      diag_new_weights.zeros(weights.n_elem, weights.n_elem);
      diag_new_weights.diag() = new_weights;
      input = diag_new_weights * input_preserved;

      // Try this weighting vector.
      int new_cor = EstimateScores(input, neighbors, labels, 5 * knns, knns);

      // Did it improve?
      if (new_cor > correct) {
        improvement = true;
        break;
      }

      factor += delta;
      if (factor > 1)
        factor = 1; // Fully random perturbation.
      attempts++; // Keep track of the number of attempts.

      if (attempts > max_iterations) {
        FATAL("Giving up after %d random attempts to find an improvement "
            "were unsuccessful.");
      }
    }

    NOTIFY("Improvement found after %d attempts.", attempts);

    weights = new_weights;
    alpha *= beta;
  
    data::Save(outputfile, weights);
    n++;
  }

  fx_done(root);
}
