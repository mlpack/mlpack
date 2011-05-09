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

// Utility function for std::pair<int, double> sorting.
bool sortPairedVector(std::pair<int, double> lhs, std::pair<int, double> rhs) {
  return lhs.second < rhs.second;
}

//
// Main method.
//
int main(int argc, char* argv[]) {
  fx_module* root = fx_init(argc, argv, &allnn_timit_doc);

  const char* inputfile = fx_param_str_req(root, "input_file");
  const char* outputfile = fx_param_str_req(root, "output_file");

  index_t knns = fx_param_int_req(root, "k");
  double alpha = fx_param_double(root, "alpha", 0.7);
  int iterations = fx_param_int(root, "iterations", 50);
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

  for(int n = 0; n < iterations; n++) {
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

    // Now we will apply random weighting vectors until we produce one which
    // improves performance.
    while(!improvement) {
      // Now we want to apply an arbitrary weighting vector
      if(new_random_vec) {
        // We produce a uniformly distributed normalized random vector with mean
        // 0.
        random_weights = arma::randu(weights.n_elem);
        random_weights -= 0.5;
        random_weights /= (norm(random_weights, 2) / initial_step);
      }

      // Let's see what happens when we apply this weight vector positively,
      // then negatively.  We will keep the number of calculations down by only
      // re-weighting and evaluating only the points which are in the set of
      // (5 * knns) we calculated earlier.  It would be more "correct" to
      // actually run the AllkNN calculation again, but that would take a long
      // time for each random vector, so we simply use this shortcut.
      int cor_pos, cor_neg;
      for (int dir = 0; dir < 2; dir++) {
        arma::mat diag_weights;
        diag_weights.zeros(weights.n_elem, weights.n_elem);

        if(dir == 0)
          diag_weights.diag() = weights +
            ((double) ((iterations - n) / (double) iterations)) *
            alpha * random_weights;
        if(dir == 1)
          diag_weights.diag() = weights -
            ((double) ((iterations - n) / (double) iterations)) *
            alpha * random_weights;

        input = diag_weights * input_preserved;

        int tmp_correct = 0;
        for (int i = 0; i < input.n_cols; i++) { // Likely slower than I want.
          std::vector<std::pair<int, double> > local_neighbors;
          for (int j = 0; j < 5 * knns; j++) {
            int ind = (i * (5 * knns)) + j;
            double dist = norm(input.col(i) - input.col(neighbors[ind]), 2);
            local_neighbors.push_back(std::make_pair(neighbors[ind], dist));
          }

          // Now sort our new vector by distance (since the order has likely
          // changed).
          std::sort(local_neighbors.begin(), local_neighbors.end(),
              sortPairedVector);
          arma::Col<index_t> tmp(local_neighbors.size());
          for(int j = 0; j < local_neighbors.size(); j++)
            tmp(j) = local_neighbors.at(j).first;

          tmp_correct += EvaluateCorrect(tmp, 5 * knns, knns, labels, i);
        }

        if (dir == 0)
          cor_pos = tmp_correct;
        if (dir == 1)
          cor_neg = tmp_correct;

        // Pick the direction that is positive
        new_random_vec = true;
        if(cor_pos >= cor_neg) {
          if(cor_pos > correct) { // apply weighting; there is an improvement
            weights += ((double) ((iterations - n) / (double) iterations))
              * alpha * random_weights;
            new_random_vec = false;
            improvement = true;
          } 
        } else {
          if(cor_neg > correct) { // apply weighting: there is an improvement
            weights -= ((double) ((iterations - n) / (double) iterations))
              * alpha * random_weights;
            new_random_vec = false;
            improvement = true;
          } 
        }
      }
    }
  }

  data::Save(outputfile, weights);

  fx_done(root);
}
