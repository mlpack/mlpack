/**
 * @file methods/tsne/tsne_main.cpp
 * @author Kiner Shah
 *
 * Main executable to run t-SNE.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME tsne

#include <mlpack/core/util/mlpack_main.hpp>

#include "tsne.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_USER_NAME("t-Distributed Stochastic Neighbor Embedding (t-SNE)");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of t-Distributed Stochastic Neighbor Embedding (t-SNE), "
    "a nonlinear dimensionality reduction technique that is particularly well "
    "suited for embedding high-dimensional data for visualization in a "
    "low-dimensional space of two or three dimensions.");

// Long description.
BINDING_LONG_DESC(
    "This program performs t-Distributed Stochastic Neighbor Embedding (t-SNE) "
    "on the given dataset, reducing it to the specified number of dimensions "
    "(default 2). t-SNE is a nonlinear dimensionality reduction technique that "
    "is particularly well suited for the visualization of high-dimensional "
    "datasets."
    "\n\n"
    "The algorithm minimizes the divergence between the distribution of pairwise "
    "similarities in the high-dimensional space and the distribution of pairwise "
    "similarities in the low-dimensional embedding. The cost function is "
    "minimized using gradient descent."
    "\n\n"
    "The " + PRINT_PARAM_STRING("perplexity") + " parameter is related to the "
    "number of nearest neighbors that is used in other manifold learning "
    "algorithms. Larger datasets usually require a larger perplexity. Consider "
    "selecting a value between 5 and 50. The perplexity parameter balances "
    "attention between local and global aspects of the data."
    "\n\n"
    "The " + PRINT_PARAM_STRING("learning_rate") + " parameter controls the "
    "step size during gradient descent optimization. If the learning rate is too "
    "high, the data may look like a ball with any point approximately equidistant "
    "from its nearest neighbors. If the learning rate is too low, most points may "
    "look compressed in a dense cloud with few outliers."
    "\n\n"
    "The " + PRINT_PARAM_STRING("max_iterations") + " parameter specifies the "
    "maximum number of iterations for the optimization procedure."
    "\n\n"
    "The " + PRINT_PARAM_STRING("early_exaggeration") + " parameter controls the "
    "tightness of clusters in the embedding. During early exaggeration, the "
    "values in P are multiplied by this coefficient to allow clusters to move "
    "around more freely.");

// Example.
BINDING_EXAMPLE(
    "For example, to reduce a dataset " + PRINT_DATASET("data") + " to 2 "
    "dimensions using t-SNE with perplexity 30 and learning rate 200, storing "
    "the output to " + PRINT_DATASET("output") + ", the following command can "
    "be used:"
    "\n\n" +
    PRINT_CALL("tsne", "input", "data", "output", "output", "perplexity", 30,
        "learning_rate", 200));

// See also...
BINDING_SEE_ALSO("t-SNE on Wikipedia",
    "https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding");
BINDING_SEE_ALSO("Visualizing Data using t-SNE (pdf)",
    "http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf");
BINDING_SEE_ALSO("TSNE C++ class documentation",
    "@doc/user/methods/tsne.md");

// Parameters for program.
PARAM_MATRIX_IN_REQ("input", "Input dataset to perform t-SNE on.", "i");
PARAM_MATRIX_OUT("output", "Matrix to save output embedding to.", "o");
PARAM_INT_IN("new_dimensionality", "Desired dimensionality of output dataset. "
    "Must be no greater than the dimensionality of the input dataset.", "d", 2);

PARAM_DOUBLE_IN("perplexity", "Perplexity of the Gaussian kernel used to "
    "compute affinities. This parameter balances attention between local and "
    "global aspects of the data. Typical values are between 5 and 50.", "p",
    30.0);
PARAM_DOUBLE_IN("learning_rate", "Learning rate for the gradient descent "
    "optimization.", "l", 200.0);
PARAM_INT_IN("max_iterations", "Maximum number of iterations for the "
    "optimization.", "n", 1000);
PARAM_DOUBLE_IN("early_exaggeration", "Coefficient for early exaggeration "
    "(>= 1.0). During early exaggeration, the values in P are multiplied by "
    "this coefficient.", "e", 12.0);
PARAM_INT_IN("random_seed", "Random seed. If 0, 'std::time(NULL)' is used.",
    "s", 0);

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // Load input dataset.
  arma::mat& dataset = params.Get<arma::mat>("input");

  // Issue a warning if the user did not specify an output file.
  RequireAtLeastOnePassed(params, { "output" }, false,
      "no output will be saved");

  // Get parameters.
  const size_t newDimension = (size_t) params.Get<int>("new_dimensionality");
  const double perplexity = params.Get<double>("perplexity");
  const double learningRate = params.Get<double>("learning_rate");
  const size_t maxIterations = (size_t) params.Get<int>("max_iterations");
  const double earlyExaggeration = params.Get<double>("early_exaggeration");
  const int randomSeed = params.Get<int>("random_seed");

  // Validate parameters.
  RequireParamValue<int>(params, "new_dimensionality",
      [](int x) { return x > 0; }, true,
      "new dimensionality must be positive");
  RequireParamValue<int>(params, "new_dimensionality",
      [dataset](int x) { return x <= (int) dataset.n_rows; }, true,
      "new dimensionality cannot be greater than existing dimensionality");

  RequireParamValue<double>(params, "perplexity",
      [](double x) { return x > 0.0; }, true,
      "perplexity must be positive");
  RequireParamValue<double>(params, "learning_rate",
      [](double x) { return x > 0.0; }, true,
      "learning rate must be positive");
  RequireParamValue<int>(params, "max_iterations",
      [](int x) { return x > 0; }, true,
      "maximum iterations must be positive");
  RequireParamValue<double>(params, "early_exaggeration",
      [](double x) { return x >= 1.0; }, true,
      "early exaggeration must be >= 1.0");

  // Set random seed.
  if (randomSeed != 0)
    RandomSeed((size_t) randomSeed);
  else
    RandomSeed((size_t) std::time(NULL));

  // Create t-SNE object and run.
  TSNE<> tsne(perplexity, learningRate, maxIterations, earlyExaggeration);
  
  arma::mat output;
  
  Log::Info << "Performing t-SNE on dataset..." << endl;
  timers.Start("tsne");
  tsne.Apply(dataset, output, newDimension);
  timers.Stop("tsne");

  Log::Info << "t-SNE complete. Output dimensionality: " << output.n_rows
      << " x " << output.n_cols << "." << endl;

  // Save the results.
  if (params.Has("output"))
    params.Get<arma::mat>("output") = std::move(output);
}
