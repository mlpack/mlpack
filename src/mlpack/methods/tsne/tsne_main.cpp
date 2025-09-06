/**
 * @file methods/tsne/tsne_main.cpp
 * @author Ranjodh Singh
 *
 * Main executable to run t-SNE.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "tsne_methods.hpp"
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME pca

#include <mlpack/core/util/mlpack_main.hpp>

#include "tsne.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_USER_NAME("t-distributed Stochastic Neighbor Embedding");

// Short description.
BINDING_SHORT_DESC(
    "t-distributed stochastic neighbor embedding (t-SNE) "
    "is a statistical method for visualizing high-dimensional data "
    "by giving each datapoint a location in a two or three-dimensional map.");

// Long description.
BINDING_LONG_DESC(
    "t-distributed stochastic neighbor embedding (t-SNE) "
    "is a statistical method for visualizing high-dimensional data "
    "by giving each datapoint a location in a two or three-dimensional map.");

// To Do: Example.
// BINDING_EXAMPLE("");

// See also...
BINDING_SEE_ALSO("t-distributed Stochastic Neighbor Embedding on Wikipedia",
                 "https://en.wikipedia.org/wiki/"
                 "T-distributed_stochastic_neighbor_embedding");
BINDING_SEE_ALSO("TSNE C++ class documentation", "@doc/user/methods/tsne.md");

// Parameters for program.
PARAM_MATRIX_IN("input", "Input dataset to perform t-SNE on.", "i");
PARAM_MATRIX_OUT("output", "Matrix to save modified dataset to.", "o");
PARAM_INT_IN("output_dimensions",
             "Dimensionality of the embedded space.",
             "d",
             2);
PARAM_DOUBLE_IN("perplexity",
                "Perplexity regulates the balance between local and global "
                "structure preservation, typically set between 5 and 50.",
                "p",
                30.0);
PARAM_DOUBLE_IN(
    "exaggeration",
    "Amplifies pairwise similarities during the initial optimization phase. "
    "This helps form tighter clusters and clearer separation between them. "
    "A higher value increases the spacing between clusters, but if the cost "
    "function grows during initial itterations, reduce this value or "
    "lower the learning rate.",
    "e",
    12.0);
PARAM_DOUBLE_IN("learning_rate", "Learning rate for the optimizer.", "l", 1.0);
PARAM_INT_IN("max_iterations", "Maximum number of iterations.", "n", 1000);
PARAM_STRING_IN("init",
                "Initialization method for the output embedding. "
                "Options are: 'random', 'pca' (default). "
                "random is not reccomended, as PCA can improve "
                " both speed and quality of the embedding.",
                "r",
                "pca");
PARAM_STRING_IN("method",
                "Gradient computation strategy. Options are: "
                "'exact', 'dual_tree', 'barnes_hut' (default)",
                "m",
                "barnes_hut");
PARAM_DOUBLE_IN(
    "theta",
    "Theta regulates the trade-off between "
    "speed and accuracy for 'barnes_hut' and 'dual_tree' approximations "s
    "the optimal value for theta is different for the two approximations.",
    "t",
    0.5);

//! Run TSNE on the specified dataset with the given policy.
template <typename TSNEStrategy>
void RunTSNE(util::Params& params, util::Timers& timers, arma::mat& dataset)
{
  TSNE<TSNEStrategy> tsne(params.Get<int>("output_dimensions"),
                          params.Get<double>("perplexity"),
                          params.Get<double>("exaggeration"),
                          params.Get<double>("learning_rate"),
                          params.Get<int>("max_iterations"),
                          params.Get<std::string>("init"),
                          params.Get<double>("theta"));


  Log::Info << "Running TSNE on dataset..." << endl;
  timers.Start("tsne");
  tsne.Embed(params.Get<arma::mat>("input"), dataset);
  timers.Stop("tsne");
  Log::Info << "TSNE Finished on dataset..." << endl;
}

// Binding Fuction
void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  arma::mat dataset;

  const std::string method = params.Get<std::string>("method");
  if (method == "exact")
  {
    RunTSNE<ExactTSNE>(params, timers, dataset);
  }
  else if (method == "dual_tree")
  {
    RunTSNE<DualTreeTSNE>(params, timers, dataset);
  }
  else if (method == "barnes_hut")
  {
    RunTSNE<BarnesHutTSNE>(params, timers, dataset);
  }
  else
  {
    /* To Do: Throw Error */
  }

  if (params.Has("output"))
    params.Get<arma::mat>("output") = std::move(dataset);
}
