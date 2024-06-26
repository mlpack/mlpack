/**
 * @file methods/dbscan/dbscan_main.cpp
 * @author Ryan Curtin
 *
 * Implementation of program to run DBSCAN.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME dbscan

#include <mlpack/core/util/mlpack_main.hpp>
#include "dbscan.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_USER_NAME("DBSCAN clustering");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of DBSCAN clustering.  Given a dataset, this can "
    "compute and return a clustering of that dataset.");

// Long description.
BINDING_LONG_DESC(
    "This program implements the DBSCAN algorithm for clustering using "
    "accelerated tree-based range search.  The type of tree that is used "
    "may be parameterized, or brute-force range search may also be used."
    "\n\n"
    "The input dataset to be clustered may be specified with the " +
    PRINT_PARAM_STRING("input") + " parameter; the radius of each range "
    "search may be specified with the " + PRINT_PARAM_STRING("epsilon") +
    " parameters, and the minimum number of points in a cluster may be "
    "specified with the " + PRINT_PARAM_STRING("min_size") + " parameter."
    "\n\n"
    "The " + PRINT_PARAM_STRING("assignments") + " and " +
    PRINT_PARAM_STRING("centroids") + " output parameters may be "
    "used to save the output of the clustering. " +
    PRINT_PARAM_STRING("assignments") + " contains the cluster assignments of "
    "each point, and " + PRINT_PARAM_STRING("centroids") + " contains the "
    "centroids of each cluster."
    "\n\n"
    "The range search may be controlled with the " +
    PRINT_PARAM_STRING("tree_type") + ", " +
    PRINT_PARAM_STRING("single_mode") + ", and " +
    PRINT_PARAM_STRING("naive") + " parameters.  " +
    PRINT_PARAM_STRING("tree_type") + " can control the type of tree used for "
    "range search; this can take a variety of values: 'kd', 'r', 'r-star', 'x',"
    " 'hilbert-r', 'r-plus', 'r-plus-plus', 'cover', 'ball'. The " +
    PRINT_PARAM_STRING("single_mode") + " parameter will force single-tree "
    "search (as opposed to the default dual-tree search), and '" +
    PRINT_PARAM_STRING("naive") + " will force brute-force range search.");

// Example.
BINDING_EXAMPLE(
    "An example usage to run DBSCAN on the dataset in " +
    PRINT_DATASET("input") + " with a radius of 0.5 and a minimum cluster size"
    " of 5 is given below:"
    "\n\n" +
    PRINT_CALL("dbscan", "input", "input", "epsilon", 0.5, "min_size", 5));

// See also...
BINDING_SEE_ALSO("DBSCAN on Wikipedia", "https://en.wikipedia.org/wiki/DBSCAN");
BINDING_SEE_ALSO("A density-based algorithm for discovering clusters in large "
    "spatial databases with noise (pdf)",
    "https://cdn.aaai.org/KDD/1996/KDD96-037.pdf");
BINDING_SEE_ALSO("DBSCAN class documentation",
    "@src/mlpack/methods/dbscan/dbscan.hpp");

PARAM_MATRIX_IN_REQ("input", "Input dataset to cluster.", "i");
PARAM_UROW_OUT("assignments", "Output matrix for assignments of each "
    "point.", "a");
PARAM_MATRIX_OUT("centroids", "Matrix to save output centroids to.", "C");

PARAM_DOUBLE_IN("epsilon", "Radius of each range search.", "e", 1.0);
PARAM_INT_IN("min_size", "Minimum number of points for a cluster.", "m", 5);

PARAM_STRING_IN("tree_type", "If using single-tree or dual-tree search, the "
    "type of tree to use ('kd', 'r', 'r-star', 'x', 'hilbert-r', 'r-plus', "
    "'r-plus-plus', 'cover', 'ball').", "t", "kd");
PARAM_STRING_IN("selection_type", "If using point selection policy, the "
    "type of selection to use ('ordered', 'random').", "s", "ordered");
PARAM_FLAG("single_mode", "If set, single-tree range search (not dual-tree) "
    "will be used.", "S");
PARAM_FLAG("naive", "If set, brute-force range search (not tree-based) "
    "will be used.", "N");

// Actually run the clustering, and process the output.
template<typename RangeSearchType, typename PointSelectionPolicy>
void RunDBSCAN(util::Params& params,
               RangeSearchType rs,
               PointSelectionPolicy pointSelector = PointSelectionPolicy())
{
  if (params.Has("single_mode"))
    rs.SingleMode() = true;

  // Load dataset.
  arma::mat dataset = std::move(params.Get<arma::mat>("input"));
  const double epsilon = params.Get<double>("epsilon");
  const size_t minSize = (size_t) params.Get<int>("min_size");
  arma::Row<size_t> assignments;

  DBSCAN<RangeSearchType, PointSelectionPolicy> d(epsilon, minSize,
      !params.Has("single_mode"), rs, pointSelector);

  // If possible, avoid the overhead of calculating centroids.
  if (params.Has("centroids"))
  {
    arma::mat centroids;

    d.Cluster(dataset, assignments, centroids);

    params.Get<arma::mat>("centroids") = std::move(centroids);
  }
  else
  {
    d.Cluster(dataset, assignments);
  }

  if (params.Has("assignments"))
    params.Get<arma::Row<size_t>>("assignments") = std::move(assignments);
}

// Choose the point selection policy.
template<typename RangeSearchType>
void ChoosePointSelectionPolicy(util::Params& params,
                                RangeSearchType rs = RangeSearchType())
{
  const string selectionType = params.Get<string>("selection_type");

  if (selectionType == "ordered")
    RunDBSCAN<RangeSearchType, OrderedPointSelection>(params, rs);
  else if (selectionType == "random")
    RunDBSCAN<RangeSearchType, RandomPointSelection>(params, rs);
}

void BINDING_FUNCTION(util::Params& params, util::Timers& /* timers */)
{
  RequireAtLeastOnePassed(params, { "assignments", "centroids" }, false,
      "no output will be saved");

  ReportIgnoredParam(params, {{ "naive", true }}, "single_mode");

  RequireParamInSet<string>(params, "tree_type", { "kd", "cover", "r", "r-star",
      "x", "hilbert-r", "r-plus", "r-plus-plus", "ball" }, true,
      "unknown tree type");

  // Value of epsilon should be positive.
  RequireParamValue<double>(params, "epsilon", [](double x) { return x > 0; },
      true, "invalid value of epsilon specified");

  // Value of min_size should be positive.
  RequireParamValue<int>(params, "min_size", [](int y) { return y > 0; },
      true, "invalid value of min_size specified");

  // Fire off naive search if needed.
  if (params.Has("naive"))
  {
    RangeSearch<> rs(true);
    ChoosePointSelectionPolicy(params, rs);
  }
  else
  {
    const string treeType = params.Get<string>("tree_type");
    if (treeType == "kd")
    {
      ChoosePointSelectionPolicy<RangeSearch<>>(params);
    }
    else if (treeType == "cover")
    {
      ChoosePointSelectionPolicy<RangeSearch<EuclideanDistance, arma::mat,
          StandardCoverTree>>(params);
    }
    else if (treeType == "r")
    {
      ChoosePointSelectionPolicy<RangeSearch<EuclideanDistance, arma::mat,
          RTree>>(params);
    }
    else if (treeType == "r-star")
    {
      ChoosePointSelectionPolicy<RangeSearch<EuclideanDistance, arma::mat,
          RStarTree>>(params);
    }
    else if (treeType == "x")
    {
      ChoosePointSelectionPolicy<RangeSearch<EuclideanDistance, arma::mat,
          XTree>>(params);
    }
    else if (treeType == "hilbert-r")
    {
      ChoosePointSelectionPolicy<RangeSearch<EuclideanDistance, arma::mat,
          HilbertRTree>>(params);
    }
    else if (treeType == "r-plus")
    {
      ChoosePointSelectionPolicy<RangeSearch<EuclideanDistance, arma::mat,
          RPlusTree>>(params);
    }
    else if (treeType == "r-plus-plus")
    {
      ChoosePointSelectionPolicy<RangeSearch<EuclideanDistance, arma::mat,
          RPlusPlusTree>>(params);
    }
    else if (treeType == "ball")
    {
      ChoosePointSelectionPolicy<RangeSearch<EuclideanDistance, arma::mat,
          BallTree>>(params);
    }
  }
}
