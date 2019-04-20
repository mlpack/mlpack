/**
 * @file isomap.hpp
 * @author Rishabh Ranjan
 * 
 * This file defines the Isomap class which implements ISOMAP algorithm on a
 * given dataset. The templates are provided so that this class can include
 * variations of constructing neighbourhood graphs and finding all pair
 * shortest paths in the graph. Presently, neighbourhood graph is constructed
 * using only K_Nearest_Neighbours and all pair shortest path is calcuated
 * using only Dijkstra's Algorithm.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ISOMAP_ISOMAP_HPP
#define MLPACK_METHODS_ISOMAP_ISOMAP_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/cmds/cmds.hpp>

#include "dijkstra.hpp"
#include "k_nearest.hpp"

namespace mlpack {
namespace isomap {

/**
 * This class implements the Isomap algorithm which is part of Manifold
 * Learning algorithms, and is used for dimensionality reduction. This
 * algorithm converts the dataset into a lower dimensional space which
 * it is assumed to belong from the higher dimensional space it is
 * currently embedded into.
 * This implementation is based on the paper:
 * @article{
 *  author = {Joshua B. Tenenbaum, Vin de Silva, John C. Langford},
 *  year = {2000},
 *  pages = {2319--2322},
 *  title = {A Global Geometric Frameworkfor Nonlinear Dimensionality 
 *      Reduction},
 *  journal = {Science Vol 290}
 * }
 */
template <typename NeighborhoodFunction = KNearest,
          typename ShortestPathAlgo = Dijkstra>
class Isomap
{
 public:
  /**
   * Create the Isomap object and set the k parameter to
   * specify the number of neighbours to consider while constructing
   * neighbourhood graph.
   * 
   * @param k Number of neighbours to consider for neighbourhood graph
   * @param d Number of dimensions to include in output
   */

  Isomap(const size_t k,
         const size_t d,
         const NeighborhoodFunction& neighborhood = NeighborhoodFunction(),
         const ShortestPathAlgo& shortestPath = ShortestPathAlgo()) :
         k(k),
         d(d),
         neighborhood(neighborhood),
         shortestPath(shortestPath)
  { }

  /**
   * This is the main driver function to perform Isomap. Just the input
   * matrix is required and it is safe to pass it as reference.
   * 
   * @param input -the input dataset to perform Isomap on.
   */
  void Apply(arma::mat& dataset)
  {
    // Constructing K-Nearest Neighborhood Graph.
    neighborhood.MakeNeighborhoodGraph(k, dataset);

    // Finding all pair shortest path in the neighborhood graph created.
    shortestPath.FindShortestPath(dataset);

    // Making sure the shortest distance matrix symmetric (required for cMDS).
    dataset = arma::symmatu(dataset);

    // Cmds object to perform classical multidimensional scaling.
    cmds::Cmds m;

    // Performing classical multidimensional scaling which stores the result
    // the same matrix given as argument.
    m.Apply(false, d, dataset);
  }

 private:
  // The number of neighbours for K-nearest.
  size_t k;

  // The number of dimensions in the output.
  size_t d;

  // The function to use to construct neighbourhood graph.
  NeighborhoodFunction neighborhood;

  // The algorithm to use to calculate all pair shortest distance
  ShortestPathAlgo shortestPath;
};

} // namespace isomap
} // namespace mlpack

#endif
