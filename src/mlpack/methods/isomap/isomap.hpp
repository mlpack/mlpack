/**
 * @file isomap.hpp
 * @author Rishabh Ranjan
 * 
 * This file defines the Isomap class which implements ISOMAP algorithm on a
 * given dataset. The templates are provided so that this class can include
 * variations of constructing neighbourhood graphs and finding all pair
 * shortest paths in the graph. Presently, neighbourhood graph is constructed
 * using only K_Nearest_Neighbours and all pair shortest path is calcuated
 * using only Dijkstra's Algorithm
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
#include "k_nearest.hpp"
#include "dijkstra.hpp"
#include "mds.hpp"

namespace mlpack {
namespace isomap {

/**
 * This class implements the Isomap algorithm which is part of Manifold
 * Learning algorithms, and is used for dimensionality reduction. This
 * algorithm converts the dataset into a lower dimensional space which
 * it is assumed to belong from the higher dimensional space it is
 * currently embedded into.
 * This implementation is based on the paper:
 * Lawrence Cayton, 2005, 17, Algorithms for Manifold Learning
 */

template <typename NeighbourhoodFunction = K_Nearest,
          typename ShortestPathAlgo = Dijkstra>
class Isomap
{
 public:
  /**
   * Create the Isomap obejct and set the n_neighbours parameter to
   * specify the number of neighbours to consider while constructing
   * neighbourhood graph.
   * 
   * @param n_neighbours -number of neighbours to consider
   *                      for neighbourhood graph
   */

  Isomap(const size_t n_neighbours,
         const NeighbourhoodFunction& neighbourhood = NeighbourhoodFunction(),
         const ShortestPathAlgo& shortestPath = ShortestPathAlgo()) :
         n_neighbours(n_neighbours),
         neighbourhood(neighbourhood),
         shortestPath(shortestPath)
  { }

  /**
   * This is the main driver function to perform Isomap. Just the input
   * matrix is required and it is safe to pass it as reference.
   * 
   * @param input -the input dataset to perform Isomap on.
   */
  void Apply(arma::mat& input)
  {
    arma::mat disMat(input.n_cols, input.n_cols);

    // calculating distance matrix from the given input matrix
    CalcDistanceMatrix(input, disMat);

    // constructing neighbourhood graph (K_nearest is used)
    neighbourhood.MakeNeighbourhoodGraph(disMat, n_neighbours);

    // finding all pair shortest path in the neighbourhood graph created
    shortestPath.FindShortestPath(disMat);

    // making the shortest distance matrix symmetric (required for cMDS)
    disMat = arma::min(disMat, disMat.t());

    // mds object to perform classical multidimensional scaling
    MDS md;

    // performing classical multidimensional scaling
    md.Apply(disMat);
    input = disMat.t();
  }

 private:
  /**
   * Function to calculate distance matric from the given input dataset.
   * disMat is requred to store the distance matrix calculated.
   * 
   * @param input -input dataset
   * @param disMat -stores the distance matrix
   */
  void CalcDistanceMatrix(const arma:: mat& input,
                         arma::mat& disMat)
  {
    for (size_t i = 0; i < input.n_cols; i++)
    {
      disMat(i, i) = 0;
      arma::rowvec tempi = input.col(i);
      for (size_t j = i+1; j < input.n_cols; j++)
      {
        arma::rowvec tempj = input.col(j);
        disMat(i, j) = metric::EuclideanDistance().Evaluate(tempi, tempj);
        disMat(j, i) = disMat(i, j);
      }
    }
  }

  // the number of neighbours for K-nearest
  size_t n_neighbours;

  // the function to use to construct neighbourhood graph
  NeighbourhoodFunction neighbourhood;

  // the algorithm to use to calculate all pair shortest distance
  ShortestPathAlgo shortestPath;
};

} // namespace isomap
} // namespace mlpack

#endif
