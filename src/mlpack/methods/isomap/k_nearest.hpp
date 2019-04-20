/**
 * @file k_nearest.hpp
 * @author Rishabh Ranjan
 * 
 * This file defines the class K_Nearest which constructs the neighbourhood
 * graph for Isomap, according to the number of neighbours provided, using the
 * neighbor_search method.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ISOMAP_K_NEAREST
#define MLPACK_METHODS_ISOMAP_K_NEAREST

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <queue>

namespace mlpack {
namespace isomap {

/**
 * This class uses the neighbor_search method to construct the k-nearest-
 * neighbor graph.
*/
class KNearest
{
 public:
  /**
   * Function to make the neighbourhood graph for the given distance matrix
   * and number of neighbours using neighbor_search method.
   * 
   * @param disMat Distance matrix to construct the graph.
   * @param k Number of neighbours to consider.
   */
  void MakeNeighborhoodGraph(const int k,
                             arma::mat& disMat)
  {
    // Using KNN method to find the k-nearest neighbors.
    neighbor::KNN neighborSearch(disMat);
    arma::Mat<size_t> neighbors;
    arma::mat distance;
    neighborSearch.Search(k, neighbors, distance);

    // Constructing adjacency matrix from the matrices (neighbors and distance).
    disMat.set_size(disMat.n_cols, disMat.n_cols);
    disMat.fill(LLONG_MAX);
    for (unsigned int j = 0; j < neighbors.n_cols; j++)
    {
      disMat(j, j) = 0;
      for (int i = 0; i < k; i++)
        disMat(j, neighbors(i, j)) = distance(i, j);
    }

    // Check if neighbourhood graph is connected.
    if (!IsConnected(disMat))
      Log::Fatal << "Constructed neighbourhood graph is not connected. "
                  << "Increase the number of neighbours (Default is 3).\n";
  }

 private:
  /**
   * Function to check if the neighbourhood graph is connected (weakly
   * connected at least) or not.
   * Isomap does not work if its not connected, thus program will be
   * terminated. Performs BFS for the check.
   * 
   * @param disMat -neighbourhood graph stored in disMat
   */
  bool IsConnected(arma::mat &disMat)
  {
    // Making disMat undirected to check if graph is at least
    // weakly connected.
    arma::mat tempMat = arma::min(disMat, disMat.t());

    arma::vec visited(tempMat.n_rows);
    visited.zeros();

    // queue required for BFS
    std::queue <size_t> q;
    q.push(0);
    visited(0) = 0;

    // BFS
    while (!q.empty())
    {
      size_t front = q.front();
      q.pop();
      for (size_t i = 0; i < tempMat.n_cols; i++)
      {
        if (!visited[i] && tempMat(front, i) < LLONG_MAX)
        {
          q.push(i);
          visited(i) = 1;
        }
      }
    }

    // checking if a node is not visited
    for (size_t i = 0; i < tempMat.n_rows; i++)
    {
      if (!visited(i))
        return false; // means disconnected graph
    }

    return true;
  }
};

} // namespace isomap
} // namespace mlpack

#endif
