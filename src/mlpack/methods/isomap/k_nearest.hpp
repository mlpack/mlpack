/**
 * @file k_nearest.hpp
 * @author Rishabh Ranjan
 * 
 * This file defines the class K_Nearest which constructs the neighbourhood
 * graph for Isomap, according to the number of neighbours provided.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ISOMAP_K_NEAREST
#define MLPACK_METHODS_ISOMAP_K_NEAREST

#include <mlpack/prereqs.hpp>
#include <queue>

namespace mlpack {
namespace isomap {
/**
 * This class implements the K-Nearest Neighbours function to construct
 * a neighbourhood graph for the given distance matrix.
*/

class K_Nearest
{
 public:
  /**
   * Function to make the neighbourhood graph for the given distance matrix
   * and number of neighbours.
   * 
   * @param disMat -distance matrix to construct the graph
   * @param n_neighbours -number of neighbours to consider
   */
  void MakeNeighbourhoodGraph(arma::mat &disMat,
                              const int n_neighbours)
  {
    // go through every row
    for (size_t i = 0; i < disMat.n_rows; i++)
    {
      arma::uvec temp;

      // find indices if row is sorted in descending order
      temp = sort_index(disMat.row(i), "descend");

      // set distance to inf for every neighbour farther that n_neighbours
      for (size_t j = 0; j < temp.size()-n_neighbours-1; j++)
        disMat(i, temp(j)) = LLONG_MAX;
    }

    // check if neighbourhood graph is connected
    if (!IsConnected(disMat))
      Log::Fatal << "Constructed neighbourhood graph is not connected."
                  << "Increase the number of neighbours (Default is 3).\n";
  }

 private:
  /**
   * Function to check if the neighbourhood graph is connected or not.
   * Isomap does not work if its not connected, thus program will be
   * terminated. Performs BFS for the check.
   * 
   * @param disMat -neighbourhood graph stored in disMat
   */
  bool IsConnected(arma::mat &disMat)
  {
    bool flag = 1;
    bool visited[disMat.n_rows] = {0};

    // queue required for BFS
    std::queue <size_t> q;
    q.push(0);
    visited[0] = 0;

    // BFS
    while (!q.empty())
    {
      size_t front = q.front();
      q.pop();
      for (size_t i = 0; i < disMat.n_cols; i++)
      {
        if (!visited[i] && disMat(front, i) < LLONG_MAX)
        {
          q.push(i);
          visited[i] = 1;
        }
      }
    }

    // checking if a node is not visited
    for (size_t i = 0; i < disMat.n_rows; i++)
    {
      if (!visited[i])
      {
        flag = 0; // means disconnected graph
        break;
      }
    }

    return flag;
  }
};

} // namespcae isomap
} // namespace mlpack

#endif
