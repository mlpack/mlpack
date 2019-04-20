/**
 * @file dijkstra.hpp
 * @author Rishabh Ranjan
 * 
 * This file defines Dijkstra class which implements Dijkstra's Algorithm
 * to find all pair shortest path for the neighbourhood graph provided.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ISOMAP_DIJKSTRA
#define MLPACK_METHODS_ISOMAP_DIJKSTRA

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace isomap {

/**
 * This class implements the Dijkstra Algorithm for all pair shortest path.
 */
class Dijkstra
{
 public:
  /**
   * Function to supply new source nodes to Dijkstra's Algorithm, so that all
   * pair shortest path is calculated.
   * 
   * @param disMat Neighbourhood graph from input dataset
  */
  void FindShortestPath(arma::mat& disMat)
  {
    for (size_t i = 0; i < disMat.n_rows; i++)
    {
      // This stores the calculated shortest distances for a given source.
      arma::rowvec temp(disMat.n_cols);

      // Performs Dijkstra's Algorithm using the given source.
      Apply(disMat, i, temp);
      disMat.row(i) = temp;
    }
  }

 private:
  /**
   * Function to implement Dijkstra's Algorithm for a given source.
   * 
   * @param disMat Neighbourhood grapgh from input dataset
   * @param source Source for Dijkstra's Algorithm
   * @param dist Stores the calculated shortest distances
   */
  void Apply(arma::mat& disMat,
              size_t source,
              arma::rowvec& dist)
  {
    for (size_t i = 0; i < disMat.n_cols; i++)
      dist(i) = LLONG_MAX;
    dist(source) = 0;
    arma::vec visited(disMat.n_cols);
    visited.zeros();

    // Minimum priority queue used for Dijkstra's Algorithm.
    std::priority_queue < std::pair< size_t, double>,
                          std::vector< std::pair< size_t, double> >,
                          std::greater < std::pair< size_t, double> > > pq;
    pq.push({source, 0.0});
    while (!pq.empty())
    {
      std::pair <size_t, double> top = pq.top();
      pq.pop();
      if (visited(top.first))
        continue;
      visited(top.first) = 1;
      for (size_t i = 0; i < disMat.n_cols; i++)
      {
        if (disMat(top.first, i) < LLONG_MAX)
        {
          if ((disMat(top.first, i)+top.second) < dist(i))
          {
            dist(i) = disMat(top.first, i) + top.second;
            pq.push({i, dist(i)});
          }
        }
      }
    }
  }
};

} // namespace isomap
} // namespace mlpack

#endif
