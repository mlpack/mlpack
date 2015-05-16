/**
 * @file dijkstra.hpp
 * @author Shangtong Zhang
 *
 * Implementation for dijkstra algorithm
 */
#ifndef __MLPACK_METHODS_MANIFOLD_LEARNING_SHORTEST_PATH_DIJKSTRA
#define __MLPACK_METHODS_MANIFOLD_LEARNING_SHORTEST_PATH_DIJKSTRA

#include <mlpack/core.hpp>

#include "queue"

namespace mlpack {
namespace manifold {

/**
 * store vertext and its priority
 */
class Vex
{
 public:
  size_t vex;
  double priority;
  Vex(size_t vex, double priority)
  : vex(vex), priority(priority)
  {
    /* nothing to do */
  }
};

/**
 * comparer for two Vex class
 */
class VexComp
{
 public:
  bool operator()(const Vex& v1, const Vex& v2)
  {
    return v1.priority > v2.priority;
  }
};

/**
 * Implementation for dijkstra algorithm
 */
template<
    typename SimilarityMatType = arma::sp_mat,
    typename MatType = arma::mat>
class Dijkstra
{
  public:
  
  /**
   * Solve multiple source shortest path problem
   * @param neighborMat The neighbor matrix for the graph, it can be asymmetrical.
   *    neighborMat(i, j) = 0 (i != j) means vertex i and vertex j aren't connected.
   * @param distMat distMat(i, j) is the shortest distance from i to j.
   */
  static void Solve(const SimilarityMatType& neighborMat, MatType& distMat)
  {
    size_t nVex = neighborMat.n_cols;
    distMat.zeros(nVex, nVex);
    
    // perform dijkstra algorithm for each vertex
    for (size_t vex = 0; vex < nVex; ++vex)
    {
      // priority queue to store visited vertex
      std::priority_queue<Vex, std::vector<Vex>, VexComp> q;
      
      // initialize distance vector for current vertex
      for (size_t i = 0; i < nVex; ++i)
      {
        if (i != vex)
        {
          distMat(vex, i) = arma::datum::inf;
          q.push(Vex(i, arma::datum::inf));
        }
        else
        {
          distMat(vex, i) = 0;
          q.push(Vex(i, 0));
        }
      }
      
      // entries to trace if a vertex has been visited
      arma::Col<size_t> visited(nVex);
      visited.zeros();
      
      // perform nVex epoch
      for (size_t epoch = 0; epoch < nVex; ++epoch)
      {
        // get the vertex u with shortest path to current vertex
        size_t u = q.top().vex;
        while (visited(u) == 1)
        {
          q.pop();
          u = q.top().vex;
        }
        q.pop();
        
        // update distance of u's neighbors
        for (size_t v = 0; v < nVex; ++v)
        {
          if (neighborMat(u, v) != 0 && visited(v) == 0)
          {
            double newD = distMat(vex, u) + neighborMat(u, v);
            if (distMat(vex, v) > newD)
            {
              distMat(vex, v) = newD;
              q.push(Vex(v, newD));
            }
          }
        }
        
        visited(u) = 1;
      } 
    }
  }
};
    
}; // namespace manifold
}; // namespace mlpack

#endif