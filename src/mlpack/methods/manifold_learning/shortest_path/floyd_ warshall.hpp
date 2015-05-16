/**
 * @file floyd_warshall.hpp
 * @author Shangtong Zhang
 *
 * Implementation for floyd warshall algorithm
 */
#ifndef __MLPACK_METHODS_MANIFOLD_LEARNING_SHORTEST_PATH_FLOYD_WARSHALL
#define __MLPACK_METHODS_MANIFOLD_LEARNING_SHORTEST_PATH_FLOYD_WARSHALL

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace manifold {
  
/**
 * Implementation for floyd warshall algorithm
 */
template<
    typename SimilarityMatType = arma::sp_mat,
    typename MatType = arma::mat>
class FloydWarshall
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
    for (size_t i = 0; i < nVex; ++i)
    {
      for (size_t j = 0; j < nVex; ++j)
      {
        if (i != j && neighborMat(i, j) == 0)
          distMat(i, j) = arma::datum::inf;
        else
          distMat(i, j) = neighborMat(i, j);
      }
    }
    
    for (size_t k = 0; k < nVex; ++k)
    {
      for (size_t i = 0; i < nVex; ++i)
      {
        for (size_t j = 0; j < nVex; ++j)
        {
          double newD = distMat(i, k) + distMat(k, j);
          if (newD < distMat(i ,j))
            distMat(i, j) = newD;
        }
      }
    }
  }
};
    
}; // namespace manifold
}; // namespace mlpack

#endif