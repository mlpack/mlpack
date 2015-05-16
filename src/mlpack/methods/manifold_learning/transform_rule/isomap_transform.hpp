/**
 * @file isomap_transform.hpp
 * @author Shangtong Zhang
 *
 * Implementation for transforming similarity matrix for Isomap
 */

#ifndef __MLPACK_METHODS_MANIFOLD_LEARNING_TRANSFORM_RULE_ISOMAP_TRANSFORM
#define __MLPACK_METHODS_MANIFOLD_LEARNING_TRANSFORM_RULE_ISOMAP_TRANSFORM

#include <mlpack/core.hpp>

#include <mlpack/methods/manifold_learning/shortest_path/floyd_ warshall.hpp>
#include <mlpack/methods/manifold_learning/shortest_path/dijkstra.hpp>

#include <mlpack/methods/manifold_learning/transform_rule/mds_transform.hpp>

namespace mlpack {
namespace manifold {
  
/**
 * This class is an implementation for transforming similarity matrix for Isomap
 * @tparam ShortestPath How to calculate shortest path within a graph
 */
template<
    typename ShortestPath = Dijkstra<>,
    typename SimilarityMatType = arma::sp_mat,
    typename MatType = arma::mat>
class IsomapTransform
{
 public:
  
  /**
   * Transform similarity matrix
   * @param M Similarity matrix to be transformed
   * @return reference for transformed similarity matrix
   */
  MatType& Transform(const SimilarityMatType& M)
  {
    ShortestPath::Solve(M, distMat);
    MDSTransform<> mdsTransform;
    gramM = mdsTransform.Transform(arma::square(distMat));
    return gramM;
  }
  
  //! Get distances matrix
  MatType& DistMat() const { return distMat; }
  //! Modify distances matrix
  MatType& DistMat() { return distMat; }
  
  //! Get transformed similarity matrix
  MatType& GramM() const { return gramM; }
  //! Modify transformed similarity matrix
  MatType& GramM() { return gramM; }
  
 private:
  //! Locally-stored distances matrix
  MatType distMat;
  
  //! Locally-stored transformed similarity matrix
  MatType gramM;
};
  
}; // namespace manifold
}; // namespace mlpack

#endif
