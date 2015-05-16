/**
 * @file lle_extended_kernel.hpp
 * @author Shangtong Zhang
 *
 * This is a kernel used to calculate the embedding vector for a new data point
 * for LLE
 */
#ifndef __MLPACK_METHODS_MANIFOLD_LEARNING_EXTENDED_KERNEL_LLE_EXTENDED_KERNEL
#define __MLPACK_METHODS_MANIFOLD_LEARNING_EXTENDED_KERNEL_LLE_EXTENDED_KERNEL

#include <mlpack/core.hpp>

namespace mlpack {
namespace manifold {
  
/**
 * This is a kernel used to calculate the embedding vector for a new data point
 * for LLE.
 * @tparam SimilarityRule How to build similarity (affinity) matrix given a dataset
 * @tparam TransformRule How to transform similarity matrix
 * @tparam EigenmapRule How to generate embedding vectors from transformed
 *    similarity matrix
 */
template<
    typename SimilarityRule = KLLESimilarity<>,
    typename TransformRule = LLETransform<>,
    typename EigenmapRule = LLEEigenmap<>,
    typename MatType = arma::mat,
    typename VecType = arma::colvec>
class LLEExtendedKernel
{
 public:
  /**
   * Create a LLEExtendedKernel object and set the parameters.
   * @param data Original data set
   * @param similarity Adopted similarity rule
   * @param transform Adopted transform rule
   * @param eigenmap Adopted eigenmap rule
   */
  LLEExtendedKernel(const MatType& data,
                    SimilarityRule& similarity,
                    TransformRule& transform,
                    EigenmapRule& eigenmap) :
      data(data), similarity(similarity), transform(transform),
      eigenmap(eigenmap), searcher(data)
  {
    /* nothing to do */
  }
  
  /**
   * Fit the new data point
   * @param point new data point
   */
  void FitPoint(const VecType& point)
  {
    similarity.BuildSimilarityVec(searcher, point, neighbors, distances);
    
    // calculate the wieght for reconstruction
    arma::mat neighborData;
    for (size_t i = 0; i < neighbors.size(); ++i)
    {
      neighborData.insert_cols(neighborData.n_cols, data.unsafe_col(neighbors[i]));
    }
    
    for (size_t i = 0; i < neighborData.n_cols; ++i)
      neighborData.col(i) -= point;
    
    MatType cov = neighborData.t() * neighborData;
    
    if (neighbors.size() > data.n_rows)
      cov += arma::eye(neighbors.size(), neighbors.size()) *
          similarity.Tol() * arma::trace(cov);
    
    arma::colvec onesVec(neighbors.size());
    onesVec.ones();
    arma::colvec w = arma::solve(cov, onesVec);
    w /= arma::accu(w);
    weight.zeros(data.n_cols, 1);
    for (size_t i = 0; i < neighbors.size(); ++i)
      weight(neighbors[i]) = w(i);
  }
  
  /**
   * Evaluate the value for the new data point and a given point
   * @param target the index of the given point
   */
  double Evaluate(size_t target)
  {
    return (transform.Mu() - 1) * weight(target);
  }
  
 private:
  
  //! Locally-stored original data set
  const MatType& data;
  
  //! Locally-stored similarity rule
  SimilarityRule& similarity;
  
  //! Locally-stored transform rule
  TransformRule& transform;
  
  //! Locally-stored eigenmap rule
  EigenmapRule& eigenmap;
  
  //! Locally-stored searcher over the original data set
  typename std::remove_reference<decltype(similarity)>::
      type::SearcherType searcher;
  
  //! Locally-stored neighbors of the new point
  std::vector<size_t> neighbors;
  
  //! Locally-stored distances between the new point and its neighbors
  std::vector<double> distances;
  
  //! Precalculated weight for reconstruction
  arma::colvec weight;
};
    
}; // namespace manifold
}; // namespace mlpack

#endif
