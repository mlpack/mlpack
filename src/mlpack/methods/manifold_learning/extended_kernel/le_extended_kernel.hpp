/**
 * @file le_extended_kernel.hpp
 * @author Shangtong Zhang
 *
 * This is a kernel used to calculate the embedding vector for a new data point
 * for LE
 */
#ifndef __MLPACK_METHODS_MANIFOLD_LEARNING_EXTENDED_KERNEL_LE_EXTENDED_KERNEL
#define __MLPACK_METHODS_MANIFOLD_LEARNING_EXTENDED_KERNEL_LE_EXTENDED_KERNEL

#include <mlpack/core.hpp>

namespace mlpack {
namespace manifold {
 
/**
 * This is a kernel used to calculate the embedding vector for a new data point
 * for LE.
 * @tparam SimilarityRule How to build similarity (affinity) matrix given a dataset
 * @tparam TransformRule How to transform similarity matrix
 * @tparam EigenmapRule How to generate embedding vectors from transformed
 *    similarity matrix
 */
template<
    typename SimilarityRule = KLESimilarity<>,
    typename TransformRule = LETransform<>,
    typename EigenmapRule = LEEigenmap<>,
    typename MatType = arma::mat,
    typename VecType = arma::colvec>
class LEExtendedKernel
{
 public:
  /**
   * Create a LEExtendedKernel object and set the parameters.
   * @param data Original data set
   * @param similarity Adopted similarity rule
   * @param transform Adopted transform rule
   * @param eigenmap Adopted eigenmap rule
   */
  LEExtendedKernel(const MatType& data,
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
    // precalculate the expectation for the new point
    pointE = 0;
    similarity.BuildSimilarityVec(searcher, point, neighbors, distances);
    kValue.zeros(data.n_cols);
    for (size_t i = 0; i < distances.size(); ++i)
    {
      if (transform.NNKernel())
        kValue(neighbors[i]) = 1;
      else
        kValue(neighbors[i]) = transform.Kernel().Evaluate(distances[i]);
      
      pointE += kValue(neighbors[i]);
    }
    pointE /= data.n_cols;
  }
  
  /**
   * Evaluate the value for the new data point and a given point
   * @param target the index of the given point
   */
  double Evaluate(size_t target)
  {
    return 1.0 / data.n_cols * K(target) /
        std::sqrt(pointE * Expectation(target));
  }
  
 private:
  
  /**
   * Calculate the value for the new point and a give point
   * @param target the index of the given point
   */
  double K(size_t target)
  {
    return kValue(target);
  }

  /**
   * calculate the expectation for a point
   * @param fixed the index of the given point
   */
  double Expectation(size_t fixed)
  {
    double E = 0;
    
    for (size_t i = 0; i < data.n_cols; ++i)
      E += transform.GramM()(i, fixed);
    
    return E / (data.n_cols - 1);
  }
  
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
  
  //! Precalculated kernel value between the new point and its neighbors
  arma::colvec kValue;
  
  //! Expectation for the new point
  double pointE;
};
    
}; // namespace manifold
}; // namespace mlpack

#endif
