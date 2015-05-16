/**
 * @file isomap_extended_kernel.hpp
 * @author Shangtong Zhang
 *
 * This is a kernel used to calculate the embedding vector for a new data point
 * for Isomap
 */
#ifndef __MLPACK_METHODS_MANIFOLD_LEARNING_EXTENDED_KERNEL_ISOMAP_EXTENDED_KERNEL
#define __MLPACK_METHODS_MANIFOLD_LEARNING_EXTENDED_KERNEL_ISOMAP_EXTENDED_KERNEL

#include <mlpack/core.hpp>

namespace mlpack {
namespace manifold {
  
/**
 * This is a kernel used to calculate the embedding vector for a new data point
 * for Isomap.
 * @tparam SimilarityRule How to build similarity (affinity) matrix given a dataset
 * @tparam TransformRule How to transform similarity matrix
 * @tparam EigenmapRule How to generate embedding vectors from transformed
 *    similarity matrix
 */
template<
    typename SimilarityRule = KIsomapSimilarity<>,
    typename TransformRule = IsomapTransform<>,
    typename EigenmapRule = IsomapEigenmap<>,
    typename MatType = arma::mat,
    typename VecType = arma::colvec>
class IsomapExtendedKernel
{
 public:
  /**
   * Create a IsomapExtendedKernel object and set the parameters.
   * @param data Original data set
   * @param similarity Adopted similarity rule
   * @param transform Adopted transform rule
   * @param eigenmap Adopted eigenmap rule
   */
  IsomapExtendedKernel(const MatType& data,
                       SimilarityRule& similarity,
                       TransformRule& transform,
                       EigenmapRule& eigenmap) :
      data(data), similarity(similarity), transform(transform),
      eigenmap(eigenmap), searcher(data), distMat(transform.DistMat())
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
    
    // precalculate the expectation over the whole original data set
    entireE = 0;
    for (size_t i = 0; i < data.n_cols; ++i)
      for (size_t j = i; j < data.n_cols; ++j)
        entireE += std::pow(GramD(i, j), 2);
    entireE /= (data.n_cols * (data.n_cols + 1) / 2);
    
    // precalculate the expectation for the new point
    pointE = 0;
    for (size_t i = 0; i < data.n_cols; ++i)
      pointE += std::pow(GramD(i), 2);
    pointE /= data.n_cols;
  }
  
  /** 
   * Evaluate the value for the new data point and a given point
   * @param target the index of the given point
   */
  double Evaluate(size_t target)
  {
    return -0.5 * (std::pow(GramD(target), 2) -
        pointE - Expectation(target) + entireE);
  }
  
 private:
  
  // calculate the extended distance between point i and point j in original data
  double GramD(size_t i, size_t j)
  {
    return distMat(i, j);
  }
  
  /**
   * calculate the extended distance between the new point and a given point
   * @param target the index of the given point
   */
  double GramD(size_t target)
  {
    double shortest = arma::datum::inf;
    for (size_t i = 0; i < neighbors.size(); ++i)
      shortest = std::min(distances[i] + distMat(neighbors[i], target), shortest);
    return shortest;
  }
  
  // calculate the expectation for a point i
  double Expectation(size_t i)
  {
    double E = 0;
    for (size_t j = 0; j < data.n_cols; ++j)
      E += std::pow(GramD(i, j), 2);
    return E / data.n_cols;
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
  
  //! Locally-stored distance matrix
  MatType& distMat;
  
  //! Locally-stored neighbors of the new point
  std::vector<size_t> neighbors;
  
  //! Locally-stored distances between the new point and its neighbors
  std::vector<double> distances;
  
  //! Expectation over the whole data set
  double entireE;
  
  //! Expectation for the new point
  double pointE;
};
  
}; // namespace manifold
}; // namespace mlpack

#endif
