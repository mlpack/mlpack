/**
 * @file mds_extended_kernel.hpp
 * @author Shangtong Zhang
 *
 * This is a kernel used to calculate the embedding vector for a new data point
 * for MDS
 */
#ifndef __MLPACK_METHODS_MANIFOLD_LEARNING_EXTENDED_KERNEL_MDS_EXTENDED_KERNEL
#define __MLPACK_METHODS_MANIFOLD_LEARNING_EXTENDED_KERNEL_MDS_EXTENDED_KERNEL

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace manifold {
  
/**
 * This is a kernel used to calculate the embedding vector for a new data point
 * for MDS.
 * @tparam SimilarityRule How to build similarity (affinity) matrix given a dataset
 * @tparam TransformRule How to transform similarity matrix
 * @tparam EigenmapRule How to generate embedding vectors from transformed
 *    similarity matrix
 */
template<
    typename SimilarityRule = MDSSimilarity<>,
    typename TransformRule = MDSTransform<>,
    typename EigenmapRule = MDSEigenmap<>,
    typename MatType = arma::mat,
    typename VecType = arma::colvec>
class MDSExtendedKernel
{
 public:
  /**
   * Create a MDSExtendedKernel object and set the parameters.
   * @param data Original data set
   * @param similarity Adopted similarity rule
   * @param transform Adopted transform rule
   * @param eigenmap Adopted eigenmap rule
   */
  MDSExtendedKernel(const MatType& data,
                    SimilarityRule& similarity,
                    TransformRule& transform,
                    EigenmapRule& eigenmap) :
      data(data), similarity(similarity), transform(transform),
      eigenmap(eigenmap), similarityMat(similarity.SimilarityMat())
  {
    /* nothing to do */
  }
  
  /**
   * Fit the new data point
   * @param point new data point
   */
  void FitPoint(const VecType& point)
  {
    distances.zeros(data.n_cols);
    
    // precalculate the expectation over the whole original data set
    entireE = 0;
    for (size_t i = 0; i < data.n_cols; ++i)
      for (size_t j = i; j < data.n_cols; ++j)
        entireE += std::pow(similarityMat(i, j), 2);
    entireE /= data.n_cols * (data.n_cols + 1) / 2;
    
    // precalculate the expectation for the new point
    pointE = 0;
    for (size_t i = 0; i < data.n_cols; ++i)
    {
      distances(i) = metric::SquaredEuclideanDistance::Evaluate(
          data.unsafe_col(i), point);
      pointE += distances(i);
    }
    pointE /= data.n_cols;
    
  }
  
  /**
   * Evaluate the value for the new data point and a given point
   * @param target the index of the given point
   */
  double Evaluate(size_t target)
  {
    return -0.5 * (distances(target) - pointE - Expectation(target) + entireE);

  }
  
 private:
  
  /**
   * calculate the expectation for a point
   * @param fixed the index of the given point
   */
  double Expectation(size_t fixed)
  {
    double E = 0;
    
    for (size_t i = 0; i < data.n_cols; ++i)
      E += std::pow(similarityMat(fixed, i), 2);
    
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
  
  //! Locally-stored similarity matrix
  const MatType& similarityMat;
  
  //! Expectation over the whole data set
  double entireE;
  
  //! Expectation for the new point
  double pointE;
  
  //! Distances between the new point and original data points
  arma::colvec distances;
  
};
    
}; // namespace manifold
}; // namespace mlpack

#endif
