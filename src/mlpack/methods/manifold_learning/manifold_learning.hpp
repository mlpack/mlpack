/**
 * @file manifold_learning.hpp
 * @author Shangtong Zhang
 *
 * Manifold Learning
 *
 * This class implements a framework for some manifold learning
 * algorithms including Locally Linear Embedding(LLE), 
 * Isometirc Feature Mapping(Isomap), Multidimensional Scaling(MDS)
 * and Laplacian Eigenmaps(LE).
 *
 * This implementation is based on
 * Bengio, Yoshua, et al. "Out-of-sample extensions for lle, isomap, mds, eigenmaps, 
 * and spectral clustering." 
 * Advances in neural information processing systems 16 (2004): 177-184.
 *
 */

#ifndef __MLPACK_METHODS_MANIFOLD_LEARNING_MANIFOLD_HPP
#define __MLPACK_METHODS_MANIFOLD_LEARNING_MANIFOLD_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <boost/utility.hpp>

#include <mlpack/methods/manifold_learning/similarity_rule/k_similarity.hpp>
#include <mlpack/methods/manifold_learning/similarity_rule/epsilon_similarity.hpp>

#include <mlpack/methods/manifold_learning/transform_rule/mds_transform.hpp>
#include <mlpack/methods/manifold_learning/transform_rule/lle_transform.hpp>
#include <mlpack/methods/manifold_learning/transform_rule/le_transform.hpp>
#include <mlpack/methods/manifold_learning/transform_rule/isomap_transform.hpp>

#include <mlpack/methods/manifold_learning/eigenmap_rule/standard_eigenmap.hpp>

#include <mlpack/methods/manifold_learning/extended_kernel/mds_extended_kernel.hpp>
#include <mlpack/methods/manifold_learning/extended_kernel/le_extended_kernel.hpp>
#include <mlpack/methods/manifold_learning/extended_kernel/lle_extended_kernel.hpp>
#include <mlpack/methods/manifold_learning/extended_kernel/isomap_extended_kernel.hpp>

namespace mlpack {
namespace manifold /** Manifold Learning. */ {

/**
 * @tparam SimlarityRule How to build similarity (affinity) matrix given a dataset
 * @tparam TransformRule How to transform similarity matrix
 * @tparam EigenmapRule How to generate embedding vectors from transformed
 *    similarity matrix
 * @tparam ExtendedKernelType This kernel is used to generate the embedding vector
 *    for a new point without recalculating the eigen value of the whole dataset
 */
template<
    typename SimilarityRule,
    typename TransformRule,
    typename EigenmapRule,
    typename ExtendedKernelType,
    typename SimilarityMatType = arma::sp_mat,
    typename MatType = arma::mat,
    typename VecType = arma::colvec>
class Manifold
{
 public:
  /**
   * Create a Manifold Learning object and set the parameters.
   * @param data The given dataset to perform manifold learning
   * @param dim The dimension of disired embedding vectors
   * @param similarity How to build similarity (affinity) matrix given a dataset
   * @param transform How to transform similarity matrix
   * @param eigenmap How to generate embedding vectors from transformed
   *    similarity matrix
   */
  Manifold(const MatType& data,
           size_t dim,
           const SimilarityRule& similarity = SimilarityRule(),
           const TransformRule& transform = TransformRule(),
           const EigenmapRule& eigenmap = EigenmapRule()) :
      data(data), similarity(similarity), transform(transform),
      eigenmap(eigenmap), dim(dim),
      extendedKernel(data, this->similarity, this->transform, this->eigenmap)
  {
    /* Nothing to do */
  }
  
  /**
   * Transform given data set into embedding vectors.
   * @param embeddingMat store embedding vectors
   *    each column represent an embedding vector.
   */
  void Transform(MatType& embeddingMat)
  {
    eigenmap.Eigenmap(
        transform.Transform(
            similarity.BuildSimilarityMat(data)), embeddingMat, dim);
  }
  
  /**
   * Transform a new point into its embedding form.
   * @param newPoint new point to be transformed.
   * @param embeddingVec store the embedding vector.
   */
  void Transform(const VecType& newPoint, VecType& embeddingVec)
  {
    extendedKernel.FitPoint(newPoint);
    embeddingVec.set_size(dim, 1);
    for (size_t i = 0; i < dim; ++i)
    {
      embeddingVec(i) = 0;
      
      for (size_t j = 0; j < data.n_cols; ++j)
        embeddingVec(i) += eigenmap.Eigvec()(j, i) * extendedKernel.Evaluate(j);
      
      embeddingVec(i) /= eigenmap.Eigval()(i);
    }
  }
  
  //! Get dataset
  MatType& Data() const { return data; }
  
  //! Get similarity rule
  SimilarityRule& Similarity() const { return similarity; }
  //! Modify similarity rule
  SimilarityRule& Similarity() { return similarity; }
  
  //! Get transform rule
  TransformRule& Transform() const { return transform; }
  //! Modify transform rule
  TransformRule& Transform() { return transform; }
  
  //! Get eigenmap rule
  EigenmapRule& Eigenmap() const { return eigenmap; }
  //! Modify eigenmap rule
  EigenmapRule& Eigenmap() { return eigenmap; }
  
  //! Get extended kernel
  ExtendedKernelType& ExtendedKernel() const { return extendedKernel; }
  //! Modify extended kernel
  ExtendedKernelType& ExtendedKernel() { return extendedKernel; }
  
  //! Get dimension of embedding vectors
  size_t Dim() const { return dim; }
  //! Modify dimension of embedding vectors
  size_t& Dim() { return dim; }
  
 private:
  //! Locally-stored dataset
  const MatType& data;
  
  //! Instantiated similarity rule
  SimilarityRule similarity;
  
  //! Instantiated transform rule
  TransformRule transform;
  
  //! Instantiated eigenmap rule
  EigenmapRule eigenmap;
  
  //! Locally-stored dimension of embedding vectors
  size_t dim;
  
  //! Instantiated extended kernel
  ExtendedKernelType extendedKernel;
  
};
  
// typedef for convenience
  
// define MDS
template<
    typename MatType = arma::mat,
    typename VecType = arma::colvec>
using MDS = Manifold<
    MDSSimilarity<MatType>,
    MDSTransform<MatType>,
    MDSEigenmap<MatType>,
    MDSExtendedKernel<
        MDSSimilarity<MatType>,
        MDSTransform<MatType>,
        MDSEigenmap<MatType>,
        MatType,
        VecType>,
    arma::mat>;

/**
 * define Isomap
 * @tparam IsomapSimilarityRule How to build similarity matrix.
 *     Use k = 50 neighbors of each point by default.
 * @tparam ShortestPath How to calculate shortest path within a graph
 * @tparam MetricType metric to calculate distances
 */
template<
    typename IsomapSimilarityRule = KIsomapSimilarity<>,
    typename ShortestPath = FloydWarshall<>,
    typename MetricType = metric::EuclideanDistance,
    typename SimilarityMatType = arma::sp_mat,
    typename MatType = arma::mat,
    typename VecType = arma::colvec>
using Isomap = Manifold<
    IsomapSimilarityRule,
    IsomapTransform<ShortestPath, SimilarityMatType, MatType>,
    IsomapEigenmap<MatType>,
    IsomapExtendedKernel<
        IsomapSimilarityRule,
        IsomapTransform<ShortestPath, SimilarityMatType, MatType>,
        IsomapEigenmap<MatType>,
        MatType,
        VecType>,
    SimilarityMatType >;
  
/**
 * define LLE
 * @tparam LLESimilarityRule How to build similarity matrix.
 *     Use k = 50 neighbors of each point by default.
 * @tparam MetricType metric to calculate distances
 */

template<
    typename LLESimilarityRule = KLLESimilarity<>,
    typename MetricType = metric::EuclideanDistance,
    typename SimilarityMatType = arma::sp_mat,
    typename MatType = arma::mat,
    typename VecType = arma::colvec>
using LLE = Manifold<
    LLESimilarityRule,
    LLETransform<SimilarityMatType, MatType>,
    LLEEigenmap<MatType>,
    LLEExtendedKernel<
        LLESimilarityRule,
        LLETransform<SimilarityMatType, MatType>,
        LLEEigenmap<MatType>,
        MatType,
        VecType>,
    SimilarityMatType >;
  
/**
 * define LE
 * @tparam LESimilarityRule How to build similarity matrix.
 *    Use k = 50 neighbors of each point by default.
 * @tparam LETransformRule  How to transform similarity matrix.
 *    Use nearest neighbor kernel by default.
 * @tparam MetricType metric to calculate distances
 */
template<
    typename LESimilarityRule = KLESimilarity<>,
    typename LETransformRule = LETransform<>,
    typename MetricType = metric::EuclideanDistance,
    typename SimilarityMatType = arma::sp_mat,
    typename MatType = arma::mat,
    typename VecType = arma::colvec>
using LE = Manifold<
    LESimilarityRule,
    LETransformRule,
    LEEigenmap<MatType>,
    LEExtendedKernel<
        LESimilarityRule,
        LETransformRule,
        LEEigenmap<MatType>,
        MatType,
        VecType>,
    SimilarityMatType >;

} // namespace manifold
} // namespace mlpack


#endif // __MLPACK_METHODS_MANIFOLD_LEARNING_MANIFOLD_LEARNING_HPP