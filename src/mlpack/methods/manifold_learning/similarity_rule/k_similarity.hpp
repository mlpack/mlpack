/**
 * @file k_similarity.hpp
 * @author Shangtong Zhang
 * 
 * Implementation for building similarity matrix on given dataset
 * using k nearest neighbors of each point.
 */
#ifndef __MLPACK_METHODS_MANIFOLD_LEARNING_SIMILARITY_RULE_K_SIMILARITY
#define __MLPACK_METHODS_MANIFOLD_LEARNING_SIMILARITY_RULE_K_SIMILARITY

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

namespace mlpack {
namespace manifold {
  
/**
 * This class is an implementation for building a similarity matrix on 
 * given dataset using k nearest neighbors of each point.
 *
 * Let M the similarity matrix.
 * M(i, j) = 0 if point j doesn't appear in the k nearest neighbors of point i
 * otherwise M(i, j) will be the distance between point i and point j
 *
 * @tparam UseLocalWeight If true, this class will try to reconstruct point i from 
 *    its k nearest neighbors. Thus M(i, j) will be the weight for point j when
 *    reconstructing point i. This is used for LLE.
 * @tparam MetricType metric to calculate distances
 */

template<
    bool UseLocalWeight = false,
    typename MetricType = metric::EuclideanDistance,
    typename SimilarityMatType = arma::sp_mat,
    typename MatType = arma::mat>
class KSimilarity
{
 public:
  
  /**
   * Create a KSimilarity object and set the parameters.
   * @param nNeighbors number of neighbors to use
   * @param tol regularizer used for calculate local weight for LLE
   */
  KSimilarity(size_t nNeighbors = 50, double tol = 1e-3) :
      nNeighbors(nNeighbors), tol(tol)
  {
    /* nothing to do */
  }
  
  // searcher type of this similarity rule
  typedef neighbor::NeighborSearch<neighbor::NearestNeighborSort, MetricType>
      SearcherType;
  
  /**
   * Given a new point, calculate its similarity vector.
   * @param searcher In Searcher containing original dataset.
   * @param data In New point
   * @param neighbors Out store neighbors of the new point
   * @param distances Out store distances between the new point and its neighbors
   */
  void BuildSimilarityVec(SearcherType& searcher,
                          const arma::colvec& data,
                          std::vector<size_t>& neighbors,
                          std::vector<double>& distances)
  {
    arma::Mat<size_t> neighborsMat;
    arma::mat distancesMat;
    searcher.Search(data, nNeighbors, neighborsMat, distancesMat);
    neighbors.clear();
    distances.clear();
    for (size_t i = 0; i < neighborsMat.n_rows; ++i)
    {
      neighbors.push_back(neighborsMat(i, 0));
      distances.push_back(distancesMat(i, 0));
    }
  }
  
  /**
   * Build similarity matrix on given dataset without calculating
   * weight for reconstruction. The similarity matrix is stored in this object.
   * 
   * @param data The dataset
   * @return The reference of similarity matrix(M)
   *    M(i, j) = 0 if point j doesn't appear in the k nearest neighbors of point i
   *    otherwise M(i, j) will be the distance between point i and point j
   */
  template<bool Local = UseLocalWeight>
  typename std::enable_if<!Local, SimilarityMatType&>::type
  BuildSimilarityMat(const MatType& data)
  {
    assert(data.n_cols);
    
    similarityMat.zeros(data.n_cols, data.n_cols);
    neighbor::NeighborSearch<neighbor::NearestNeighborSort, MetricType> neighborSearcher(data);
    arma::Mat<size_t> neighbors;
    arma::mat distances;
    neighborSearcher.Search(nNeighbors, neighbors, distances);
    for (size_t i = 0; i < neighbors.n_cols; ++i)
    {
      for (size_t j = 0; j < nNeighbors; ++j)
      {
        similarityMat(i, neighbors(j, i)) = distances(j, i);
      }
    }
    return similarityMat;
  }
  
  /**
   * Build similarity matrix on given dataset while calculating
   * weight for reconstruction. The similarity matrix is stored in this object.
   *
   * @param data The dataset
   * @return The reference of similarity matrix(M)
   *    M(i, j) = 0 if point j doesn't appear in the k nearest neighbors of point i.
   *    Otherwise M(i, j) will be the weight for point j when reconstructing point i.
   */
  template<bool Local = UseLocalWeight>
  typename std::enable_if<Local, SimilarityMatType&>::type
  BuildSimilarityMat(const MatType& data)
  {
    assert(data.n_cols);
    
    similarityMat.zeros(data.n_cols, data.n_cols);
    neighbor::NeighborSearch<neighbor::NearestNeighborSort, MetricType> neighborSearcher(data);
    arma::Mat<size_t> neighbors;
    arma::mat distances;
    neighborSearcher.Search(nNeighbors, neighbors, distances);
    
    arma::colvec onesVec(nNeighbors);
    onesVec.ones();
    
    // calculate weight for reconstruction
    for (size_t i = 0; i < data.n_cols; ++i)
    {
      // retrieve neighbors of point i
      MatType neighborData;
      for (size_t index = 0; index < nNeighbors; ++index)
        neighborData.insert_cols(
            neighborData.n_cols, data.unsafe_col(neighbors(index, i)));
      
      // shift i's neighbors to origin
      for (size_t j = 0; j < nNeighbors; ++j)
        neighborData.col(j) -= data.unsafe_col(i);
      
      // calculate the local covariance
      MatType cov = neighborData.t() * neighborData;
      
      // if regularization is needed
      if (nNeighbors > data.n_rows)
        cov += arma::eye(nNeighbors, nNeighbors) * tol * arma::trace(cov);
      
      // solve cov * weight = 1
      arma::colvec w = arma::solve(cov, onesVec);
      
      // enforce sum(w)=1
      w /= arma::accu(w);
      
      // store the weight
      for (size_t j = 0; j < nNeighbors; ++j)
        similarityMat(i, neighbors(j, i)) = w(j);
    }
    return similarityMat;
    
  }
  
  //! Get # of neighbors
  size_t NNeighbors() const { return nNeighbors; }
  //! Modify # of neighbors
  size_t& NNeighbors() { return nNeighbors; }

  //! Get similarity matrix
  SimilarityMatType& SimilarityMat() const { return similarityMat; }
  //! Modify similarity matrix
  SimilarityMatType& SimilarityMat() { return similarityMat; }
  
  //! Get tolerance for regularization
  double& Tol() { return tol; }
  //! Modify tolerance for regularization
  double Tol() const { return tol; }
  
 private:
  //! Locally-stored # of neighbors
  size_t nNeighbors;
  
  //! Locally-stored similarity matrix
  SimilarityMatType similarityMat;
  
  //! Locally-stored tolerance for regularization
  double tol;
};
  
// typedef for convenience
  
// define similarity rule for Isomap
template<
    typename MetricType = metric::EuclideanDistance,
    typename SimilarityMatType = arma::sp_mat,
    typename MatType = arma::mat>
using KIsomapSimilarity =
    KSimilarity<false, MetricType, SimilarityMatType, MatType>;
  
// define similarity rule for LE
template<
    typename MetricType = metric::EuclideanDistance,
    typename SimilarityMatType = arma::sp_mat,
    typename MatType = arma::mat>
using KLESimilarity =
    KSimilarity<false, MetricType, SimilarityMatType, MatType>;
  
// define similarity rule for LLE
template<
    typename MetricType = metric::EuclideanDistance,
    typename SimilarityMatType = arma::sp_mat,
    typename MatType = arma::mat>
using KLLESimilarity =
    KSimilarity<true, MetricType, SimilarityMatType, MatType>;
  
}; // namespace manifold
}; // namespace mlpack

#endif
