/**
 * @file epsilon_similarity.hpp
 * @author Shangtong Zhang
 *
 * Implementation for building similarity matrix on given dataset
 * using neighbors of each point, distance between a neighbor and 
 * the point should be less than a given epsilon
 */
#ifndef __MLPACK_METHODS_MANIFOLD_LEARNING_SIMILARITY_RULE_EPSILON_SIMILARITY
#define __MLPACK_METHODS_MANIFOLD_LEARNING_SIMILARITY_RULE_EPSILON_SIMILARITY

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/range_search/range_search.hpp>

namespace mlpack {
namespace manifold {

/**
 * This class is an implementation for building a similarity matrix on given dataset
 * using neighbors of each point, distance between a neighbor and
 * the point should be less than a given epsilon
 *
 * Let M the similarity matrix.
 * M(i, j) = 0 if point j doesn't appear in the neighbors of point i
 * otherwise M(i, j) will be the distance between point i and point j
 *
 * @tparam UseLocalWeight If true, this class will try to reconstruct point i from
 *    its neighbors. Thus M(i, j) will be the weight for point j when
 *    reconstructing point i. This is used for LLE.
 * @tparam MetricType metric to calculate distances
 */
template<
    bool UseLocalWeight = false,
    typename MetricType = metric::EuclideanDistance,
    typename SimilarityMatType = arma::sp_mat,
    typename MatType = arma::mat>
class EpsilonSimilarity
{
 public:
  
  /**
   * Create an EpsilonSimilarity object and set the parameters.
   * @param epsilon Point i is the neighbor of point j if and only if the distance
   *    between point i and point j is less than epsilon.
   * @param tol regularizer used for calculate local weight for LLE
   */
  EpsilonSimilarity(double epsilon = arma::datum::inf, double tol = 1e-3) :
      epsilon(epsilon), tol(tol)
  {
    assert(epsilon > 0);
  }
  
  // searcher type of this similarity rule
  typedef range::RangeSearch<MetricType> SearcherType;
  
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
    math::Range validRadius(0, epsilon);
    std::vector<std::vector<double> > allDistances;
    std::vector<std::vector<size_t> > allNeighbors;
    searcher.Search(data, validRadius, allNeighbors, allDistances);
    neighbors.clear();
    neighbors = allNeighbors[0];
    distances.clear();
    distances = allDistances[0];
  }
  
  /**
   * Build similarity matrix on given dataset without calculating
   * weight for reconstruction. The similarity matrix is stored in this object.
   *
   * @param data The dataset
   * @return The reference of similarity matrix(M)
   *    M(i, j) = 0 if point j doesn't appear in the neighbors of point i
   *    otherwise M(i, j) will be the distance between point i and point j
   */
  template<bool Local = UseLocalWeight>
  typename std::enable_if<!Local, SimilarityMatType&>::type
  BuildSimilarityMat(const MatType& data)
  {
    assert(data.n_cols);
    similarityMat.zeros(data.n_cols, data.n_cols);
    
    if (epsilon == arma::datum::inf)
    {
      // just calculate the distance of each pair
      for (size_t i = 0; i < similarityMat.n_rows; ++i)
        for (size_t j = i + 1; j < similarityMat.n_cols; ++j)
          similarityMat(i, j) = similarityMat(j, i) =
              MetricType::Evaluate(data.unsafe_col(i), data.unsafe_col(j));
    }
    else
    {
      range::RangeSearch<MetricType> rangeSearcher(data);
      math::Range validRadius(0, epsilon);
      std::vector<std::vector<size_t> > neighbors;
      std::vector<std::vector<double> > distances;
      rangeSearcher.Search(validRadius, neighbors, distances);
      
      for (size_t i = 0; i < neighbors.size(); ++i)
        for (size_t j = 0; j < neighbors[i].size(); ++j)
          similarityMat(i, neighbors[i][j]) = distances[i][j];
    }
    return similarityMat;
  }
  
  /**
   * Build similarity matrix on given dataset while calculating
   * weight for reconstruction. The similarity matrix is stored in this object.
   *
   * @param data The dataset
   * @return The reference of similarity matrix(M)
   *    M(i, j) = 0 if point j doesn't appear in the neighbors of point i.
   *    Otherwise M(i, j) will be the weight for point j when reconstructing point i.
   */
  template<bool Local = UseLocalWeight>
  typename std::enable_if<Local, SimilarityMatType&>::type
  BuildSimilarityMat(const MatType& data)
  {
    assert(data.n_cols);
    
    similarityMat.zeros(data.n_cols, data.n_cols);
    
    range::RangeSearch<MetricType> rangeSearcher(data);
    math::Range validRadius(0, epsilon);
    std::vector<std::vector<size_t> > neighbors;
    std::vector<std::vector<double> > distances;
    rangeSearcher.Search(validRadius, neighbors, distances);
    
    // calculate weight for reconstruction
    for (size_t i = 0; i < data.n_cols; ++i)
    {
      size_t nNeighbors = neighbors[i].size();
      
      // retrieve neighbors of point i and shift i's neighbors to origin
      MatType neighborData(data.n_rows, nNeighbors);
      for (size_t j = 0; j < nNeighbors; ++j)
        neighborData.col(j) = data.unsafe_col(neighbors[i][j]) - data.unsafe_col(i);
      
      // calculate the local covariance
      MatType cov = neighborData.t() * neighborData;
      
      // if regularization is needed
      if (nNeighbors > data.n_rows)
        cov += arma::eye(nNeighbors, nNeighbors) * tol * arma::trace(cov);
      
      // solve cov * weight = 1
      arma::colvec onesVec(nNeighbors);
      onesVec.ones();
      arma::colvec w = arma::solve(cov, onesVec);
      
      // enforce sum(w)=1
      w /= arma::accu(w);
      
      // store the weight
      for (size_t j = 0; j < nNeighbors; ++j)
        similarityMat(i, neighbors[i][j]) = w(j);
    }
    return similarityMat;
  }
  
  //! Get epsilon
  double Epsilon() const { return epsilon; }
  //! Modify epsilon
  double& Epsilon() { return epsilon; }
  
  //! Get similarity matrix
  SimilarityMatType& SimilarityMat() const { return similarityMat; }
  //! Modify simiarity matrix
  SimilarityMatType& SimilarityMat() { return similarityMat; }
  
  //! Get tolerance for regularization
  double& Tol() { return tol; }
  //! Modify tolerance for regularization
  double Tol() const { return tol; }
  
 private:
  //! Locally-stored epsilon
  double epsilon;
  
  //! Locally-stored similarity matrix
  SimilarityMatType similarityMat;
  
  //! Locally-stored tolerance for regularization
  double tol;
};
  
// typedef for convenience
  
// define similarity rule for MDS
template<
    typename MatType = arma::mat>
using MDSSimilarity =
    EpsilonSimilarity<false, metric::SquaredEuclideanDistance, arma::mat, MatType>;

// define similarity rule for Isomap
template<
    typename MetricType = metric::EuclideanDistance,
    typename SimilarityMatType = arma::sp_mat,
    typename MatType = arma::mat>
using EpsilonIsomapSimilarity =
    EpsilonSimilarity<false, MetricType, SimilarityMatType, MatType>;
  
// define similarity rule for LE
template<
    typename MetricType = metric::EuclideanDistance,
    typename SimilarityMatType = arma::sp_mat,
    typename MatType = arma::mat>
using EpsilonLESimilarity =
    EpsilonSimilarity<false, MetricType, SimilarityMatType, MatType>;
  
// define similarity rule for LLE
template<
    typename MetricType = metric::EuclideanDistance,
    typename SimilarityMatType = arma::sp_mat,
    typename MatType = arma::mat>
using EpsilonLLESimilarity =
    EpsilonSimilarity<true, MetricType, SimilarityMatType, MatType>;
  
}; // namespace manifold
}; // namespace mlpack

#endif
