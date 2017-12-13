/**
 * @file hdbscan.hpp
 * @author Sudhanshu Ranjan
 *
 * An implementation of the HDBSCAN clustering method.
 * HDBSCAN works with variable density.
 *  
 *
 * @code
 * @inproceedings{
 *   author = {R. J. G. B. Campello, D. Moulavi & Sander, J.},
 *   title = {{Density-Based Clustering Based
 *             on Hierarchical Density Estimates.}},
 * }
 * @endcode
 * Reference : "https://github.com/EdwardRaff/JSAT/blob/master/JSAT/
 *              src/jsat/clustering/HDBSCAN.java"
 * Reference : "http://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html"
 */
#ifndef __MLPACK_METHODS_HDBSCAN_HDBSCAN_HPP
#define __MLPACK_METHODS_HDBSCAN_HDBSCAN_HPP

#include <mlpack/core.hpp>
#include <boost/dynamic_bitset.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/emst/dtb.hpp>
#include <mlpack/core/metrics/hdbscan_metric.hpp>

namespace mlpack {
namespace hdbscan {

template<typename NeighborSearch = neighbor::NeighborSearch
                                             <neighbor::NearestNeighborSort,
                                             metric::EuclideanDistance>,
         typename MetricType = metric::HdbscanMetric,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType = tree::BallTree>
class HDBSCAN
{
 public:
  /**
   * Construct the HDBSCAN object with the given parameters.
   *
   * @param minPoints The number of points present in a cluster.
   */
  HDBSCAN(const size_t minPoints = 10,
          bool allowSingleCluster = false);

  /**
   * Performs HDBSCAN clustering on the data,
   * returning the list of cluster assignments.
   * The assigned clusters have values from 0 onwards. The noise points  
   * are labelled as (total nuber of points + 1)
   *
   * @param MatType Type of matrix (arma::mat or arma::sp_mat).
   * @param data Dataset to cluster.
   * @param assignments Vector to store cluster assignments.
   */
  template<typename MatType>
  void Cluster(const MatType& data,
               arma::Row<size_t>& assignments);

 private:
  //! The parameter to compute core distance (and also
  // specifies the number of points in clusters now).
  size_t minPoints;

  //! Single cluster is allowed (or not)
  bool allowSingleCluster;

  //! Instantiated neighbor search.
  NeighborSearch neighborSearch;

  //! Metric defined
  MetricType hMetric;

  /**
   * Converts the input (sorted MST) to a single linkage tree.
   * Each row of the output contaisn 3 values
   * (cluster1 cluster2 d size)
   *  d = distance between cluster1 and cluster2
   *  size = size of cluster formed by merging cluster1 and cluster2
   *  cluster1 and cluster2 may contain more than one point
   *  Initially points are labelled as 0,1,2,...,(n-1) 
   *  and are single cluster(contain only one point)
   *  Then we go on merging 2 clusters at a time until we
   *  reach a point when all points are in one cluster
   *
   * @param MatType The type of matrix
   * @param inputMST The minimum spaning tree matrix is the input.
   * @param singleLinkageTree The output matrix in which single
                              linkage tree is stored.
   */
  template<typename MatType>
  void SingleLinkageTreeClustering(const MatType& inputMST,
                                   MatType& singleLinkageTree);

  /**
   * Performs a modified bfs on single linkage tree, 
   * helper function to condense tree
   * Only those clusters of single linkage tree 
   * are present in BFS which have at least 2 points.  
   *
   * @param MatType The type of matrix
   * @param singleLinkageTree The single linkage tree on which bfs 
   *                          will to be done.
   * @param BFS The output matrix in which bfs is stored.
   * @param rootOfBFS The root node of the BFS
   */
  template < typename MatType>
  void SingleLinkageTreeToModifiedBFS(const MatType& singleLinkageTree,
                                      std::vector<size_t>& BFS,
                                      size_t rootOfBFS);

  /**
   * This function condenses the single linkage tree .
   * Each row of output of condensed tree contains 4 values.
   * (parent child lambda size)
   * parent -> this cluster is above child in the condensed tree, 
   *           does not need to be immediate parent
   * child ->  this cluster is child,
   *           the parent belongs to a bigger cluster of which the
   *           child cluster is a part 
   * lambda -> lambda = (1/d)
   *           where d is distance between cluster containing 
   *           parent and clsuter containing child 
   * size -> size of cluster to which child belongs 
   *
   *
   * @param MatType The type of matrix
   * @param singleLinkageTree The single linkage tree on which has to be 
   *                          condensed.
   * @param result The output matrix in which condensed tree will be stored.
   * @param minClusterSize The minimum number of points that must remain 
   *                       in a cluster
   */
  template< typename MatType>
  void CondenseTree(const MatType& singleLinkageTree,
                    MatType& result,
                    size_t minClusterSize = 10);

  /**
   * This function coverts a clustered tree to BFS and chooses root as the 
   * specified node.  
   * Helper function of getClusters
   *
   * @param MatType The type of matrix
   * @param clusteredTree The clustered tree on on which bfs has to be 
   *                      performed.
   * @param resultBfs The vector in which output BFS will be store.
   * @param rootNode The root of the bfs trre.
   */
  template<typename MatType>
  void GetBfsFromClusteredTree(MatType& clusteredTree,
                               size_t rootNode,
                               std::vector<size_t>& resultBFS);

  /**
   * This function helps in assigning labels to all the points 
   * provided a condensedTree and points which can be roots of cluster.
   * Roots of cluster -> Present at the top of a cluster in condensed tree.
   * Helper function of getClusters 
   * Roots of cluster are labeled and all the children of a root are 
   * labeled same as root.  
   *
   * @param MatType Type of matrix
   * @param condensedTree Condensed tree .
   * @param clusters Vector in which potential roots are stored.
   * @param result Matrix whixh will contain final labels of all points.
   */
  template<typename MatType>
  void GetLabels(const MatType& condensedTree,
                 std::vector<size_t> clusters,
                 arma::Mat<size_t>& result);

  /**
   * This function helps in computing stabilities 
   * helper function of getClusters
   * 
   * For a cluster lmabda_birth = lambda when cluster split off from larger
   *                              cluster and became a new cluster
   * For a smaller cluster p inside a bigger cluster C, 
   * define lambda_p = lambda when cluster p falls off from 
   *                              the parent cluster C
   * Define stability of cluster C as :
   * stability = sum( (lambda_p - lamba_birth) * (size of child cluster) )
   *                              for all smaller clusters p in 
   *                              bigger cluster C
   * 
   * @param MatType Type of matrix
   * @param condensedTree Condensed tree .
   * @param result Stabilities of node.
   */
  template<typename MatType>
  void GetStabilities(const MatType& condensedTree,
                      std::map<size_t, double>& result);

  /**
   * This functin provides label to each and every point
   * Noise points are labelled as ( total number of points + 1) 
   *
   * @param MatType Type of matrix
   * @param condensedTree Condensed tree .
   * @param result Matrix whixh will contain final labels of all points.
   */
  template<typename MatType>
  void GetClusters(MatType& condensedTree,
                   arma::Mat<size_t>& result);
};

} // namespace hdbscan
} // namespace mlpack

// Include implementation.
#include "hdbscan_impl.hpp"

#endif
