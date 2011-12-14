/**
 * @file dtb.hpp
 *
 * @author Bill March (march@gatech.edu)
 *
 * Contains an implementation of the DualTreeBoruvka algorithm for finding a
 * Euclidean Minimum Spanning Tree using the kd-tree data structure.
 *
 * Citation: March, W. B.; Ram, P.; and Gray, A. G.  Fast Euclidean Minimum
 * Spanning Tree: Algorithm, Analysis, Applications.  In KDD, 2010.
 *
 */
#ifndef __MLPACK_METHODS_EMST_DTB_HPP
#define __MLPACK_METHODS_EMST_DTB_HPP

#include "edge_pair.hpp"

#include <mlpack/core.hpp>
#include <mlpack/core/tree/bounds.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace emst {

/**
 * A Stat class for use with fastlib's trees.  This one only stores two values.
 *
 * @param max_neighbor_distance The upper bound on the distance to the nearest
 * neighbor of any point in this node.
 *
 * @param component_membership The index of the component that all points in
 * this node belong to.  This is the same index returned by UnionFind for all
 * points in this node.  If points in this node are in different components,
 * this value will be negative.
 */
class DTBStat
{
 private:
  double max_neighbor_distance_;
  int component_membership_;

 public:
  void set_max_neighbor_distance(double distance);

  double max_neighbor_distance();

  void set_component_membership(int membership);

  int component_membership();

  /**
   * A generic initializer.
   */
  DTBStat();

  /**
   * An initializer for leaves.
   */
  template<typename MatType>
  DTBStat(const MatType& dataset, const size_t start, const size_t count);

  /**
   * An initializer for non-leaves.  Simply calls the leaf initializer.
   */
  template<typename MatType>
  DTBStat(const MatType& dataset, const size_t start, const size_t count,
          const DTBStat& leftStat, const DTBStat& rightStat);

}; // class DTBStat

/**
 * Performs the MST calculation using the Dual-Tree Boruvka algorithm.
 */
class DualTreeBoruvka
{
 public:
  // For now, everything is in Euclidean space
  static const size_t metric = 2;

  typedef tree::BinarySpaceTree<bound::HRectBound<metric>, DTBStat> DTBTree;

  //////// Member Variables /////////////////////

 private:
  size_t number_of_edges_;
  std::vector<EdgePair> edges_; // must use vector with non-numerical types
  size_t number_of_points_;
  UnionFind connections_;
  struct datanode* module_;
  arma::mat data_points_;
  size_t leaf_size_;

  // lists
  std::vector<size_t> old_from_new_permutation_;
  arma::Col<size_t> neighbors_in_component_;
  arma::Col<size_t> neighbors_out_component_;
  arma::vec neighbors_distances_;

  // output info
  double total_dist_;
  size_t number_of_loops_;
  size_t number_distance_prunes_;
  size_t number_component_prunes_;
  size_t number_leaf_computations_;
  size_t number_q_recursions_;
  size_t number_r_recursions_;
  size_t number_both_recursions_;

  bool do_naive_;

  DTBTree* tree_;

  // for sorting the edge list after the computation
  struct SortEdgesHelper_
  {
    bool operator() (const EdgePair& pairA, const EdgePair& pairB)
    {
      return (pairA.distance() < pairB.distance());
    }
  } SortFun;
  

////////////////// Constructors ////////////////////////
 public:
  DualTreeBoruvka() { }

  ~DualTreeBoruvka();

  ////////////////////////// Private Functions ////////////////////
 private:
  /**
   * Adds a single edge to the edge list
   */
  void AddEdge_(size_t e1, size_t e2, double distance);
  
  /**
   * Adds all the edges found in one iteration to the list of neighbors.
   */
  void AddAllEdges_();
  
  /**
   * Handles the base case computation.  Also called by naive.
   */
  double ComputeBaseCase_(size_t query_start, size_t query_end,
                          size_t reference_start, size_t reference_end);
  
  /**
   * Handles the recursive calls to find the nearest neighbors in an iteration
   */
  void ComputeNeighborsRecursion_(DTBTree *query_node, DTBTree *reference_node,
                                  double incoming_distance);
  
  /**
   * Computes the nearest neighbor of each point in each iteration
   * of the algorithm
   */
  void ComputeNeighbors_();

  
  void SortEdges_();
  
  /**
   * Unpermute the edge list and output it to results
   *
   */
  void EmitResults_(arma::mat& results);

  /**
   * This function resets the values in the nodes of the tree nearest neighbor
   * distance, check for fully connected nodes
   */
  void CleanupHelper_(DTBTree* tree);

  /**
   * The values stored in the tree must be reset on each iteration.
   */
  void Cleanup_();
  
  /**
   * Format and output the results
   */
  void OutputResults_();
  
  /////////// Public Functions ///////////////////
 public:
  size_t number_of_edges();

  /**
   * Takes in a reference to the data set.  Copies the data, builds the tree,
   * and initializes all of the member variables.
   */
  void Init(const arma::mat& data, bool naive, size_t leafSize);
  
  /**
   * Call this function after Init.  It will iteratively find the nearest
   * neighbor of each component until the MST is complete.
   */
  void ComputeMST(arma::mat& results);
  
}; // class DualTreeBoruvka

}; // namespace emst
}; // namespace mlpack

#include "dtb_impl.hpp"

#endif // __MLPACK_METHODS_EMST_DTB_HPP
