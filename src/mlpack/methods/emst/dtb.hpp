/**
* @file dtb.h
 *
 * @author Bill March (march@gatech.edu)
 *
 * Contains an implementation of the DualTreeBoruvka algorithm for finding a
 * Euclidean Minimum Spanning Tree.
 *
 * Citation: March, W. B.; Ram, P.; and Gray, A. G.  Fast Euclidean Minimum Spanning
 * Tree: Algorithm, Analysis, Applications.  In KDD, 2010.
 *
 */

#ifndef __MLPACK_METHODS_EMST_DTB_HPP
#define __MLPACK_METHODS_EMST_DTB_HPP

#include "emst.hpp"

#include <mlpack/core.h>
#include <mlpack/core/tree/bounds.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/kernels/lmetric.hpp>

PARAM(size_t, "leaf_size", "Size of the leaves.", "naive", 1, false);

namespace mlpack {
namespace emst {

/*
const fx_submodule_doc dtb_submodules[] = {
FX_SUBMODULE_DOC_DONE
};
 */

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
class DTBStat {
 private:
  double max_neighbor_distance_;
  size_t component_membership_;

 public:
  void set_max_neighbor_distance(double distance) {
    max_neighbor_distance_ = distance;
  }

  double max_neighbor_distance() {
    return max_neighbor_distance_;
  }

  void set_component_membership(size_t membership) {
    component_membership_ = membership;
  }

  size_t component_membership() {
    return component_membership_;
  }

  /**
    * A generic initializer.
    */
  DTBStat() {
    set_max_neighbor_distance(DBL_MAX);
    set_component_membership(-1);
  }

  /**
    * An initializer for leaves.
   */
  DTBStat(const arma::mat& dataset, size_t start, size_t count) {
    if (count == 1) {
      set_component_membership(start);
      set_max_neighbor_distance(DBL_MAX);
    } else {
      set_max_neighbor_distance(DBL_MAX);
      set_component_membership(-1);
    }
  }

  /**
    * An initializer for non-leaves.  Simply calls the leaf initializer.
   */
  DTBStat(const arma::mat& dataset, size_t start, size_t count,
          const DTBStat& left_stat, const DTBStat& right_stat) {
    if (count == 1) {
      set_component_membership(start);
      set_max_neighbor_distance(DBL_MAX);
    } else {
      set_max_neighbor_distance(DBL_MAX);
      set_component_membership(-1);
    }
  }

}; // class DTBStat


/**
 * Performs the MST calculation using the Dual-Tree Boruvka algorithm.
 */
class DualTreeBoruvka {

//  FORBID_ACCIDENTAL_COPIES(DualTreeBoruvka);

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

  int do_naive_;

  DTBTree* tree_;


////////////////// Constructors ////////////////////////

 public:

  DualTreeBoruvka() {}

  ~DualTreeBoruvka() {
    if (tree_ != NULL) {
      delete tree_;
    }
  }


  ////////////////////////// Private Functions ////////////////////
 private:

  /**
  * Adds a single edge to the edge list
   */
  void AddEdge_(size_t e1, size_t e2, double distance) {

    //EdgePair edge;
    mlpack::Log::Assert((e1 != e2),
        "Indices are equal in DualTreeBoruvka.add_edge(...)");

    mlpack::Log::Assert((distance >= 0.0),
        "Negative distance input in DualTreeBoruvka.add_edge(...)");

    if (e1 < e2) {
      edges_[number_of_edges_].Init(e1, e2, distance);
    }
    else {
      edges_[number_of_edges_].Init(e2, e1, distance);
    }

    number_of_edges_++;

  } // AddEdge_


  /**
   * Adds all the edges found in one iteration to the list of neighbors.
   */
  void AddAllEdges_() {

    for (size_t i = 0; i < number_of_points_; i++) {
      size_t component_i = connections_.Find(i);
      size_t in_edge_i = neighbors_in_component_[component_i];
      size_t out_edge_i = neighbors_out_component_[component_i];
      if (connections_.Find(in_edge_i) != connections_.Find(out_edge_i)) {
        double dist = neighbors_distances_[component_i];
        //total_dist_ = total_dist_ + dist;
        // changed to make this agree with the cover tree code
        total_dist_ = total_dist_ + sqrt(dist);
        AddEdge_(in_edge_i, out_edge_i, dist);
        connections_.Union(in_edge_i, out_edge_i);
      }
    }

  } // AddAllEdges_


  /**
    * Handles the base case computation.  Also called by naive.
  */
  double ComputeBaseCase_(size_t query_start, size_t query_end,
                          size_t reference_start, size_t reference_end) {

    number_leaf_computations_++;

    double new_upper_bound = -1.0;

    for (size_t query_index = query_start; query_index < query_end;
         query_index++) {

      // Find the index of the component the query is in
      size_t query_component_index = connections_.Find(query_index);

      arma::vec query_point = data_points_.col(query_index);

      for (size_t reference_index = reference_start;
           reference_index < reference_end; reference_index++) {

        size_t reference_component_index = connections_.Find(reference_index);

        if (query_component_index != reference_component_index) {

          arma::vec reference_point = data_points_.col(reference_index);

          double distance = mlpack::kernel::LMetric<2>::Evaluate(query_point,
              reference_point);

          if (distance < neighbors_distances_[query_component_index]) {

            mlpack::Log::Assert(query_index != reference_index);

            neighbors_distances_[query_component_index] = distance;
            neighbors_in_component_[query_component_index] = query_index;
            neighbors_out_component_[query_component_index] = reference_index;

          } // if distance

        } // if indices not equal


      } // for reference_index

      if (new_upper_bound < neighbors_distances_[query_component_index]) {
        new_upper_bound = neighbors_distances_[query_component_index];
      }

    } // for query_index

    mlpack::Log::Assert(new_upper_bound >= 0.0);
    return new_upper_bound;

  } // ComputeBaseCase_


  /**
    * Handles the recursive calls to find the nearest neighbors in an iteration
   */
  void ComputeNeighborsRecursion_(DTBTree *query_node, DTBTree *reference_node,
                                  double incoming_distance) {

    // Check for a distance prune
    if (query_node->stat().max_neighbor_distance() < incoming_distance) {
      //pruned by distance
      number_distance_prunes_++;
    }
    // Check for a component prune
    else if ((query_node->stat().component_membership() >= 0)
             && (query_node->stat().component_membership() ==
                 reference_node->stat().component_membership())) {
      //pruned by component membership

      mlpack::Log::Assert(reference_node->stat().component_membership() >= 0);

      number_component_prunes_++;
    }
    // The base case
    else if (query_node->is_leaf() && reference_node->is_leaf()) {

      double new_bound =
      ComputeBaseCase_(query_node->begin(), query_node->end(),
                       reference_node->begin(), reference_node->end());

      query_node->stat().set_max_neighbor_distance(new_bound);

    }
    // Other recursive calls
    else if (query_node->is_leaf()) {
      //recurse on reference_node only
      number_r_recursions_++;

      double left_dist =
        query_node->bound().MinDistance(reference_node->left()->bound());
      double right_dist =
        query_node->bound().MinDistance(reference_node->right()->bound());
      mlpack::Log::Assert(left_dist >= 0.0);
      mlpack::Log::Assert(right_dist >= 0.0);

      if (left_dist < right_dist) {
        ComputeNeighborsRecursion_(query_node,
                                   reference_node->left(), left_dist);
        ComputeNeighborsRecursion_(query_node,
                                   reference_node->right(), right_dist);
      }
      else {
        ComputeNeighborsRecursion_(query_node,
                                   reference_node->right(), right_dist);
        ComputeNeighborsRecursion_(query_node,
                                   reference_node->left(), left_dist);
      }

    }
    else if (reference_node->is_leaf()) {
      //recurse on query_node only

      number_q_recursions_++;

      double left_dist =
        query_node->left()->bound().MinDistance(reference_node->bound());
      double right_dist =
        query_node->right()->bound().MinDistance(reference_node->bound());

      ComputeNeighborsRecursion_(query_node->left(),
                                 reference_node, left_dist);
      ComputeNeighborsRecursion_(query_node->right(),
                                 reference_node, right_dist);

      // Update query_node's stat
      query_node->stat().set_max_neighbor_distance(
          std::max(query_node->left()->stat().max_neighbor_distance(),
              query_node->right()->stat().max_neighbor_distance()));

    }
    else {
      //recurse on both

      number_both_recursions_++;

      double left_dist = query_node->left()->bound().
        MinDistance(reference_node->left()->bound());
      double right_dist = query_node->left()->bound().
        MinDistance(reference_node->right()->bound());

      if (left_dist < right_dist) {
        ComputeNeighborsRecursion_(query_node->left(),
                                   reference_node->left(), left_dist);
        ComputeNeighborsRecursion_(query_node->left(),
                                   reference_node->right(), right_dist);
      }
      else {
        ComputeNeighborsRecursion_(query_node->left(),
                                   reference_node->right(), right_dist);
        ComputeNeighborsRecursion_(query_node->left(),
                                   reference_node->left(), left_dist);
      }

      left_dist = query_node->right()->bound().
        MinDistance(reference_node->left()->bound());
      right_dist = query_node->right()->bound().
        MinDistance(reference_node->right()->bound());

      if (left_dist < right_dist) {
        ComputeNeighborsRecursion_(query_node->right(),
                                   reference_node->left(), left_dist);
        ComputeNeighborsRecursion_(query_node->right(),
                                   reference_node->right(), right_dist);
      }
      else {
        ComputeNeighborsRecursion_(query_node->right(),
                                   reference_node->right(), right_dist);
        ComputeNeighborsRecursion_(query_node->right(),
                                   reference_node->left(), left_dist);
      }

      query_node->stat().set_max_neighbor_distance(
          std::max(query_node->left()->stat().max_neighbor_distance(),
              query_node->right()->stat().max_neighbor_distance()));

    }// end else

  } // ComputeNeighborsRecursion_

  /**
    * Computes the nearest neighbor of each point in each iteration
   * of the algorithm
   */
  void ComputeNeighbors_() {
    if (do_naive_) {
      ComputeBaseCase_(0, number_of_points_, 0, number_of_points_);
    }
    else {
      ComputeNeighborsRecursion_(tree_, tree_, DBL_MAX);
    }
  } // ComputeNeighbors_


  struct SortEdgesHelper_ {
    bool operator() (const EdgePair& pairA, const EdgePair& pairB) {
      return (pairA.distance() > pairB.distance());
    }
  } SortFun;

  void SortEdges_() {

    std::sort(edges_.begin(), edges_.end(), SortFun);

  } // SortEdges_()

  /**
    * Unpermute the edge list and output it to results
   *
   * TODO: Make this sort the edge list by distance as well for hierarchical
   * clusterings.
   */
  void EmitResults_(arma::mat& results) {

    SortEdges_();

    mlpack::Log::Assert(number_of_edges_ == number_of_points_ - 1);
    results.set_size(3, number_of_edges_);

    if (!do_naive_) {
      for (size_t i = 0; i < (number_of_points_ - 1); i++) {

        edges_[i].set_lesser_index(old_from_new_permutation_[edges_[i]
          .lesser_index()]);

        edges_[i].set_greater_index(old_from_new_permutation_[edges_[i]
          .greater_index()]);

        results(0, i) = edges_[i].lesser_index();
        results(1, i) = edges_[i].greater_index();
        results(2, i) = sqrt(edges_[i].distance());

      }
    }
    else {

      for (size_t i = 0; i < number_of_edges_; i++) {
        results(0, i) = edges_[i].lesser_index();
        results(1, i) = edges_[i].greater_index();
        results(2, i) = sqrt(edges_[i].distance());
      }

    }

  } // EmitResults_



  /**
    * This function resets the values in the nodes of the tree
   * nearest neighbor distance, check for fully connected nodes
   */
  void CleanupHelper_(DTBTree* tree) {

    tree->stat().set_max_neighbor_distance(DBL_MAX);

    if (!tree->is_leaf()) {
      CleanupHelper_(tree->left());
      CleanupHelper_(tree->right());

      if ((tree->left()->stat().component_membership() >= 0)
          && (tree->left()->stat().component_membership() ==
              tree->right()->stat().component_membership())) {
        tree->stat().set_component_membership(tree->left()->stat().
                                              component_membership());
      }
    }
    else {

      size_t new_membership = connections_.Find(tree->begin());

      for (size_t i = tree->begin(); i < tree->end(); i++) {
        if (new_membership != connections_.Find(i)) {
          new_membership = -1;
          mlpack::Log::Assert(tree->stat().component_membership() < 0);
          return;
        }
      }
      tree->stat().set_component_membership(new_membership);

    }

  } // CleanupHelper_

  /**
    * The values stored in the tree must be reset on each iteration.
   */
  void Cleanup_() {

    for (size_t i = 0; i < number_of_points_; i++) {
      neighbors_distances_[i] = DBL_MAX;
      //DEBUG_ONLY(neighbors_in_component_[i] = BIG_BAD_NUMBER);
      //DEBUG_ONLY(neighbors_out_component_[i] = BIG_BAD_NUMBER);
    }
    number_of_loops_++;

    if (!do_naive_) {
      CleanupHelper_(tree_);
    }
  }

  /**
    * Format and output the results
   */
  void OutputResults_() {

    /* fx_result_double(module_, "total_squared_length", total_dist_);
    fx_result_int(module_, "number_of_points", number_of_points_);
    fx_result_int(module_, "dimension", data_points_.n_rows);
    fx_result_int(module_, "number_of_loops", number_of_loops_);
    fx_result_int(module_, "number_distance_prunes", number_distance_prunes_);
    fx_result_int(module_, "number_component_prunes", number_component_prunes_);
    fx_result_int(module_, "number_leaf_computations", number_leaf_computations_);
    fx_result_int(module_, "number_q_recursions", number_q_recursions_);
    fx_result_int(module_, "number_r_recursions", number_r_recursions_);
    fx_result_int(module_, "number_both_recursions", number_both_recursions_);*/
    // TODO, not sure how I missed this last time.
    mlpack::Log::Info << "total_squared_length" << total_dist_ << std::endl;
    mlpack::Log::Info << "number_of_points" << number_of_points_ << std::endl;
    mlpack::Log::Info << "dimension" << data_points_.n_rows << std::endl;
    mlpack::Log::Info << "number_of_loops" << std::endl;
    mlpack::Log::Info << "number_distance_prunes" << std::endl;
    mlpack::Log::Info << "number_component_prunes" << std::endl;
    mlpack::Log::Info << "number_leaf_computations" << std::endl;
    mlpack::Log::Info << "number_q_recursions" << std::endl;
    mlpack::Log::Info << "number_r_recursions" << std::endl;
    mlpack::Log::Info << "number_both_recursions" << std::endl;
  } // OutputResults_

  /////////// Public Functions ///////////////////

 public:

  size_t number_of_edges() {
    return number_of_edges_;
  }


  /**
   * Takes in a reference to the data set and a module.  Copies the data,
   * builds the tree, and initializes all of the member variables.
   *
   * This module will be checked for the optional parameters "leaf_size" and
   * "do_naive".
   */
  void Init(const arma::mat& data) {

    number_of_edges_ = 0;
    data_points_ = data; // copy

    do_naive_ = CLI::GetParam<bool>("naive/do_naive");

    if (!do_naive_) {
      // Default leaf size is 1
      // This gives best pruning empirically
      // Use leaf_size=1 unless space is a big concern
      CLI::GetParam<int>("tree/leaf_size") =
          CLI::GetParam<size_t>("naive/leaf_size");

      CLI::StartTimer("naive/tree_building");

      tree_ = new DTBTree(data_points_, old_from_new_permutation_);

      CLI::StopTimer("naive/tree_building");
    }
    else {
      tree_ = NULL;
      old_from_new_permutation_.resize(0);
    }

    number_of_points_ = data_points_.n_cols;
    edges_.resize(number_of_points_ - 1, EdgePair()); // fill with EdgePairs
    connections_.Init(number_of_points_);

    neighbors_in_component_.set_size(number_of_points_);
    neighbors_out_component_.set_size(number_of_points_);
    neighbors_distances_.set_size(number_of_points_);
    neighbors_distances_.fill(DBL_MAX);

    total_dist_ = 0.0;
    number_of_loops_ = 0;
    number_distance_prunes_ = 0;
    number_component_prunes_ = 0;
    number_leaf_computations_ = 0;
    number_q_recursions_ = 0;
    number_r_recursions_ = 0;
    number_both_recursions_ = 0;
  } // Init


  /**
   * Call this function after Init.  It will iteratively find the nearest
   * neighbor of each component until the MST is complete.
   */
  void ComputeMST(arma::mat& results) {

    CLI::StartTimer("emst/MST_computation");

    while (number_of_edges_ < (number_of_points_ - 1)) {
      ComputeNeighbors_();

      AddAllEdges_();

      Cleanup_();

      Log::Info << "number_of_loops = " << number_of_loops_ << std::endl;
    }

    CLI::StopTimer("emst/MST_computation");

//    if (results != NULL) {

      EmitResults_(results);

//    }


    OutputResults_();

  } // ComputeMST

}; //class DualTreeBoruvka

}; // namespace emst
}; // namespace mlpack

#endif // __MLPACK_METHODS_EMST_DTB_HPP
