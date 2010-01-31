// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file thor/kdtree.h
 *
 * Tools for kd-trees.
 *
 * @experimental
 */

#ifndef THOR_KDTREE_H
#define THOR_KDTREE_H

#include "fastlib/base/base.h"

#include "thortree.h"
#include "cachearray.h"

#include "fastlib/file/textfile.h"
#include "fastlib/data/dataset.h"
#include "fastlib/tree/bounds.h"
#include "fastlib/col/arraylist.h"
#include "fastlib/fx/fx.h"

/**
 * Single-threaded kd-tree builder.
 *
 * Rearranges points in place and attempts to take advantage of the block
 * structure.
 *
 * The algorithm uses a combination of midpoint and median splits.
 * At the higher levels of the tree, a median-like split is done such that
 * the split falls on the block boundary (or otherwise specified chunk_size)
 * that is closest to the middle index.  Once the number of points
 * considered is smaller than the chunk size, midpoint splits are done.
 * The median splits simplify load balancing and allow more efficient
 * storage of data, and actually help the dual-tree algorithm in the
 * initial few layers -- however, the midpoint splits help to separate
 * outliers from the rest of the data.  Leaves are created once the number
 * of points is at most leaf_size.
 */
template<typename TPoint, typename TNode, typename TParam>
class KdTreeHybridBuilder {
 public:
  typedef TNode Node;
  typedef TPoint Point;
  typedef typename TNode::Bound Bound;
  typedef TParam Param;
  typedef ThorTreeDecomposition<Node> TreeDecomposition;
  typedef typename TreeDecomposition::DecompNode DecompNode;

 private:
  struct HrectPartitionCondition {
    int dimension;
    double value;

    HrectPartitionCondition(int dimension_in, double value_in)
      : dimension(dimension_in)
      , value(value_in) {}

    bool is_left(const Vector& vector) const {
      return vector.get(dimension) < value;
    }
  };

 private:
  const Param* param_;
  CacheArray<Point> points_;
  CacheArray<Node> nodes_;
  index_t leaf_size_;
  index_t chunk_size_;
  index_t block_size_;
  index_t n_points_;

 public:
  /**
   * Builds a kd-tree.
   *
   * See class comments.
   *
   * @param module module for tuning parameters: leaf_size (maximum
   *        number of points per leaf), and chunk_size (rounding granularity
   *        for median splits)
   * @param param parameters needed by the bound or other structures
   * @param begin_index the first index that I'm building
   * @param end_index one beyond the last index
   * @param points_inout the points, to be reordered
   * @param nodes_create the nodes, which will be allocated one by one
   * @param decomposition_out will be initialized to a tree decomposition
   */
  void Doit(
      struct datanode* module, const Param* param,
      index_t begin_index, index_t end_index,
      DistributedCache* points_inout, DistributedCache* nodes_create,
      TreeDecomposition* decomposition_out);
 
 private:
  /** Determines the bounding box for a range of points. */
  void FindBoundingBox_(index_t begin_index, index_t end_index, Bound* bound);

  /** Builds a specific node in the tree. */
  index_t Build_(index_t begin_col, index_t end_col,
      int begin_rank, int end_rank,
      const Bound& bound, Node* parent, DecompNode** decomp_pp);

  /** Splits a node in the tree. */
  void Split_(Node* node, int begin_rank, int end_rank, int split_dim,
      Node *parent,
      DecompNode** left_decomp_pp, DecompNode** right_decomp_pp);
};

namespace thor {

  /**
   * Creates a THOR kd-tree.
   *
   * @param param the parameter object for initializations
   * @param nodes_channel the channel for the nodes cache
   * @param block_size_kb the upper-limit block size in kilobytes
   * @param megs the number of megabytes dedicated to the nodes cache
   * @param module where to get tree-building parameters such as leaf_size
   * @param n_points the number of data points
   * @param points_cache the cache of points to reorder
   * @param nodes_cache the cache of nodes to initialize and create
   * @param decomposition the resulting tree decomposition
   */
  template<typename Point, typename Node, typename Param>
  void CreateKdTreeMaster(const Param& param,
      int nodes_channel, int block_size_kb, double megs, datanode *module,
      index_t n_points,
      DistributedCache *points_cache, DistributedCache *nodes_cache,
      ThorTreeDecomposition<Node> *decomposition);

  /**
   * Creates a THOR kd-tree from an existing cache.
   *
   * Parameters taken from the @c module data node:
   * @li @c leaf_size number of points per leaf
   * @li @c block_size_kb maximum block size, in kilobytes
   * @li @c megs cache size, in megabytes (floating-point allowed)
   *
   * @param param parameter object for initializing new nodes
   * @param nodes_channel the channel for the nodes distributed cache
   * @param extra_channel a free channel used for internal purposes
   * @param module the module to load config parameters from
   * @param n_points the number of points
   * @param points_cache the data points to reorder, must be allocated via
   *        the new operator
   * @param tree_out the tree encapsulation to create
   */
  template<typename Point, typename Node, typename Param>
  void CreateKdTree(const Param& param,
      int nodes_channel, int extra_channel,
      datanode *module, index_t n_points,
      DistributedCache *points_cache,
      ThorTree<Param, Point, Node> *tree_out);

};

#include "kdtree_impl.h"

#endif
