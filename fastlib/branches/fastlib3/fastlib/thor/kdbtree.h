// Copyright 2008 Georgia Institute of Technology. All rights reserved.
/**
 * @file thor/kdbtree.h
 *
 * Tools for kdB-trees.
 *
 * @experimental
 */

#ifndef THOR_KDBTREE_H
#define THOR_KDBTREE_H

#include "fastlib/base/base.h"

#include "thortree.h"
#include "cachearray.h"
#include "kdtree.h"

#include "fastlib/file/textfile.h"
#include "fastlib/data/dataset.h"
#include "fastlib/tree/bounds.h"
#include "fastlib/col/arraylist.h"
#include "fastlib/fx/fx.h"

/**
 * Single-threaded kdB-tree builder.
 *
 * Mimics adding points one at a time in accordance with the splitting
 * rules for kdB_FD trees.  Specifically, splits are always median
 * splits, and the first split made for as many points can fit in a
 * block is reused for all points that ultimately inhabit that region.
 * Nodes are maintained in a height-balanced data layout (broken only
 * to avoid creating pages containing just one subregion).
 *
 * To assist with dual-tree computation, this structure stores more
 * information for binary nodes than a traditional kdB-tree.  It keeps
 * bounding boxes and cached statistics.  It also forms nodes beneath
 * the minimum level needed to fit a region's points within one block
 * to optimize for base-case performance.
 */
template<typename TPoint, typename TNode, typename TParam>
class KdBTreeBuilder {
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
  CacheArray<Point> inputs_;
  CacheArray<Point> points_;
  CacheArray<Node> nodes_;
  index_t point_block_size_;
  index_t node_block_size_;
  index_t leaf_size_;
  index_t n_points_;
  index_t spare_block_i_;

 public:
  /**
   * Builds a kdB-tree.
   *
   * See class comments.
   */
  void Doit(
      fx_module* module, const Param* param_in,
      index_t begin_index, index_t end_index,
      DistributedCache* input_in,
      DistributedCache* points_create,
      DistributedCache* nodes_create,
      TreeDecomposition* decomposition);
 
 private:
  void FindBoundingBox_(index_t begin_index, index_t end_index, Bound* bound);

  index_t Build_(index_t begin_col, index_t end_col);
  void Insert_(const Point &input, index_t node_i);

  index_t SplitLeaf_(index_t node_i, bool make_new_block);
  index_t CreateChildren_(index_t node_i);
  index_t PackNodes_(
      index_t dest_i, index_t parent_i, Node *parent,
      index_t offset_i, index_t *node_ip);
  void FixChildrenParents_(index_t parent_i, const Node *parent);

  void MedianSplit_(const Node *node, Bound *left_bound, Bound *right_bound);

  index_t SplitLeafPages_(index_t node_i);
  void Postprocess_(Node *node);
};

namespace thor {

  /**
   * Creates a THOR kd-tree.
   *
   * @param param the parameter object for initializations
   * @param points_channel the channel for the points cache
   * @param nodes_channel the channel for the nodes cache
   * @param block_size_kb the upper-limit block size in kilobytes
   * @param megs the number of megabytes dedicated to the nodes cache
   * @param module where to get tree-building parameters such as leaf_size
   * @param n_points the number of data points
   * @param input_cache the cache of points to read from
   * @param points_cache the cache of points to initialize and create
   * @param nodes_cache the cache of nodes to initialize and create
   * @param decomposition the resulting tree decomposition
   */
  template<typename Point, typename Node, typename Param>
  void CreateKdBTreeMaster(const Param& param,
      int points_channel, int nodes_channel,
      int block_size_kb, double megs, datanode *module,
      index_t n_points, DistributedCache *input_cache,
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
   * @param points_channel the channel for the points distributed cache
   * @param nodes_channel the channel for the nodes distributed cache
   * @param extra_channel a free channel used for internal purposes
   * @param module the module to load config parameters from
   * @param n_points the number of points
   * @param input_cache the data points to read from
   * @param tree_out the tree encapsulation to create
   */
  template<typename Point, typename Node, typename Param>
  void CreateKdBTree(const Param& param,
      int points_channel, int nodes_channel, int extra_channel,
      datanode *module, index_t n_points, DistributedCache *input_cache,
      ThorTree<Param, Point, Node> *tree_out);

};

#include "kdbtree_impl.h"

#endif
