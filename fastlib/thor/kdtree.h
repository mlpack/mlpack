// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file kdtree.h
 *
 * Tools for kd-trees.
 *
 * @experimental
 */

#ifndef THOR_KDTREE_H
#define THOR_KDTREE_H

#include "kdtree_builder.h"
#include "thortree.h"
#include "cachearray.h"

#include "file/textfile.h"
#include "data/dataset.h"
#include "tree/bounds.h"
#include "base/common.h"
#include "col/arraylist.h"
#include "fx/fx.h"

namespace thor {
/**
 * Creates a THOR kd-tree.
 *
 * @param param the parameter object for initializations
 * @param nodes_channel the channel for the nodes cache
 * @param block_size_kb the upper-limit block size in kilobytes
 * @param megs the number of megabytes dedicated to the nodes cache
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
 * @param param parameter object
 * @param base_channel the first of a contiguous group of 5 channels
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

template<typename Point, typename Node, typename Param>
void thor::CreateKdTreeMaster(const Param& param,
    int nodes_channel, int block_size_kb, double megs, datanode *module,
    index_t n_points,
    DistributedCache *points_cache, DistributedCache *nodes_cache,
    ThorTreeDecomposition<Node> *decomposition) {
  Node example_node;

  example_node.stat().Init(param);
  Point example_point;
  CacheArray<Point>::GetDefaultElement(points_cache, &example_point);
  example_node.bound().Init(example_point.vec().length());

  CacheArray<Node>::CreateCacheMaster(nodes_channel,
      CacheArray<Node>::ConvertBlockSize(example_node, block_size_kb),
      example_node, megs, nodes_cache);
  KdTreeHybridBuilder<Point, Node, Param> builder;
  builder.Doit(module, &param, 0, n_points, points_cache, nodes_cache,
      decomposition);
}

template<typename Point, typename Node, typename Param>
void thor::CreateKdTree(const Param& param,
    int nodes_channel, int extra_channel,
    datanode *module, index_t n_points,
    DistributedCache *points_cache,
    ThorTree<Param, Point, Node> *tree_out) {
  double megs = fx_param_double(module, "megs", 1000);
  DistributedCache *nodes_cache = new DistributedCache();
  Broadcaster<ThorTreeDecomposition<Node> > broadcaster;

  if (rpc::is_root()) {
    ThorTreeDecomposition<Node> decomposition;
    int block_size_kb = fx_param_int(module, "block_size_kb", 64);
    CreateKdTreeMaster<Point, Node>(param,
        nodes_channel, block_size_kb, megs, module, n_points,
        points_cache, nodes_cache, &decomposition);
    broadcaster.SetData(decomposition);
  } else {
    CacheArray<Node>::CreateCacheWorker(nodes_channel, megs, nodes_cache);
  }

  points_cache->Sync();
  nodes_cache->Sync();
  broadcaster.Doit(extra_channel); // broadcast the decomposition

  tree_out->Init(param, broadcaster.get(), points_cache, nodes_cache);
}

#endif
