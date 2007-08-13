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

#include "thortree.h"
#include "cachearray.h"

#include "file/textfile.h"
#include "data/dataset.h"
#include "tree/bounds.h"
#include "base/common.h"
#include "col/arraylist.h"
#include "fx/fx.h"

/* Implementation */

/**
 * A distributed kd-tree (works non-distributed too).
 */
template<typename TParam, typename TPoint, typename TNode>
class ThorKdTree {
  FORBID_COPY(ThorKdTree);

 public:
  typedef TParam Param;
  typedef TPoint Point;
  typedef TNode Node;
  typedef ThorTreeDecomposition<Node> TreeDecomposition;
  typedef typename ThorTreeDecomposition<Node>::DecompNode DecompNode;

 private:
  DistributedCache points_;
  DistributedCache nodes_;
  Param param_;
  TreeDecomposition *decomp_;

 public:
  ThorKdTree() {}
  ~ThorKdTree() {}

  create an init method

  /**
   * Performs a distributed tree-update.
   *
   * Note that this does NOT broadcast or reduce the visitor -- you must
   * take care of it if you need to store data in the visitor!
   */
  template<typename Result, typename Visitor>
  void Update(DistributedCache *results_cache, Visitor *visitor);

  /**
   * Creates a cache that's suitable for storing results or anything else.
   *
   * This array will be decomposed in the exact same way the points
   * array is decomposed (same block size and everything).
   */
  template<typename Result>
  void MakeDistributedCacheMaster(int channel, const Result& default_result,
      size_t total_ram, DistributedCache *results);

  /**
   * The worker version of create-a-new-distributed-cache.
   */
  template<typename Result>
  void MakeDistributedCacheWorker(int channel, size_t total_ram,
      DistributedCache *results);

  /**
   * Initializes a distributed cache all initialized to a particular value,
   * with the same topography as the original.
   *
   * Automatically dispatches between the master and worker method.
   *
   * If only the master machine can create a default result, it's probably
   * better not to call this method.
   */
  template<typename Result>
  void MakeDistributedCache(int channel, const Result& default_result,
      size_t total_ram, DistributedCache *results);

  /**
   * Gets the distributed cache associated with points.
   */
  DistributedCache& points() {
    return points_;
  }
  /**
   * Gets the distributed cache associated with nodes.
   */
  DistributedCache& nodes() { return nodes_; }

  /**
   * Gets the known parameter object.
   *
   * This object is not updated except when the tree is created.
   * If you need to update this, then use a broadcaster to relay the
   * new param object between machines, and use set_param!
   */
  Param& param() const {
    return *config_->param;
  }

  /**
   * Sets the parameter objct.
   */
  void set_param(Param &new_param) {
    delete *config_->param;
    config_->param = new Param(new_param);
  }

  index_t nodes_block() const { return config_->nodes_block; }
  index_t points_block() const { return config_->points_block; }
  index_t n_points() const { return config_->n_points; }

  ThorTreeDecomposition<Node>& decomposition() const {
    return config_->decomp;
  }

  void MasterBuildTree(int tag, datanode *module);

 private:
  void MasterLoadData(
      Config *config, Param *param, int tag, datanode *module);
};

template<typename TParam, typename TPoint, typename TNode>
void ThorKdTree<TParam, TPoint, TNode>::Init(Param **parampp, int param_tag,
    int base_channel, datanode *module) {
  datanode *nodes_module = fx_submodule(module, "nodes", "nodes");

  points_channel_ = base_channel + 0;
  nodes_channel_ = base_channel + 1;

  points_mb_ = fx_param_int(points_module_, "mb", 2000);
  nodes_mb_ = fx_param_int(nodes_module_, "mb", 500);
  if (rpc::is_root()) {
    Config config;
    MasterLoadData_(&config, *parampp, param_tag, module);
    MasterBuildTree_(&config, *parampp, param_tag, module);
    config.param = new Param(**parampp);
    config_broadcaster_.SetData(config);
  } else {
    CacheArray<Point>::MakeDistributedCacheWorker(points_channel_,
       size_t(points_mb_) * MEGABYTE, &points_);
    CacheArray<Node>::MakeDistributedCacheWorker(nodes_channel_,
       size_t(nodes_mb_) * MEGABYTE, &nodes_);
  }
  points_.StartSync();
  nodes_.StartSync();
  points_.WaitSync();
  nodes_.WaitSync();
  config_broadcaster_.Doit(base_channel + 2);
  config_ = &config_broadcaster_.get();
  delete *parampp;
  *parampp = new Param(*config_->param);
}

template<typename TParam, typename TPoint, typename TNode>
void ThorKdTree<TParam, TPoint, TNode>::MasterLoadData_(
    Config *config, Param *param, int tag, datanode *module) {
  config->points_block = fx_param_int(points_module_, "block", 1024);

  fprintf(stderr, "master: Reading data\n");

  fx_timer_start(module, "read");
  TextLineReader reader;
  if (FAILED(reader.Open(fx_param_str_req(module, "")))) {
    FATAL("Could not open data file '%s'", fx_param_str_req(module, ""));
  }
  DatasetInfo schema;
  schema.InitFromFile(&reader, "data");
  config->dim = schema.n_features();

  TPoint default_point;
  default_point.vec().Init(config->dim);
  default_point.vec().SetZero();
  param->InitPointExtras(tag, &default_point);

  CacheArray<Point>::MakeDistributedCacheMaster(
      points_channel_, config->points_block, default_point,
      size_t(points_mb_) * MEGABYTE,
      &points_);
  CacheArray<Point> points_array;
  points_array.Init(&points_, BlockDevice::M_CREATE);
  index_t i = 0;

  for (;;) {
    i = points_array.AllocD(rpc::rank(), 1);
    CacheWrite<Point> point(&points_array, i);
    bool is_done;
    success_t rv = schema.ReadPoint(&reader, point->vec().ptr(), &is_done);
    param->SetPointExtras(tag, i, point);
    if (unlikely(FAILED(rv))) {
      FATAL("Data file has problems");
    }
    if (is_done) {
      break;
    }
  }

  config->n_points = i;
  param->Bootstrap(tag, config->dim, config->n_points);

  fx_timer_stop(module, "read");
}

template<typename TParam, typename TPoint, typename TNode>
void ThorKdTree<TParam, TPoint, TNode>::BuildTree(
    Config *config, Param *param, int tag, datanode *module) {
  config->nodes_block = fx_param_int(nodes_module_, "block", 256);

  fprintf(stderr, "master: Building tree\n");
  fx_timer_start(module, "tree");
  Node example_node;
  example_node.Init(config->dim, *param);
  CacheArray<Node>::MakeDistributedCacheMaster(
      nodes_channel_, config->nodes_block, example_node,
      size_t(nodes_mb_) * MEGABYTE,
      &nodes_);
  KdTreeHybridBuilder<Point, Node, Param> builder;
  builder.Doit(module, param, 0, config->n_points,
          &points_, &nodes_, &config->decomp);
  fx_timer_stop(module, "tree");
}

#endif
