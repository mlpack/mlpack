/**
 * @file dfs.h
 *
 * Depth-first dual-tree solver.
 */

#ifndef THOR_DFS_H
#define THOR_DFS_H

#include "gnp.h"
#include "cachearray.h"

/**
 * Depth-first dual-tree solver.
 */
template<typename GNP>
class DualTreeDepthFirst {
  FORBID_COPY(DualTreeDepthFirst);

 private:
  struct QMutables {
    typename GNP::QSummaryResult summary_result;
    typename GNP::QPostponed postponed;

    OT_DEF(QMutables) {
      OT_MY_OBJECT(summary_result);
      OT_MY_OBJECT(postponed);
    }
  };

 private:
  typename GNP::Param param_;
  typename GNP::GlobalResult global_result_;

  CacheArray<typename GNP::QPoint> q_points_;
  CacheArray<typename GNP::QNode> q_nodes_;
  CacheArray<typename GNP::QResult> q_results_;
  SubsetArray<QMutables> q_mutables_;

  CacheArray<typename GNP::RPoint> r_points_;
  CacheArray<typename GNP::RNode> r_nodes_;
  const typename GNP::RNode *r_root_;

  bool do_naive_;
  DualTreeRecursionStats stats_;

 public:
  DualTreeDepthFirst() {}
  ~DualTreeDepthFirst();

  /**
   * Solves the GNP.
   *
   * Results are stored in q_results and in this->global_result.
   * The datanode contains possible parameters, and records some
   * recursion statistics when debugging is enabled.
   * All the other arguments are the GNP input, and are not modified.
   */
  void Doit(
      const typename GNP::Param& param_in,
      index_t q_root_index,
      index_t q_node_end_index,
      DistributedCache *q_points,
      DistributedCache *q_nodes,
      DistributedCache *r_points,
      DistributedCache *r_nodes,
      DistributedCache *q_results);

  /**
   * Gets the global result after computation.
   */
  const typename GNP::GlobalResult& global_result() const {
    return global_result_;
  }
  
  const DualTreeRecursionStats& stats() const {
    return stats_;
  }

 private:
  COMPILER_NOINLINE
  void Begin_(index_t q_root_index);
  COMPILER_NOINLINE
  void Pair_(
      const typename GNP::QNode *q_node,
      const typename GNP::RNode *r_node,
      const typename GNP::Delta& delta,
      const typename GNP::QSummaryResult& unvisited,
      QMutables *q_node_mut);
  void BaseCase_(
      const typename GNP::QNode *q_node,
      const typename GNP::RNode *r_node,
      const typename GNP::Delta& delta,
      const typename GNP::QSummaryResult& unvisited,
      QMutables *q_node_mut);
  /**
   * Postprocesses results and pushes down any postponed prunes.
   */
  void PushDownPostprocess_(index_t q_node_i, QMutables *q_node_mut);
};

#include "dfs_impl.h"

#endif
