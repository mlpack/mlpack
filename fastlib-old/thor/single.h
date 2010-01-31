/**
 * @file single.h
 *
 * Single-tree simulator
 */

#ifndef THOR_SINGLE_H
#define THOR_SINGLE_H

#include "gnp.h"
#include "cachearray.h"

/**
 * Depth-first dual-tree solver.
 */
template<typename GNP>
class SingleTreeBreadth {
  FORBID_ACCIDENTAL_COPIES(SingleTreeBreadth);

 private:
  struct QueueItem {
    typename GNP::Delta delta;
    index_t r_index;
  };

  struct Queue {
    ArrayList<QueueItem> q;
    typename GNP::QSummaryResult summary_result;
    typename GNP::QPostponed postponed;

    void Init(const typename GNP::Param& param);

    /**
     * Consider a query-reference pair, possibly making an intrinsic prune.
     */
    void Consider(const typename GNP::Param& param,
        const typename GNP::QNode& q_node, const typename GNP::RNode& r_node,
        index_t r_index, const typename GNP::Delta& delta,
        typename GNP::GlobalResult *global_result);

    /**
     * Add an existing item back to the queue,
     * probably because it is a leaf.
     */
    void Reconsider(const typename GNP::Param& param,
        const QueueItem& item);
    
    void Done(const typename GNP::Param& param,
        const typename GNP::QPostponed& parent_postponed,
        const typename GNP::QNode& q_node);
  };

 private:
  typename GNP::Param param_;
  typename GNP::GlobalResult global_result_;

  CacheArray<typename GNP::QPoint> q_points_;
  CacheArray<typename GNP::QNode> q_nodes_;
  CacheArray<typename GNP::QResult> q_results_;

  CacheArray<typename GNP::RPoint> r_points_;
  CacheArray<typename GNP::RNode> r_nodes_;
  const typename GNP::RNode *r_root_;

  bool do_naive_;
  DualTreeRecursionStats stats_;

 public:
  SingleTreeBreadth() {}
  ~SingleTreeBreadth();

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
  COMPILER_NO_INLINE
  void Begin_(index_t q_root_index);
  COMPILER_NO_INLINE
  bool BeginExploringQueue_(
    const typename GNP::QNode& q_node, Queue *parent_queue);
#if 0
  void Divide_(index_t q_node_i);
#endif
  void DivideReferences_(
    const typename GNP::QPoint &q_point, index_t q_i,
    typename GNP::QResult *q_result,
    Queue* parent_queue);
  void BaseCase_(
      const typename GNP::QNode& q_node,
      const typename GNP::RNode& r_node,
      const typename GNP::Delta& delta,
      const typename GNP::QSummaryResult& unvisited);
  /**
   * Postprocesses results and pushes down any postponed prunes.
   */
  void PushDownPostprocess_(const typename GNP::QNode& q_node,
      const typename GNP::QPostponed& postponed);
};

#include "single_impl.h"

#endif
