#ifndef NBR_DFS_H
#define NBR_DFS_H

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
  //datanode *datanode_;
  uint64 n_naive_;
  uint64 n_pre_naive_;
  uint64 n_recurse_;

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

 private:
  void Begin_(index_t q_root_index);
  void Pair_(
      const typename GNP::QNode *q_node,
      const typename GNP::RNode *r_node,
      const typename GNP::Delta& delta,
      const typename GNP::QSummaryResult& unvisited,
      QMutables *q_node_mut);
  void BaseCase_(
      const typename GNP::QNode *q_node,
      const typename GNP::RNode *r_node,
      const typename GNP::QSummaryResult& unvisited,
      QMutables *q_node_mut);
  void PushDown_(index_t q_node_i, QMutables *q_node_mut);
};

template<typename GNP>
DualTreeDepthFirst<GNP>::~DualTreeDepthFirst() {
  r_nodes_.StopRead(0);
}

template<typename GNP>
void DualTreeDepthFirst<GNP>::Doit(
    const typename GNP::Param& param_in,
    index_t q_root_index,
    index_t q_end_index,
    DistributedCache *q_points,
    DistributedCache *q_nodes,
    DistributedCache *r_points,
    DistributedCache *r_nodes,
    DistributedCache *q_results) {
  param_.Copy(param_in);

  q_nodes_.Init(q_nodes, BlockDevice::M_READ);
  r_points_.Init(r_points, BlockDevice::M_READ);
  r_nodes_.Init(r_nodes, BlockDevice::M_READ);

  const typename GNP::QNode *q_root = q_nodes_.StartRead(q_root_index);
  q_results_.Init(q_results, BlockDevice::M_OVERWRITE,
      q_root->begin(), q_root->end());
  q_points_.Init(q_points, BlockDevice::M_READ,
      q_root->begin(), q_root->end());
  q_nodes_.StopRead(q_root_index);

  QMutables default_mutable;
  default_mutable.summary_result.Init(param_);
  default_mutable.postponed.Init(param_);
  q_mutables_.Init(default_mutable, q_root_index, q_end_index);

  global_result_.Init(param_);

  r_root_ = r_nodes_.StartRead(0);

  //datanode_ = datanode_in;
  do_naive_ = false;

  Begin_(q_root_index);
}

template<typename GNP>
void DualTreeDepthFirst<GNP>::Begin_(index_t q_root_index) {
  typename GNP::Delta delta;
  const typename GNP::QNode *q_root = q_nodes_.StartRead(q_root_index);
  QMutables *q_root_mut = &q_mutables_[q_root_index];

  //fx_timer_start(datanode_, "execute");

  DEBUG_ONLY(n_naive_ = 0);
  DEBUG_ONLY(n_pre_naive_ = 0);
  DEBUG_ONLY(n_recurse_ = 0);

  bool need_explore = GNP::Algorithm::ConsiderPairIntrinsic(
      param_, *q_root, *r_root_, &delta,
      &global_result_, &q_root_mut->postponed);

  if (need_explore) {
    typename GNP::QSummaryResult empty_summary_result;

    empty_summary_result.Init(param_);

    if (do_naive_) {
      BaseCase_(q_root, r_root_, empty_summary_result, q_root_mut);
    } else {
      Pair_(q_root, r_root_, delta, empty_summary_result, q_root_mut);
    }
    PushDown_(q_root_index, q_root_mut);
  }

  q_nodes_.StopRead(q_root_index);

  //fx_timer_stop(datanode_, "execute");

#warning we should collect statistics
  /*DEBUG_ONLY(fx_format_result(datanode_, "naive_ratio", "%f",
      1.0 * n_naive_ / q_root->count() / r_root_->count()));
  DEBUG_ONLY(fx_format_result(datanode_, "naive_per_query", "%f",
      1.0 * n_naive_ / q_root->count()));
  DEBUG_ONLY(fx_format_result(datanode_, "pre_naive_ratio", "%f",
      1.0 * n_pre_naive_ / q_root->count() / r_root_->count()));
  DEBUG_ONLY(fx_format_result(datanode_, "pre_naive_per_query", "%f",
      1.0 * n_pre_naive_ / q_root->count()));
  DEBUG_ONLY(fx_format_result(datanode_, "recurse_ratio", "%f",
      1.0 * n_recurse_ / q_root->count() / r_root_->count()));
  DEBUG_ONLY(fx_format_result(datanode_, "recurse_per_query", "%f",
      1.0 * n_recurse_ / q_root->count()));*/

/*  if (fx_param_bool(datanode_, "print", 0)) {
    ot::Print(q_results_);
  }*/
}

template<typename GNP>
void DualTreeDepthFirst<GNP>::PushDown_(
    index_t q_node_i, QMutables *q_node_mut) {
  const typename GNP::QNode *q_node = q_nodes_.StartRead(q_node_i);

  if (q_node->is_leaf()) {
    for (index_t q_i = q_node->begin(); q_i < q_node->end(); q_i++) {
      typename GNP::QResult *q_result = q_results_.StartWrite(q_i);
      const typename GNP::QPoint *q_point = q_points_.StartRead(q_i);

      q_result->ApplyPostponed(param_, q_node_mut->postponed, *q_point, q_i);
      q_result->Postprocess(param_, *q_point, q_i, *r_root_);
      global_result_.ApplyResult(param_, *q_point, q_i, *q_result);
      q_results_.StopWrite(q_i);
      q_points_.StopRead(q_i);
    }
  } else {
    for (index_t k = 0; k < 2; k++) {
      index_t q_child_i = q_node->child(k);
      QMutables *q_child_mut = &q_mutables_[q_child_i];

      q_child_mut->postponed.ApplyPostponed(param_, q_node_mut->postponed);

      PushDown_(q_child_i, q_child_mut);
    }
  }

  q_nodes_.StopRead(q_node_i);
}

template<typename GNP>
void DualTreeDepthFirst<GNP>::Pair_(
    const typename GNP::QNode *q_node,
    const typename GNP::RNode *r_node,
    const typename GNP::Delta& delta,
    const typename GNP::QSummaryResult& unvisited,
    QMutables *q_node_mut) {
  //printf("pair(%d:%d) at (%d:%d)\n", q_node->begin(), q_node->count(), r_node->begin(), r_node->count());
  DEBUG_MSG(1.0, "Checking (%d,%d) x (%d,%d)",
      q_node->begin(), q_node->end(),
      r_node->begin(), r_node->end());
  DEBUG_ONLY(n_recurse_++);

  /* begin prune checks */
  typename GNP::QSummaryResult mu(q_node_mut->summary_result);
  mu.ApplyPostponed(param_, q_node_mut->postponed, *q_node);
  mu.ApplySummaryResult(param_, unvisited);
  mu.ApplyDelta(param_, delta);

  if (!GNP::Algorithm::ConsiderQueryTermination(
         param_, *q_node, mu, global_result_, &q_node_mut->postponed)) {
    q_node_mut->summary_result.ApplyDelta(param_, delta);
    DEBUG_MSG(1.0, "Termination prune");
  } else if (!GNP::Algorithm::ConsiderPairExtrinsic(
          param_, *q_node, *r_node, delta, mu, global_result_,
          &q_node_mut->postponed)) {
    DEBUG_MSG(1.0, "Extrinsic prune");
  } else {
    global_result_.UndoDelta(param_, delta);

    if (q_node->is_leaf() && r_node->is_leaf()) {
      DEBUG_MSG(1.0, "Base case");
      BaseCase_(q_node, r_node, unvisited, q_node_mut);
    } else if (r_node->is_leaf()
        || (q_node->count() >= r_node->count() && !q_node->is_leaf())) {
      DEBUG_MSG(1.0, "Splitting Q");
      // Phase 2: Explore children, and reincorporate their results.
      q_node_mut->summary_result.StartReaccumulate(param_, *q_node);

      for (index_t k = 0; k < 2; k++) {
        typename GNP::Delta child_delta;
        index_t q_child_i = q_node->child(k);
        const typename GNP::QNode *q_child = q_nodes_.StartRead(q_child_i);
        QMutables *q_child_mut = &q_mutables_[q_child_i];
        q_child_mut->postponed.ApplyPostponed(
            param_, q_node_mut->postponed);
        child_delta.Init(param_);

        if (GNP::Algorithm::ConsiderPairIntrinsic(
                param_, *q_child, *r_node, &child_delta,
                &global_result_, &q_child_mut->postponed)) {
          Pair_(q_child, r_node, delta, unvisited, q_child_mut);
        }

        // We must VERY carefully apply both the horizontal and vertical join
        // operators here for postponed results.
        typename GNP::QSummaryResult tmp_result(q_child_mut->summary_result);
        tmp_result.ApplyPostponed(param_, q_child_mut->postponed, *q_child);
        q_node_mut->summary_result.Accumulate(param_, tmp_result, q_node->count());

        q_nodes_.StopRead(q_child_i);
      }

      q_node_mut->summary_result.FinishReaccumulate(param_, *q_node);
      q_node_mut->postponed.Reset(param_);
    } else {
      DEBUG_MSG(1.0, "Splitting R");
      index_t r_child1_i = r_node->child(0);
      index_t r_child2_i = r_node->child(1);
      const typename GNP::RNode *r_child1 = r_nodes_.StartRead(r_child1_i);
      const typename GNP::RNode *r_child2 = r_nodes_.StartRead(r_child2_i);
      typename GNP::Delta delta1;
      typename GNP::Delta delta2;
      double heur1;
      double heur2;

      delta1.Init(param_);
      delta2.Init(param_);

      if (GNP::Algorithm::ConsiderPairIntrinsic(
          param_, *q_node, *r_child1, &delta1,
          &global_result_, &q_node_mut->postponed)) {
        heur1 = GNP::Algorithm::Heuristic(param_, *q_node, *r_child1, delta1);
      } else {
        r_child1 = NULL;
        heur1 = DBL_MAX;
      }
      if (GNP::Algorithm::ConsiderPairIntrinsic(
          param_, *q_node, *r_child2, &delta2,
          &global_result_, &q_node_mut->postponed)) {
        heur2 = GNP::Algorithm::Heuristic(param_, *q_node, *r_child2, delta2);
      } else {
        r_child2 = NULL;
        heur2 = DBL_MAX;
      }

      if (likely(heur1 <= heur2)) {
        if (r_child1 != NULL) {
          typename GNP::QSummaryResult unvisited_for_r1(
              unvisited);
          if (r_child2 != NULL) {
            unvisited_for_r1.ApplyDelta(param_, delta2);
          }
          Pair_(q_node, r_child1, delta1, unvisited_for_r1, q_node_mut);
        }
        if (r_child2 != NULL) {
          Pair_(q_node, r_child2, delta2, unvisited, q_node_mut);
        }
      } else {
        if (r_child2 != NULL) {
          typename GNP::QSummaryResult unvisited_for_r1(
              unvisited);
          if (r_child1 != NULL) {
            unvisited_for_r1.ApplyDelta(param_, delta1);
          }
          Pair_(q_node, r_child2, delta2, unvisited_for_r1, q_node_mut);
        }
        if (r_child1 != NULL) {
          Pair_(q_node, r_child1, delta1, unvisited, q_node_mut);
        }
      }
      r_nodes_.StopRead(r_child1_i);
      r_nodes_.StopRead(r_child2_i);
    }
  }
}

template<typename GNP>
void DualTreeDepthFirst<GNP>::BaseCase_(
    const typename GNP::QNode *q_node,
    const typename GNP::RNode *r_node,
    const typename GNP::QSummaryResult& unvisited,
    QMutables *q_node_mut) {
  typename GNP::PairVisitor visitor;

  DEBUG_ONLY(n_pre_naive_ += q_node->count() * r_node->count());

  visitor.Init(param_);

  q_node_mut->summary_result.StartReaccumulate(param_, *q_node);

  CacheRead<typename GNP::QPoint> first_q_point(&q_points_, q_node->begin());
  CacheWrite<typename GNP::QResult> first_q_result(&q_results_, q_node->begin());
  CacheRead<typename GNP::RPoint> first_r_point(&r_points_, r_node->begin());
  size_t q_point_stride = q_points_.n_elem_bytes();
  size_t q_result_stride = q_results_.n_elem_bytes();
  const typename GNP::QPoint *q_point = first_q_point;
  typename GNP::QResult *q_result = first_q_result;

  for (index_t q_i = q_node->begin(); q_i < q_node->end(); ++q_i) {
    q_result->ApplyPostponed(param_, q_node_mut->postponed, *q_point, q_i);

    if (visitor.StartVisitingQueryPoint(param_, *q_point, q_i, *r_node,
          unvisited, q_result, &global_result_)) {
      const typename GNP::RPoint *r_point = first_r_point;
      size_t r_point_stride = r_points_.n_elem_bytes();
      index_t r_end = r_node->end();
      index_t r_i = r_node->begin();

      for (;;) {
        visitor.VisitPair(param_, *q_point, q_i, *r_point, r_i);
        if (++r_i >= r_end) {
          break;
        }
        r_point = mem::PointerAdd(r_point, r_point_stride);
      }

      visitor.FinishVisitingQueryPoint(param_, *q_point, q_i, *r_node,
          unvisited, q_result, &global_result_);

      DEBUG_ONLY(n_naive_ += r_node->count());
    }

    q_node_mut->summary_result.Accumulate(param_, *q_result);

    q_point = mem::PointerAdd(q_point, q_point_stride);
    q_result = mem::PointerAdd(q_result, q_result_stride);
  }

  q_node_mut->summary_result.FinishReaccumulate(param_, *q_node);
  q_node_mut->postponed.Reset(param_);
}

// The old version -- doesn't allow delta re-use for heuristics
//
//       double r_child1_h = GNP::Algorithm::Heuristic(
//           param_, *q_node, *r_child1);
//       double r_child2_h = GNP::Algorithm::Heuristic(
//           param_, *q_node, *r_child2);
//
//       if (unlikely(r_child2_h < r_child1_h)) {
//         const typename GNP::RNode *r_child_t = r_child1;
//         r_child1 = r_child2;
//         r_child2 = r_child_t;
//
//         index_t r_child_t_i = r_child1_i;
//         r_child1_i = r_child2_i;
//         r_child2_i = r_child_t_i;
//       }
//
//       typename GNP::Delta delta1;
//       typename GNP::Delta delta2;
//
//       delta1.Init(param_);
//       delta2.Init(param_);
//
//       bool do_r2 = GNP::Algorithm::ConsiderPairIntrinsic(
//           param_, *q_node, *r_child2, &delta2,
//           &global_result_, &q_node_mut->postponed);
//
//       if (GNP::Algorithm::ConsiderPairIntrinsic(
//           param_, *q_node, *r_child1, &delta1,
//           &global_result_, &q_node_mut->postponed)) {
//         typename GNP::QSummaryResult unvisited_for_r1(
//             unvisited);
//         if (do_r2) {
//           unvisited_for_r1.ApplyDelta(param_, delta2);
//         }
//         Pair_(q_node, r_child1, delta1, unvisited_for_r1, q_node_mut);
//       }
//       if (do_r2) {
//         Pair_(q_node, r_child2, delta2, unvisited, q_node_mut);
//       }


#endif
