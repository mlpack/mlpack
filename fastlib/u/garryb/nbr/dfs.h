#ifndef NBR_DFS_H
#define NBR_DFS_H

#include "gnp.h"

template<typename GNP>
class DualTreeDepthFirst {
  FORBID_COPY(DualTreeDepthFirst);
  
 private:
  struct QMutableInfo {
    typename GNP::QMassResult mass_result;
    typename GNP::QPostponed postponed;
    
    OT_DEF(QMutableInfo) {
      OT_MY_OBJECT(mass_result);
      OT_MY_OBJECT(postponed);
    }
  };
  
 private:
  /*
  KdTreeMidpointBuilder<typename GNP::QPointInfo, typename GNP::QNode,
      typename GNP::Param> q_tree_;
  KdTreeMidpointBuilder<typename GNP::RPointInfo, typename GNP::RNode,
      typename GNP::Param> r_tree_;
  */
  typename GNP::Param param_;
  typename GNP::GlobalResult global_result_;

  CacheArray<typename GNP::Point> q_points_;
  CacheArray<typename GNP::QNode> q_nodes_;
  CacheArray<typename GNP::QResult> q_results_;
  TempCacheArray<QMutableInfo> q_mutables_;
  
  CacheArray<typename GNP::Point> r_points_;
  CacheArray<typename GNP::RNode> r_nodes_;
  const typename GNP::RNode *r_root_;

  bool do_naive_;
  datanode *datanode_;
  uint64 n_naive_;
  uint64 n_pre_naive_;
  uint64 n_recurse_;
  
 public:
  DualTreeDepthFirst() {}
  ~DualTreeDepthFirst();
  
  void InitSolve(
      datanode *datanode_in,
      const typename GNP::Param& param_in,
      index_t q_root_index,
      SmallCache *q_points,
      SmallCache *q_nodes,
      SmallCache *r_points,
      SmallCache *r_nodes,
      SmallCache *q_results);
  
  const typename GNP::GlobalResult& global_result() const {
    return global_result_;
  }

 private:
  void Begin_(index_t q_root_index);
  void Pair_(
      const typename GNP::QNode *q_node,
      const typename GNP::RNode *r_node,
      const typename GNP::Delta& delta,
      const typename GNP::QMassResult& exclusive_unvisited,
      QMutableInfo *q_node_mut);
  void BaseCase_(
      const typename GNP::QNode *q_node,
      const typename GNP::RNode *r_node,
      const typename GNP::QMassResult& exclusive_unvisited,
      QMutableInfo *q_node_mut);
  void PushDown_(index_t q_node_i, QMutableInfo *q_node_mut);
};

template<typename GNP>
DualTreeDepthFirst<GNP>::~DualTreeDepthFirst() {
  r_nodes_.StopRead(0);
  
  q_points_.Flush();
  q_nodes_.Flush();
  r_points_.Flush();
  r_nodes_.Flush();
  q_results_.Flush();
}

template<typename GNP>
void DualTreeDepthFirst<GNP>::InitSolve(
    struct datanode *datanode_in,
    const typename GNP::Param& param_in,
    index_t q_root_index,
    SmallCache *q_points,
    SmallCache *q_nodes,
    SmallCache *r_points,
    SmallCache *r_nodes,
    SmallCache *q_results) {
  param_.Copy(param_in);
  
  q_nodes_.Init(q_nodes, BlockDevice::READ);
  r_points_.Init(r_points, BlockDevice::READ);
  r_nodes_.Init(r_nodes, BlockDevice::READ);
  
  const typename GNP::QNode *q_root = q_nodes_.StartRead(q_root_index);
  q_results_.Init(q_results, BlockDevice::CREATE,
      q_root->begin(), q_root->end());
  q_points_.Init(q_points, BlockDevice::READ,
      q_root->begin(), q_root->end());
  q_nodes_.StopRead(q_root_index);
  
  
  QMutableInfo default_mutable;
  default_mutable.mass_result.Init(param_);
  default_mutable.postponed.Init(param_);
  q_mutables_.Init(default_mutable, q_nodes_.end_index(),
      q_nodes_.n_block_elems());
  
  global_result_.Init(param_);
  
  r_root_ = r_nodes_.StartRead(0);
  
  datanode_ = datanode_in;
  do_naive_ = fx_param_bool(datanode_, "do_naive", false);
  
  Begin_(q_root_index);
}

template<typename GNP>
void DualTreeDepthFirst<GNP>::Begin_(index_t q_root_index) {
  typename GNP::Delta delta;
  const typename GNP::QNode *q_root = q_nodes_.StartRead(q_root_index);
  QMutableInfo *q_root_mut = q_mutables_.StartWrite(q_root_index);

  fx_timer_start(datanode_, "execute");

  DEBUG_ONLY(n_naive_ = 0);
  DEBUG_ONLY(n_pre_naive_ = 0);
  DEBUG_ONLY(n_recurse_ = 0);

  bool need_explore = GNP::Algorithm::ConsiderPairIntrinsic(
      param_, *q_root, *r_root_, &delta,
      &global_result_, &q_root_mut->postponed);

  if (need_explore) {
    typename GNP::QMassResult empty_mass_result;

    empty_mass_result.Init(param_);

    if (do_naive_) {
      BaseCase_(q_root, r_root_, empty_mass_result, q_root_mut);
    } else {
      Pair_(q_root, r_root_, delta, empty_mass_result, q_root_mut);
      PushDown_(q_root_index, q_root_mut);
    }
  }

  q_nodes_.StopRead(q_root_index);
  q_mutables_.StopWrite(q_root_index);

  fx_timer_stop(datanode_, "execute");

  DEBUG_ONLY(fx_format_result(datanode_, "naive_ratio", "%f",
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
      1.0 * n_recurse_ / q_root->count()));

/*  if (fx_param_bool(datanode_, "print", 0)) {
    ot::Print(q_results_);
  }*/
}

template<typename GNP>
void DualTreeDepthFirst<GNP>::PushDown_(
    index_t q_node_i, QMutableInfo *q_node_mut) {
  const typename GNP::QNode *q_node = q_nodes_.StartRead(q_node_i);

  if (q_node->is_leaf()) {
    for (index_t q_i = q_node->begin(); q_i < q_node->end(); q_i++) {
      typename GNP::QResult *q_result = q_results_.StartWrite(q_i);
      const typename GNP::Point *q_point = q_points_.StartRead(q_i);
      const typename GNP::QPointInfo *q_info = NULL;
      
      q_result->ApplyPostponed(param_, q_node_mut->postponed, *q_point);
      q_result->Postprocess(param_, *q_point, *q_info, *r_root_);
      q_results_.StopWrite(q_i);
      q_points_.StopRead(q_i);
    }
  } else {
    for (index_t k = 0; k < 2; k++) {
      index_t q_child_i = q_node->child(k);
      QMutableInfo *q_child_mut = q_mutables_.StartWrite(q_child_i);
      
      q_child_mut->postponed.ApplyPostponed(param_, q_node_mut->postponed);
      
      PushDown_(q_child_i, q_child_mut);
      q_mutables_.StopWrite(q_child_i);
    }
  }

  q_nodes_.StopRead(q_node_i);
}

template<typename GNP>
void DualTreeDepthFirst<GNP>::Pair_(
    const typename GNP::QNode *q_node,
    const typename GNP::RNode *r_node,
    const typename GNP::Delta& delta,
    const typename GNP::QMassResult& exclusive_unvisited,
    QMutableInfo *q_node_mut) {
  DEBUG_MSG(1.0, "Checking (%d,%d) x (%d,%d)",
      q_node->begin(), q_node->end(),
      r_node->begin(), r_node->end());
  DEBUG_ONLY(n_recurse_++);

  /* begin prune checks */
  typename GNP::QMassResult mu(q_node_mut->mass_result);
  mu.ApplyPostponed(param_, q_node_mut->postponed, *q_node);
  mu.ApplyMassResult(param_, exclusive_unvisited);
  mu.ApplyDelta(param_, delta);
  
  if (!GNP::Algorithm::ConsiderQueryTermination(
         param_, *q_node, mu, global_result_, &q_node_mut->postponed)) {
    q_node_mut->mass_result.ApplyDelta(param_, delta);
    DEBUG_MSG(1.0, "Termination prune");
  } else if (!GNP::Algorithm::ConsiderPairExtrinsic(
          param_, *q_node, *r_node, delta, mu, global_result_,
          &q_node_mut->postponed)) {
    DEBUG_MSG(1.0, "Extrinsic prune");
  } else {
    global_result_.UndoDelta(param_, delta);

    if (q_node->is_leaf() && r_node->is_leaf()) {
      DEBUG_MSG(1.0, "Base case");
      BaseCase_(q_node, r_node, exclusive_unvisited, q_node_mut);
    } else if (r_node->is_leaf()
        || (q_node->count() >= r_node->count() && !q_node->is_leaf())) {
      DEBUG_MSG(1.0, "Splitting Q");
      // Phase 2: Explore children, and reincorporate their results.
      q_node_mut->mass_result.StartReaccumulate(param_, *q_node);

      for (index_t k = 0; k < 2; k++) {
        typename GNP::Delta child_delta;
        index_t q_child_i = q_node->child(k);
        const typename GNP::QNode *q_child = q_nodes_.StartRead(q_child_i);
        QMutableInfo *q_child_mut = q_mutables_.StartWrite(q_child_i);
        q_child_mut->postponed.ApplyPostponed(
            param_, q_node_mut->postponed);
        child_delta.Init(param_);

        if (GNP::Algorithm::ConsiderPairIntrinsic(
                param_, *q_child, *r_node, &child_delta,
                &global_result_, &q_child_mut->postponed)) {
          Pair_(q_child, r_node, delta, exclusive_unvisited, q_child_mut);
        }

        // We must VERY carefully apply both the horizontal and vertical join
        // operators here for postponed results.
        typename GNP::QMassResult tmp_result(q_child_mut->mass_result);
        tmp_result.ApplyPostponed(param_, q_child_mut->postponed, *q_child);
        q_node_mut->mass_result.Accumulate(param_, tmp_result, q_node->count());

        q_mutables_.StopWrite(q_child_i);
        q_nodes_.StopRead(q_child_i);
      }

      q_node_mut->mass_result.FinishReaccumulate(param_, *q_node);
      q_node_mut->postponed.Reset(param_);
    } else {
      DEBUG_MSG(1.0, "Splitting R");
      index_t r_child1_i = r_node->child(0);
      index_t r_child2_i = r_node->child(1);
      const typename GNP::RNode *r_child1 = r_nodes_.StartRead(r_child1_i);
      const typename GNP::RNode *r_child2 = r_nodes_.StartRead(r_child2_i);
      typename GNP::Delta delta1;
      typename GNP::Delta delta2;
      typename const GNP::Delta *second_delta;
      double heur1;
      double heur2;

      delta1.Init(param_);
      delta2.Init(param_);

      if (!GNP::Algorithm::ConsiderPairIntrinsic(
          param_, *q_node, *r_child1, &delta1,
          &global_result_, &q_node_mut->postponed)) {
        r_child1 = NULL;
        heur1 = DBL_MAX;
      } else {
        heur1 = GNP::Algorithm::Heuristic(param_, *q_node, *r_child2, *delta1);
      }
      if (!GNP::Algorithm::ConsiderPairIntrinsic(
          param_, *q_node, *r_child2, &delta2,
          &global_result_, &q_node_mut->postponed)) {
        r_child2 = NULL;
        heur2 = DBL_MAX;
      } else {
        heur2 = GNP::Algorithm::Heuristic(param_, *q_node, *r_child2, *delta2);
      }

      if (unlikely(heur2 < heur1)) {
        const typename GNP::RNode *r_child_t = r_child1;
        r_child1 = r_child2;
        r_child2 = r_child_t;
        second_delta = &delta1;
      } else {
        second_delta = &delta2;
      }
      
      if (r_child1 != NULL) {
        typename GNP::QMassResult exclusive_unvisited_for_r1(
            exclusive_unvisited);
        if (r_child2 != NULL) {
          exclusive_unvisited_for_r1.ApplyDelta(param_, *second_delta);
        }
        Pair_(q_node, r_child1, delta1, exclusive_unvisited_for_r1, q_node_mut);
      }
      if (r_child2 != NULL) {
        Pair_(q_node, r_child2, delta2, exclusive_unvisited, q_node_mut);
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
    const typename GNP::QMassResult& exclusive_unvisited,
    QMutableInfo *q_node_mut) {
  typename GNP::PairVisitor visitor;

  DEBUG_ONLY(n_pre_naive_ += q_node->count() * r_node->count());

  visitor.Init(param_);

  q_node_mut->mass_result.StartReaccumulate(param_, *q_node);

  for (index_t q_i = q_node->begin(); q_i < q_node->end(); ++q_i) {
    const typename GNP::Point *q_point = q_points_.StartRead(q_i);
    const typename GNP::QPointInfo *q_info = NULL;
    typename GNP::QResult *q_result = q_results_.StartWrite(q_i);

    q_result->ApplyPostponed(param_, q_node_mut->postponed, *q_point);

    if (visitor.StartVisitingQueryPoint(param_, *q_point, *q_info, *r_node,
          exclusive_unvisited, q_result, &global_result_)) {
      index_t r_count = r_node->count();

      for (index_t r_i_rel = 0; r_i_rel < r_count; ++r_i_rel) {
        const typename GNP::Point *r_point =
            r_points_.StartRead(r_i_rel + r_node->begin());
        typename GNP::RPointInfo *r_info_null = NULL;
        
        visitor.VisitPair(param_, *q_point, *q_info, q_i,
            *r_point, *r_info_null,
            r_i_rel + r_node->begin());
        
        r_points_.StopRead(r_i_rel + r_node->begin());
      }

      visitor.FinishVisitingQueryPoint(param_, *q_point, *q_info, *r_node,
          exclusive_unvisited, q_result, &global_result_);
      
      DEBUG_ONLY(n_naive_ += r_node->count());
    }

    q_node_mut->mass_result.Accumulate(param_, *q_result);

    q_points_.StopRead(q_i);
    q_results_.StopWrite(q_i);
  }

  q_node_mut->mass_result.FinishReaccumulate(param_, *q_node);
  q_node_mut->postponed.Reset(param_);
}

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
//         typename GNP::QMassResult exclusive_unvisited_for_r1(
//             exclusive_unvisited);
//         if (do_r2) {
//           exclusive_unvisited_for_r1.ApplyDelta(param_, delta2);
//         }
//         Pair_(q_node, r_child1, delta1, exclusive_unvisited_for_r1, q_node_mut);
//       }
//       if (do_r2) {
//         Pair_(q_node, r_child2, delta2, exclusive_unvisited, q_node_mut);
//       }
      

#endif
