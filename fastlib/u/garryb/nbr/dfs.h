#ifndef NBR_DFS_H
#define NBR_DFS_H

#include "gnp.h"

template<typename GNP>
class DualTreeDepthFirst {
 private:
  struct QMutableInfo {
    typename GNP::QMassResult mass_result;
    typename GNP::QPostponed postponed;
  };
  
 private:
  KdTreeMidpointBuilder<typename GNP::QPointInfo, typename GNP::QNode,
      typename GNP::Param> q_tree_;
  KdTreeMidpointBuilder<typename GNP::RPointInfo, typename GNP::RNode,
      typename GNP::Param> r_tree_;
  typename GNP::Param param_;
  typename GNP::GlobalResult global_result_;
  ArrayList<typename GNP::QResult> q_results_;
  ArrayList<QMutableInfo> q_mutables_;
  bool do_naive_;
  datanode *datanode_;
  uint64 n_naive_;
  uint64 n_pre_naive_;
  uint64 n_recurse_;
  
 public:
  void Init(datanode *datanode);
  void Begin();

 private:
  void Pair_(index_t q_node_i, index_t r_node_i,
      const typename GNP::Delta& delta,
      const typename GNP::QMassResult& exclusive_unvisited);
  void BaseCase_(
      typename GNP::QNode *q_node,
      const typename GNP::RNode *r_node,
      const typename GNP::QMassResult& exclusive_unvisited,
      QMutableInfo *q_node_mut);
  void PushDown_(index_t q_node_i);
 private:
  typename GNP::QNode *qnode_(index_t i) {
    return &q_tree_.nodes()[i];
  }
  const typename GNP::RNode *rnode_(index_t i) {
    return &r_tree_.nodes()[i];
  }
};

template<typename GNP>
void DualTreeDepthFirst<GNP>::Init(datanode *datanode) {
  datanode_ = datanode;
  
  do_naive_ = fx_param_bool(datanode, "do_naive", 0);
  
  fx_timer_start(datanode_, "q_matrix");
  Matrix q_matrix;
  ASSERT_PASS(data::Load(fx_param_str_req(datanode, "q"), &q_matrix));
  fx_timer_stop(datanode_, "q_matrix");
  
  fx_timer_start(datanode_, "r_matrix");
  Matrix r_matrix;
  ASSERT_PASS(data::Load(fx_param_str_req(datanode, "r"), &r_matrix));
  fx_timer_stop(datanode_, "r_matrix");
  
  param_.Init(fx_submodule(datanode, "param", "param"),
      q_matrix, r_matrix);
  
  ArrayList<typename GNP::QPointInfo> q_point_info;
  q_point_info.Init(q_matrix.n_cols());
  // TODO: Read info?
  
  ArrayList<typename GNP::RPointInfo> r_point_info;
  r_point_info.Init(r_matrix.n_cols());
  // TODO: Read info?
  
  fx_timer_start(datanode_, "q_tree");
  q_tree_.Init(&param_, q_matrix, q_point_info);
  q_tree_.Build();
  fx_timer_stop(datanode_, "q_tree");
  
  fx_timer_start(datanode_, "r_tree");
  r_tree_.Init(&param_, r_matrix, r_point_info);
  r_tree_.Build();
  fx_timer_stop(datanode_, "r_tree");
  
  q_results_.Init(q_tree_.points().size());
  for (index_t i = 0; i < q_tree_.points().size(); i++) {
    q_results_[i].Init(param_, q_tree_.points()[i], q_tree_.point_info()[i],
        r_tree_.nodes()[0]);
  }
  
  q_mutables_.Init(q_tree_.nodes().size());
  for (index_t i = 0; i < q_tree_.nodes().size(); i++) {
    q_mutables_[i].mass_result.Init(param_);
    q_mutables_[i].postponed.Init(param_);
  }
  
  global_result_.Init(param_);
  
}

template<typename GNP>
void DualTreeDepthFirst<GNP>::Begin() {
  typename GNP::Delta delta;
  typename GNP::QNode *q_root = qnode_(0);
  QMutableInfo *q_root_mut = &q_mutables_[0];
  const typename GNP::RNode *r_root = rnode_(0);

  fx_timer_start(datanode_, "execute");
  
  DEBUG_ONLY(n_naive_ = 0);
  DEBUG_ONLY(n_pre_naive_ = 0);
  DEBUG_ONLY(n_recurse_ = 0);
  
  bool need_explore = GNP::Algorithm::ConsiderPairIntrinsic(
      param_, *q_root, *r_root, &delta,
      &global_result_, &q_root_mut->postponed);

  
  if (need_explore) {
    typename GNP::QMassResult empty_mass_result;

    empty_mass_result.Init(param_);

    if (do_naive_) {
      BaseCase_(qnode_(0), rnode_(0), empty_mass_result, &q_mutables_[0]);
    } else {
      Pair_(0, 0, delta, empty_mass_result);
      PushDown_(0);
    }
  }
  fx_timer_stop(datanode_, "execute");
  
  DEBUG_ONLY(fx_format_result(datanode_, "naive_ratio", "%f",
      1.0 * n_naive_ / q_root->count() / r_root->count()));
  DEBUG_ONLY(fx_format_result(datanode_, "naive_per_query", "%f",
      1.0 * n_naive_ / q_root->count()));
  DEBUG_ONLY(fx_format_result(datanode_, "pre_naive_ratio", "%f",
      1.0 * n_pre_naive_ / q_root->count() / r_root->count()));
  DEBUG_ONLY(fx_format_result(datanode_, "pre_naive_per_query", "%f",
      1.0 * n_pre_naive_ / q_root->count()));
  DEBUG_ONLY(fx_format_result(datanode_, "recurse_ratio", "%f",
      1.0 * n_recurse_ / q_root->count() / r_root->count()));
  DEBUG_ONLY(fx_format_result(datanode_, "recurse_per_query", "%f",
      1.0 * n_recurse_ / q_root->count()));
  
  if (fx_param_bool(datanode_, "print", 0)) {
    ot::Print(q_results_);
  }
}

template<typename GNP>
void DualTreeDepthFirst<GNP>::PushDown_(index_t q_node_i) {
  typename GNP::QNode *q_node = qnode_(q_node_i);
  QMutableInfo *q_node_mut = &q_mutables_[q_node_i];
  
  if (q_node->is_leaf()) {
    for (index_t q_i = q_node->begin(); q_i < q_node->end(); q_i++) {
      typename GNP::QResult *q_result = &q_results_[q_i];
      typename GNP::Point *q_point = &q_tree_.points()[q_i];
      typename GNP::QPointInfo *q_info = &q_tree_.point_info()[q_i];
      q_result->ApplyPostponed(param_, q_node_mut->postponed, *q_point);
      q_result->Postprocess(param_, *q_point, *q_info, r_tree_.nodes()[0]);
    }
  } else {
    for (index_t k = 0; k < 2; k++) {
      index_t q_child_i = q_node->child(k);
      QMutableInfo *q_child_mut = &q_mutables_[q_child_i];
      
      q_child_mut->postponed.ApplyPostponed(param_, q_node_mut->postponed);
      PushDown_(q_child_i);
    }
  }
}

template<typename GNP>
void DualTreeDepthFirst<GNP>::Pair_(index_t q_node_i, index_t r_node_i,
    const typename GNP::Delta& delta,
    const typename GNP::QMassResult& exclusive_unvisited) {
  typename GNP::QNode *q_node = qnode_(q_node_i);
  const typename GNP::RNode *r_node = rnode_(r_node_i);
  QMutableInfo *q_node_mut = &q_mutables_[q_node_i];

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
    return;
  }
  
  if (!GNP::Algorithm::ConsiderPairExtrinsic(
          param_, *q_node, *r_node, delta, mu, global_result_,
          &q_node_mut->postponed)) {
    DEBUG_MSG(1.0, "Extrinsic prune");
    return;
  }
  /* end prune checks */

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
      typename GNP::QNode *q_child = qnode_(q_child_i);
      QMutableInfo *q_child_mut = &q_mutables_[q_child_i];

      q_child_mut->postponed.ApplyPostponed(
          param_, q_node_mut->postponed);
      child_delta.Init(param_);

      if (GNP::Algorithm::ConsiderPairIntrinsic(
              param_, *q_child, *r_node, &child_delta,
              &global_result_, &q_child_mut->postponed)) {
        Pair_(q_child_i, r_node_i, delta, exclusive_unvisited);
      }

      // We must VERY carefully apply both the horizontal and vertical join
      // operators here for postponed results.
      typename GNP::QMassResult tmp_result(q_child_mut->mass_result);
      tmp_result.ApplyPostponed(param_, q_child_mut->postponed, *q_child);
      q_node_mut->mass_result.Accumulate(param_, tmp_result, q_node->count());
    }

    q_node_mut->mass_result.FinishReaccumulate(param_, *q_node);
    q_node_mut->postponed.Reset(param_);
  } else {
    DEBUG_MSG(1.0, "Splitting R");
    index_t r_child1_i = r_node->child(0);
    index_t r_child2_i = r_node->child(1);
    const typename GNP::RNode *r_child1 = rnode_(r_child1_i);
    const typename GNP::RNode *r_child2 = rnode_(r_child2_i);
    
    double r_child1_h = GNP::Algorithm::Heuristic(
        param_, *q_node, *rnode_(r_child1_i));
    double r_child2_h = GNP::Algorithm::Heuristic(
        param_, *q_node, *rnode_(r_child2_i));

    if (unlikely(r_child2_h < r_child1_h)) {
      const typename GNP::RNode *r_child_t = r_child1;
      r_child1 = r_child2;
      r_child2 = r_child_t;
      
      index_t r_child_t_i = r_child1_i;
      r_child1_i = r_child2_i;
      r_child2_i = r_child_t_i;
    }

    typename GNP::Delta delta1;
    typename GNP::Delta delta2;

    delta1.Init(param_);
    delta2.Init(param_);

    bool do_r2 = GNP::Algorithm::ConsiderPairIntrinsic(
        param_, *q_node, *r_child2, &delta2,
        &global_result_, &q_node_mut->postponed);
    
    if (GNP::Algorithm::ConsiderPairIntrinsic(
        param_, *q_node, *r_child1, &delta1,
        &global_result_, &q_node_mut->postponed)) {
      typename GNP::QMassResult exclusive_unvisited_for_r1(
          exclusive_unvisited);
      if (do_r2) {
        exclusive_unvisited_for_r1.ApplyDelta(param_, delta2);
      }
      Pair_(q_node_i, r_child1_i, delta1, exclusive_unvisited_for_r1);
    }
    if (do_r2) {
      Pair_(q_node_i, r_child2_i, delta2, exclusive_unvisited);
    }
  }
}

template<typename GNP>
void DualTreeDepthFirst<GNP>::BaseCase_(
    typename GNP::QNode *q_node,
    const typename GNP::RNode *r_node,
    const typename GNP::QMassResult& exclusive_unvisited,
    QMutableInfo *q_node_mut) {
  typename GNP::PairVisitor visitor;

  DEBUG_ONLY(n_pre_naive_ += q_node->count() * r_node->count());
  
  visitor.Init(param_);

  q_node_mut->mass_result.StartReaccumulate(param_, *q_node);
  DEBUG_ASSERT_MSG(q_node->count() != 0, "index %d, count = %d, begin = %d",
      int(q_node - qnode_(0)), q_node->begin(), q_node->count());

  for (index_t q_i = q_node->begin(); q_i < q_node->end(); ++q_i) {
    typename GNP::Point *q_point = &q_tree_.points()[q_i];
    typename GNP::QPointInfo *q_info = &q_tree_.point_info()[q_i];
    typename GNP::QResult *q_result = &q_results_[q_i];

    q_result->ApplyPostponed(param_, q_node_mut->postponed, *q_point);

    if (visitor.StartVisitingQueryPoint(param_, *q_point, *q_info, *r_node,
          exclusive_unvisited, q_result, &global_result_)) {
      index_t r_end = r_node->end();

      for (index_t r_i = r_node->begin(); r_i < r_end; ++r_i) {
        typename GNP::Point *r_point = &r_tree_.points()[r_i];
        typename GNP::RPointInfo *r_info = &r_tree_.point_info()[r_i];

        visitor.VisitPair(param_, *q_point, *q_info, q_i,
            *r_point, *r_info, r_i);
      }

      visitor.FinishVisitingQueryPoint(param_, *q_point, *q_info, *r_node,
          exclusive_unvisited, q_result, &global_result_);
      
      DEBUG_ONLY(n_naive_ += r_node->count());
    }

    q_node_mut->mass_result.Accumulate(param_, *q_result);
  }

  q_node_mut->mass_result.FinishReaccumulate(param_, *q_node);
  q_node_mut->postponed.Reset(param_);
}

#endif

