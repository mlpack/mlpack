#ifndef GNP_H
#define GNP_H

#include "kdtree.h"
#include "spnode.h"

#include "fastlib/fastlib.h"

template<
  typename TParam, typename TAlgorithm,
  typename TPoint, typename TBound,
  typename TQPointInfo, typename TQStat,
  typename TRPointInfo, typename TRStat,
  typename TPairVisitor, typename TDelta,
  typename TQResult, typename TQMassResult, typename TQPostponed,
  typename TGlobalResult>
class DualTreeGNP {
 public:
  typedef TParam Param;
  typedef TAlgorithm Algorithm;

  typedef TPoint Point;
  typedef TBound Bound;

  typedef TQPointInfo QPointInfo;
  typedef TQStat QStat;

  typedef TRPointInfo RPointInfo;
  typedef TRStat RStat;

  typedef TPairVisitor PairVisitor;
  typedef TDelta Delta;

  typedef TQResult QResult;
  typedef TQMassResult QMassResult;
  typedef TQPostponed QPostponed;

  typedef TGlobalResult GlobalResult;

  typedef SpNode<Bound, QStat> QNode;
  typedef SpNode<Bound, RStat> RNode;
};

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
  do_naive_ = fx_param_bool(datanode, "do_naive", 0);
  
  Matrix q_matrix;
  data::Load(fx_param_str_req(datanode, "q"), &q_matrix);
  
  Matrix r_matrix;
  data::Load(fx_param_str_req(datanode, "r"), &r_matrix);
  
  param_.Init(fx_submodule(datanode, "param", "param"),
      q_matrix, r_matrix);
  
  ArrayList<typename GNP::QPointInfo> q_point_info;
  q_point_info.Init(q_matrix.n_cols());
  // TODO: Read info?
  
  ArrayList<typename GNP::RPointInfo> r_point_info;
  r_point_info.Init(r_matrix.n_cols());
  // TODO: Read info?
  
  q_tree_.Init(&param_, q_matrix, q_point_info);
  q_tree_.Build();
  
  r_tree_.Init(&param_, r_matrix, r_point_info);
  r_tree_.Build();
  
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
  
  datanode_ = datanode;
}

template<typename GNP>
void DualTreeDepthFirst<GNP>::Begin() {
  typename GNP::Delta delta;
  typename GNP::QNode *q_root = qnode_(0);
  QMutableInfo *q_root_mut = &q_mutables_[0];
  const typename GNP::RNode *r_root = rnode_(0);
  bool need_explore = GNP::Algorithm::ConsiderPairIntrinsic(
      param_, *q_root, *r_root, &delta,
      &global_result_, &q_root_mut->postponed);

  fx_timer_start(datanode_, "execute");
  
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
  
  //ot::Print(q_results_);
}

template<typename GNP>
void DualTreeDepthFirst<GNP>::PushDown_(index_t q_node_i) {
  typename GNP::QNode *q_node = qnode_(q_node_i);
  QMutableInfo *q_node_mut = &q_mutables_[q_node_i];
  
  if (q_node->is_leaf()) {
    for (index_t q_i = q_node->begin(); q_i < q_node->end(); q_i++) {
      typename GNP::QResult *q_result = &q_results_[q_i];
      typename GNP::Point *q_point = &q_tree_.points()[q_i];
      q_result->ApplyPostponed(param_, q_node_mut->postponed, *q_point);
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

  /* begin prune checks */
  typename GNP::QMassResult mu(q_node_mut->mass_result);
  mu.ApplyPostponed(param_, q_node_mut->postponed, *q_node);
  mu.ApplyMassResult(param_, exclusive_unvisited);
  mu.ApplyDelta(param_, delta);

  if (!GNP::Algorithm::ConsiderQueryTermination(
      param_, *q_node, mu, global_result_, &q_node_mut->postponed)) {
    return;
  }

  if (!GNP::Algorithm::ConsiderPairExtrinsic(
      param_, *q_node, *r_node, delta, mu, global_result_,
      &q_node_mut->postponed)) {
    return;
  }
  /* end prune checks */

  global_result_.UndoDelta(param_, delta);

  if (q_node->is_leaf() && r_node->is_leaf()) {
    BaseCase_(q_node, r_node, exclusive_unvisited, q_node_mut);
  } else if (r_node->is_leaf()
      || (q_node->count() >= r_node->count() && !q_node->is_leaf())) {
    typename GNP::Delta sub_deltas[2];
    bool do_child[2];

    // Phase 1: Do intrinsic checking, calculate deltas,
    // and update gamma to a valid state.
    for (index_t k = 0; k < 2; k++) {
      index_t q_child_i = q_node->child(k);
      typename GNP::QNode *q_child = qnode_(q_child_i);
      QMutableInfo *q_child_mut = &q_mutables_[q_child_i];

      q_child_mut->postponed.ApplyPostponed(
          param_, q_node_mut->postponed);
      do_child[k] = GNP::Algorithm::ConsiderPairIntrinsic(
          param_, *q_child, *r_node, &sub_deltas[k],
          &global_result_, &q_child_mut->postponed);
    }

    // Phase 2: Explore children, and reincorporate their results.
    q_node_mut->mass_result.StartReaccumulate(param_, *q_node);

    for (index_t k = 0; k < 2; k++) {
      index_t q_child_i = q_node->child(k);
      typename GNP::QNode *q_child = qnode_(q_child_i);

      if (likely(do_child[k])) {
        Pair_(q_child_i, r_node_i, sub_deltas[k], exclusive_unvisited);
      }

      // We must VERY carefully apply both the horizontal and vertical join
      // operators here for postponed results.
      const QMutableInfo *q_child_mut = &q_mutables_[q_child_i];
      typename GNP::QMassResult tmp_result(q_child_mut->mass_result);
      tmp_result.ApplyPostponed(param_, q_child_mut->postponed, *q_child);
      q_node_mut->mass_result.Accumulate(param_, tmp_result, q_node->count());
    }

    q_node_mut->mass_result.FinishReaccumulate(param_, *q_node);
    q_node_mut->postponed.Reset(param_);
  } else {
    index_t r_child1_i = r_node->child(0);
    index_t r_child2_i = r_node->child(1);
    double r_child1_h = GNP::Algorithm::Heuristic(
        param_, *q_node, *rnode_(r_child1_i));
    double r_child2_h = GNP::Algorithm::Heuristic(
        param_, *q_node, *rnode_(r_child2_i));

    if (unlikely(r_child2_h < r_child1_h)) {
      r_child1_i = r_node->child(1);
      r_child2_i = r_node->child(0);
    }

    const typename GNP::RNode *r_child1;
    const typename GNP::RNode *r_child2;
    r_child1 = rnode_(r_child1_i);
    r_child2 = rnode_(r_child2_i);
    //fprintf(stderr, "Ordering: (%d,%d) chooses (%d,%d)\n",
    //    q_node->begin(), q_node->end()-1,
    //    r_child1->begin(), r_child1->end()-1);

    typename GNP::Delta delta1;
    typename GNP::Delta delta2;

    delta1.Init(param_);
    delta2.Init(param_);

    if (!GNP::Algorithm::ConsiderPairIntrinsic(
        param_, *q_node, *r_child2, &delta2,
        &global_result_, &q_mutables_[q_node_i].postponed)) {
      r_child2 = NULL;
    }
    if (GNP::Algorithm::ConsiderPairIntrinsic(
        param_, *q_node, *r_child1, &delta1,
        &global_result_, &q_mutables_[q_node_i].postponed)) {
      typename GNP::QMassResult exclusive_unvisited_for_r1(
          exclusive_unvisited);
      exclusive_unvisited_for_r1.ApplyDelta(param_, delta2);
      Pair_(q_node_i, r_child1_i, delta1, exclusive_unvisited_for_r1);
    }
    if (r_child2 != NULL) {
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

  visitor.Init(param_);

  q_node_mut->mass_result.StartReaccumulate(param_, *q_node);

  for (index_t q_i = q_node->begin(); q_i < q_node->end(); ++q_i) {
    typename GNP::Point *q_point = &q_tree_.points()[q_i];
    typename GNP::QPointInfo *q_info = &q_tree_.point_info()[q_i];
    typename GNP::QResult *q_result = &q_results_[q_i];

    q_result->ApplyPostponed(param_, q_node_mut->postponed, *q_point);
    
    if (visitor.StartVisitingQueryPoint(param_, *q_point, *q_info, *r_node,
          exclusive_unvisited, q_result, &global_result_)) {
      for (index_t r_i = r_node->begin(); r_i < r_node->end(); ++r_i) {
        typename GNP::Point *r_point = &r_tree_.points()[r_i];
        typename GNP::RPointInfo *r_info = &r_tree_.point_info()[r_i];

        visitor.VisitPair(param_, *q_point, *q_info, q_i,
            *r_point, *r_info, r_i);
      }

      visitor.FinishVisitingQueryPoint(param_, *q_point, *q_info, *r_node,
          exclusive_unvisited, q_result, &global_result_);
    }

    q_node_mut->mass_result.Accumulate(param_, *q_result);
  }
  q_node_mut->mass_result.FinishReaccumulate(param_, *q_node);
}


#endif
