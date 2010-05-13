/**
 * @file dfs_impl.h
 *
 * Depth-first dual-tree solver template implementations.
 */

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
  param_.InitCopy(param_in);

  q_nodes_.Init(q_nodes, BlockDevice::M_READ);
  r_points_.Init(r_points, BlockDevice::M_READ);
  r_nodes_.Init(r_nodes, BlockDevice::M_READ);

  const typename GNP::QNode *q_root = q_nodes_.StartRead(q_root_index);
  q_results_.Init(q_results, BlockDevice::M_OVERWRITE,
      q_root->begin(), q_root->end());
  q_points_.Init(q_points, BlockDevice::M_READ,
      q_root->begin(), q_root->end());

  // Seed q_results
  {
    CacheWriteIter<typename GNP::QResult> q_results_iter(&q_results_,
        q_root->begin());
    CacheReadIter<typename GNP::QPoint> q_points_iter(&q_points_,
        q_root->begin());
    for (int i = q_root->begin(); i < q_root->end(); ++i,
           q_results_iter.Next(), q_points_iter.Next()) {
      (*q_results_iter).Seed(param_, *q_points_iter);
    }
  }

  q_nodes_.StopRead(q_root_index);

  QMutables default_mutable;
  default_mutable.summary_result.Init(param_);
  default_mutable.postponed.Init(param_);
  q_mutables_.Init(default_mutable, q_root_index, q_end_index);

  // Seed summary_results
  {
    CacheReadIter<typename GNP::QNode> q_nodes_iter(&q_nodes_,
        q_root_index);
    for (int i = q_root_index; i < q_end_index; ++i, q_nodes_iter.Next()) {
      q_mutables_[i].summary_result.Seed(param_, *q_nodes_iter);
    }
  }

  global_result_.Init(param_);

  r_root_ = r_nodes_.StartRead(0);

  do_naive_ = false;

  Begin_(q_root_index);
}

template<typename GNP>
void DualTreeDepthFirst<GNP>::Begin_(index_t q_root_index) {
  typename GNP::Delta delta;
  typename GNP::Delta empty_delta;
  CacheRead<typename GNP::QNode> q_root(&q_nodes_, q_root_index);
  QMutables *q_root_mut = &q_mutables_[q_root_index];

  empty_delta.Init(param_);
  delta.Init(param_);
  bool need_explore = GNP::Algorithm::ConsiderPairIntrinsic(
      param_, *q_root, *r_root_, empty_delta, &delta,
      &global_result_, &q_root_mut->postponed);

  stats_.Init();
  stats_.tuples_analyzed = double(q_root->count()) * r_root_->count();
  stats_.n_queries = q_root->count();

  if (need_explore) {
    typename GNP::QSummaryResult empty_summary_result;

    empty_summary_result.Init(param_);

    if (do_naive_) {
      BaseCase_(q_root, r_root_, delta, empty_summary_result, q_root_mut);
    } else {
      Pair_(q_root, r_root_, delta, empty_summary_result, q_root_mut);
    }
  }

  PushDownPostprocess_(q_root_index, q_root_mut);
}

template<typename GNP>
void DualTreeDepthFirst<GNP>::PushDownPostprocess_(
    index_t q_node_i, QMutables *q_node_mut) {
  CacheRead<typename GNP::QNode> q_node(&q_nodes_, q_node_i);

  if (q_node->is_leaf()) {
    index_t q_i = q_node->begin();
    CacheWriteIter<typename GNP::QResult> q_result(&q_results_, q_i);
    CacheReadIter<typename GNP::QPoint> q_point(&q_points_, q_i);
    
    for (; q_i < q_node->end(); q_i++, q_result.Next(), q_point.Next()) {
      q_result->ApplyPostponed(param_, q_node_mut->postponed, *q_point, q_i);
      q_result->Postprocess(param_, *q_point, q_i, *r_root_);
      global_result_.ApplyResult(param_, *q_point, q_i, *q_result);
    }
  } else {
    for (index_t k = 0; k < 2; k++) {
      index_t q_child_i = q_node->child(k);
      QMutables *q_child_mut = &q_mutables_[q_child_i];

      q_child_mut->postponed.ApplyPostponed(param_, q_node_mut->postponed);

      PushDownPostprocess_(q_child_i, q_child_mut);
    }
  }
}

template<typename GNP>
void DualTreeDepthFirst<GNP>::Pair_(
    const typename GNP::QNode *q_node,
    const typename GNP::RNode *r_node,
    const typename GNP::Delta& delta,
    const typename GNP::QSummaryResult& unvisited,
    QMutables *q_node_mut) {
  //printf("pair(%d:%d) at (%d:%d)\n", q_node->begin(), q_node->count(), r_node->begin(), r_node->count());
  VERBOSE_MSG(1.0, "Checking (%d,%d) x (%d,%d)",
      q_node->begin(), q_node->end(),
      r_node->begin(), r_node->end());
  DEBUG_ONLY(stats_.node_node_considered++);

  /* begin prune checks */
  typename GNP::QSummaryResult mu(q_node_mut->summary_result);
  mu.ApplyPostponed(param_, q_node_mut->postponed, *q_node);
  mu.ApplySummaryResult(param_, unvisited);
  mu.ApplyDelta(param_, delta);

  if (!GNP::Algorithm::ConsiderQueryTermination(
         param_, *q_node, mu, global_result_, &q_node_mut->postponed)) {
    // TODO: This behavior should be re-thinked.
    q_node_mut->summary_result.ApplyDelta(param_, delta);
    VERBOSE_MSG(1.0, "Termination prune");
  } else if (!GNP::Algorithm::ConsiderPairExtrinsic(
          param_, *q_node, *r_node, delta, mu, global_result_,
          &q_node_mut->postponed)) {
    VERBOSE_MSG(1.0, "Extrinsic prune");
  } else {
    if (q_node->is_leaf() && r_node->is_leaf()) {
      VERBOSE_MSG(1.0, "Base case");
      BaseCase_(q_node, r_node, delta, unvisited, q_node_mut);
    } else if (r_node->is_leaf()
        || (q_node->count() >= r_node->count() && !q_node->is_leaf())) {
      VERBOSE_MSG(1.0, "Splitting Q");
      // Phase 2: Explore children, and reincorporate their results.
      q_node_mut->summary_result.StartReaccumulate(param_, *q_node);

      for (index_t k = 0; k < 2; k++) {
        typename GNP::Delta child_delta;
        index_t q_child_i = q_node->child(k);
        CacheRead<typename GNP::QNode> q_child(&q_nodes_, q_child_i);
        QMutables *q_child_mut = &q_mutables_[q_child_i];

        child_delta.Init(param_);
        q_child_mut->postponed.ApplyPostponed(
            param_, q_node_mut->postponed);

        if (GNP::Algorithm::ConsiderPairIntrinsic(
                param_, *q_child, *r_node, delta, &child_delta,
                &global_result_, &q_child_mut->postponed)) {
          Pair_(q_child, r_node, child_delta, unvisited, q_child_mut);
        }

        // We must VERY carefully apply both the horizontal and vertical join
        // operators here for postponed results.
        typename GNP::QSummaryResult tmp_result(q_child_mut->summary_result);
        tmp_result.ApplyPostponed(param_, q_child_mut->postponed, *q_child);
        q_node_mut->summary_result.Accumulate(param_, tmp_result, q_node->count());
      }

      q_node_mut->summary_result.FinishReaccumulate(param_, *q_node);
      q_node_mut->postponed.Reset(param_);
    } else {
      VERBOSE_MSG(1.0, "Splitting R");
      const typename GNP::RNode *r_child1 = r_nodes_.StartRead(r_node->child(0));
      const typename GNP::RNode *r_child2 = r_nodes_.StartRead(r_node->child(1));
      typename GNP::Delta delta1;
      typename GNP::Delta delta2;

      delta1.Init(param_);
      delta2.Init(param_);

      bool explore_r1 = GNP::Algorithm::ConsiderPairIntrinsic(
          param_, *q_node, *r_child1, delta, &delta1,
          &global_result_, &q_node_mut->postponed);
      bool explore_r2 = GNP::Algorithm::ConsiderPairIntrinsic(
          param_, *q_node, *r_child2, delta, &delta2,
          &global_result_, &q_node_mut->postponed);

      if (!explore_r1) {
        if (explore_r2) {
          Pair_(q_node, r_child2, delta2, unvisited, q_node_mut);
        }
      } else if (!explore_r2) {
        Pair_(q_node, r_child1, delta1, unvisited, q_node_mut);
      } else {
        double heur1;
        double heur2;

        heur1 = GNP::Algorithm::Heuristic(param_, *q_node, *r_child1, delta1);
        heur2 = GNP::Algorithm::Heuristic(param_, *q_node, *r_child2, delta2);

        if (!(heur1 > heur2)) {
          typename GNP::QSummaryResult unvisited_for_r1(unvisited);
          unvisited_for_r1.ApplyDelta(param_, delta2);
          Pair_(q_node, r_child1, delta1, unvisited_for_r1, q_node_mut);
          Pair_(q_node, r_child2, delta2, unvisited, q_node_mut);
        } else {
          typename GNP::QSummaryResult unvisited_for_r2(unvisited);
          unvisited_for_r2.ApplyDelta(param_, delta1);
          Pair_(q_node, r_child2, delta2, unvisited_for_r2, q_node_mut);
          Pair_(q_node, r_child1, delta1, unvisited, q_node_mut);
        }
      }

      r_nodes_.StopRead(r_node->child(0));
      r_nodes_.StopRead(r_node->child(1));
    }
  }
}

template<typename GNP>
void DualTreeDepthFirst<GNP>::BaseCase_(
    const typename GNP::QNode *q_node,
    const typename GNP::RNode *r_node,
    const typename GNP::Delta& delta,
    const typename GNP::QSummaryResult& unvisited,
    QMutables *q_node_mut) {

  DEBUG_ONLY(stats_.node_point_considered += q_node->count());

  q_node_mut->summary_result.StartReaccumulate(param_, *q_node);

  typename GNP::PairVisitor visitor;
  visitor.Init(param_);

  CacheRead<typename GNP::QPoint> first_q_point(&q_points_, q_node->begin());
  CacheWrite<typename GNP::QResult> first_q_result(&q_results_, q_node->begin());
  CacheRead<typename GNP::RPoint> first_r_point(&r_points_, r_node->begin());
  size_t q_point_stride = q_points_.n_elem_bytes();
  size_t q_result_stride = q_results_.n_elem_bytes();
  size_t r_point_stride = r_points_.n_elem_bytes();
  index_t q_end = q_node->end();
  const typename GNP::QPoint *q_point = first_q_point;
  typename GNP::QResult *q_result = first_q_result;

  for (index_t q_i = q_node->begin(); q_i < q_end; ++q_i) {
    q_result->ApplyPostponed(param_, q_node_mut->postponed, *q_point, q_i);

    if (visitor.StartVisitingQueryPoint(param_, *q_point, q_i, *r_node,
          delta, unvisited, q_result, &global_result_)) {
      const typename GNP::RPoint *r_point = first_r_point;
      index_t r_i = r_node->begin();
      index_t r_left = r_node->count();

      for (;;) {
        visitor.VisitPair(param_, *q_point, q_i, *r_point, r_i);
        if (unlikely(--r_left == 0)) {
          break;
        }
        r_i++;
        r_point = mem::PtrAddBytes(r_point, r_point_stride);
      }

      visitor.FinishVisitingQueryPoint(param_, *q_point, q_i, *r_node,
          unvisited, q_result, &global_result_);

      DEBUG_ONLY(stats_.point_point_considered += r_node->count());
    }

    q_node_mut->summary_result.Accumulate(param_, *q_result);

    q_point = mem::PtrAddBytes(q_point, q_point_stride);
    q_result = mem::PtrAddBytes(q_result, q_result_stride);
  }

  q_node_mut->summary_result.FinishReaccumulate(param_, *q_node);
  q_node_mut->postponed.Reset(param_);
}

