/**
 * @file single_impl.h
 *
 * Single-tree simulator implementation.
 */

template<typename GNP>
SingleTreeBreadth<GNP>::~SingleTreeBreadth() {
  r_nodes_.StopRead(0);
}

template<typename GNP>
void SingleTreeBreadth<GNP>::Doit(
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

  global_result_.Init(param_);

  r_root_ = r_nodes_.StartRead(0);

  do_naive_ = false;

  Begin_(q_root_index);
}

template<typename GNP>
void SingleTreeBreadth<GNP>::Begin_(index_t q_root_index) {
  CacheRead<typename GNP::QNode> q_root(&q_nodes_, q_root_index);

  stats_.Init();
  stats_.tuples_analyzed = double(q_root->count()) * r_root_->count();
  stats_.n_queries = q_root->count();

#if 0
  typename GNP::Delta empty_delta;
  Queue queue;

  empty_delta.Init(param_);
  queue.Init(param_);
  queue.Consider(param_, *q_root, *r_root_, 0,
      empty_delta, &global_result_);

  Divide_(q_root_index, &queue);
#endif

#if 0
  Divide_(q_root_index);
#endif

  index_t begin = q_root->begin();
  index_t end = q_root->end();
  CacheReadIter<typename GNP::QPoint> points_iter(&q_points_, begin);
  CacheWriteIter<typename GNP::QResult> result_iter(&q_results_, begin);
  for (index_t i = begin; i < end; i++,
	 result_iter.Next(), points_iter.Next()) {
    Queue queue;
    queue.Init(param_);
    QueueItem *item = queue.q.AddBack();
    item->r_index = 0;

    DivideReferences_(*points_iter, i, &(*result_iter), &queue);
    result_iter->Postprocess(param_, *points_iter, i, *r_root_);
    global_result_.ApplyResult(param_, *points_iter, i, *result_iter);
  }
}

template<typename GNP>
void SingleTreeBreadth<GNP>::PushDownPostprocess_(
    const typename GNP::QNode& q_node,
    const typename GNP::QPostponed& postponed) {
  if (q_node.is_leaf()) {
    index_t q_i = q_node.begin();
    CacheWriteIter<typename GNP::QResult> q_result(&q_results_, q_i);
    CacheReadIter<typename GNP::QPoint> q_point(&q_points_, q_i);
    
    for (; q_i < q_node.end(); q_i++, q_result.Next(), q_point.Next()) {
      q_result->ApplyPostponed(param_, postponed, *q_point, q_i);
      q_result->Postprocess(param_, *q_point, q_i, *r_root_);
      global_result_.ApplyResult(param_, *q_point, q_i, *q_result);
    }
  } else {
    for (int k = 0; k < GNP::QNode::CARDINALITY; k++) {
      CacheRead<typename GNP::QNode> q_child(&q_nodes_, q_node.child(k));

      PushDownPostprocess_(*q_child, postponed);
    }
  }
}

template<typename GNP>
bool SingleTreeBreadth<GNP>::BeginExploringQueue_(
    const typename GNP::QNode& q_node, Queue *parent_queue) {
  if (parent_queue->q.size() == 0
      || !GNP::Algorithm::ConsiderQueryTermination(
          param_, q_node, parent_queue->summary_result,
          global_result_, &parent_queue->postponed)) {
    // Distribute mass results to the leaves
    PushDownPostprocess_(q_node, parent_queue->postponed);
    return false;
  } else {
    return true;
  }
}

template<typename GNP>
void SingleTreeBreadth<GNP>::Queue::Init(
    const typename GNP::Param& param) {
  q.Init();
  summary_result.Init(param);
  postponed.Init(param);
}

template<typename GNP>
void SingleTreeBreadth<GNP>::Queue::Consider(
    const typename GNP::Param& param,
    const typename GNP::QNode& q_node, const typename GNP::RNode& r_node,
    index_t r_index, const typename GNP::Delta& parent_delta,
    typename GNP::GlobalResult *global_result) {
  QueueItem *item = q.AddBack();
  item->r_index = r_index;
  item->delta.Init(param);
  if (likely(GNP::Algorithm::ConsiderPairIntrinsic(param, q_node, r_node,
      parent_delta, &item->delta, global_result, &postponed))) {
    summary_result.ApplyDelta(param, item->delta);
  } else {
    q.PopBack();
  }
}

#if 0
template<typename GNP>
void SingleTreeBreadth<GNP>::Queue::Reconsider(
    const typename GNP::Param& param,
    const QueueItem& item) {
  new(q.AddBack())QueueItem(item);
  summary_result.ApplyDelta(param, item.delta);
}
#endif

#if 0
template<typename GNP>
void SingleTreeBreadth<GNP>::Queue::Done(
    const typename GNP::Param& param,
    const typename GNP::QPostponed& parent_postponed,
    const typename GNP::QNode& q_node) {
  postponed.ApplyPostponed(param, parent_postponed);
  //summary_result.ApplyPostponed(param, postponed, q_node);
}
#endif

template<typename GNP>
void SingleTreeBreadth<GNP>::DivideReferences_(
    const typename GNP::QPoint &q_point, index_t q_i,
    typename GNP::QResult *q_result,
    Queue* parent_queue) {
  Queue child_queue;
  child_queue.Init(param_);

  typename GNP::PairVisitor visitor;
  visitor.Init(param_);

  typename GNP::Delta delta;
  delta.Init(param_);

  typename GNP::QSummaryResult unvisited;
  unvisited.Init(param_);

  for (index_t i = parent_queue->q.size(); i--;) {
    const QueueItem *item = &parent_queue->q[i];
    CacheRead<typename GNP::RNode> r_node(&r_nodes_, item->r_index);


    if (visitor.StartVisitingQueryPoint(param_, q_point, q_i, *r_node,
          delta, unvisited, q_result, &global_result_)) {
      if (!r_node->is_leaf()) {
        for (int k_r = 0; k_r < GNP::RNode::CARDINALITY; k_r++) {
          index_t r_child_i = r_node->child(k_r);
          CacheRead<typename GNP::RNode> r_child(&r_nodes_, r_child_i);

	  child_queue.q.AddBack()->r_index = r_child_i;

	  DRange d_density_pos;
	  d_density_pos.Init(0, 0);
	  if (r_child->stat().count_pos > 0) {
	    DRange dist_sq =
	      r_node->stat().bound_pos.RangeDistanceSq(q_point.vec());
	    d_density_pos.hi = param_.kernel_pos.EvalUnnormOnSq(dist_sq.lo);
	    d_density_pos.lo = param_.kernel_pos.EvalUnnormOnSq(dist_sq.hi);
	  }

	  DRange d_density_neg;
	  d_density_neg.Init(0, 0);
	  if (r_child->stat().count_neg > 0) {
	    DRange dist_sq =
	      r_node->stat().bound_neg.RangeDistanceSq(q_point.vec());
	    d_density_neg.hi = param_.kernel_neg.EvalUnnormOnSq(dist_sq.lo);
	    d_density_neg.lo = param_.kernel_neg.EvalUnnormOnSq(dist_sq.hi);
	  }

	  d_density_pos *= r_child->stat().count_pos;
	  d_density_neg *= r_child->stat().count_neg;
	  delta.d_density_pos = d_density_pos;
	  delta.d_density_neg = d_density_neg;

	  unvisited.ApplyDelta(param_, delta);
        }
      } else {
	CacheRead<typename GNP::RPoint>
	  first_r_point(&r_points_, r_node->begin());
	size_t r_point_stride = r_points_.n_elem_bytes();
	const typename GNP::RPoint *r_point = first_r_point;
	index_t r_i = r_node->begin();
	index_t r_left = r_node->count();

	for (;;) {
	  visitor.VisitPair(param_, q_point, q_i, *r_point, r_i);
	  if (unlikely(--r_left == 0)) {
	    break;
	  }
	  r_i++;
	  r_point = mem::PtrAddBytes(r_point, r_point_stride);
	}
      }
    }
  }

  const QueueItem *item = &parent_queue->q[0];
  CacheRead<typename GNP::RNode> r_node(&r_nodes_, item->r_index);
  visitor.FinishVisitingQueryPoint(param_, q_point, q_i, *r_node,
      unvisited, q_result, &global_result_);

  if (child_queue.q.size() > 0 && q_result->label == 3) {
    DivideReferences_(q_point, q_i, q_result, &child_queue);
  }
}

#if 0
template<typename GNP>
void SingleTreeBreadth<GNP>::DivideReferences_(
    index_t q_node_i, Queue* parent_queue) {
  const typename GNP::QNode q_node(*q_nodes_.StartRead(q_node_i));
  q_nodes_.StopRead(q_node_i);

  if (q_node.is_leaf()) {
    // Make sure we take into account results at the leaves in our
    // summary result.
    // While we're at it, let's also take care of postponed results.
    typename GNP::QSummaryResult mu;
    mu.Init(param_);
    mu.StartReaccumulate(param_, q_node);
    for (index_t q_i = q_node.begin(); q_i < q_node.end(); q_i++) {
      CacheRead<typename GNP::QPoint> q_point(&q_points_, q_i);
      CacheWrite<typename GNP::QResult> q_result(&q_results_, q_i);
      q_result->ApplyPostponed(param_, parent_queue->postponed, *q_point, q_i);
      mu.Accumulate(param_, *q_result);
    }
    mu.FinishReaccumulate(param_, q_node);
    // Divide_ will apply the postponed results to the summary result.
    // Instead, we apply the points' results to the summary result.
    parent_queue->summary_result.ApplySummaryResult(param_, mu);
    parent_queue->postponed.Reset(param_);
  } else {
    FATAL("Unwritten code -- someone can write it");
  }

  if (!BeginExploringQueue_(q_node, parent_queue)) {
    return;
  }

  Queue child_queue;
  child_queue.Init(param_);

  ArrayList<typename GNP::QSummaryResult> summaries;

  summaries.Init(parent_queue->q.size());
  summaries[0].Init(param_);
  // note: parent's postponed is empty
  //summaries[0].ApplyPostponed(parent.postponed);

  for (index_t i = 1; i < parent_queue->q.size(); i++) {
    summaries[i].Copy(summaries[i-1]);
    summaries[i].ApplyDelta(param_, parent_queue->q[i-1].delta);
  }

  DEBUG_ONLY(stats_.node_node_considered += parent_queue->q.size());

  for (index_t i = parent_queue->q.size(); i--;) {
    const QueueItem *item = &parent_queue->q[i];
    CacheRead<typename GNP::RNode> r_node(&r_nodes_, item->r_index);

    if (likely(GNP::Algorithm::ConsiderPairExtrinsic(
        param_, q_node, *r_node, item->delta, parent_queue->summary_result,
        global_result_, &child_queue.postponed))) {
      if (!r_node->is_leaf()) {
        for (int k_r = 0; k_r < GNP::RNode::CARDINALITY; k_r++) {
          index_t r_child_i = r_node->child(k_r);
          CacheRead<typename GNP::RNode> r_child(&r_nodes_, r_child_i);

          child_queue.Consider(param_, q_node, *r_child, r_child_i,
              parent_queue->q[i].delta, &global_result_);
        }
      } else {
        // Only for leaf computations will we go through the trouble of
        // recomputing summary results.

        // Here, we compute summary results for all computations on the
        // queue EXCEPT this one.  We'll start with the left-to-right from
        // the previous level, add in the right-to-left result from the
        // children, add in the postponed prunes we made on this level.
        summaries[i].ApplySummaryResult(param_, child_queue.summary_result);
        summaries[i].ApplyPostponed(param_, child_queue.postponed, q_node);
        // TODO: Instead of applying postponed at the node level, we can
        // eagerly forward it to the points themselves.
        BaseCase_(q_node, *r_node, parent_queue->q[i].delta, summaries[i]);
      }
    }
  }

  // parent_queue->postponed is *still* empty
  ////child_queue.Done(param_, parent_queue->postponed, q_node);
  //child_queue.postponed.ApplyPostponed(param_, parent_queue->postponed);
  DivideReferences_(q_node_i, &child_queue);
}
#endif

#if 0
template<typename GNP>
void SingleTreeBreadth<GNP>::Divide_(index_t q_node_i) {
  const typename GNP::QNode q_node(*q_nodes_.StartRead(q_node_i));
  q_nodes_.StopRead(q_node_i);

  if (q_node.is_leaf()) {
    Queue queue;
    typename GNP::Delta empty_delta;

    queue.Init(param_);
    empty_delta.Init(param_);
    queue.Consider(param_, q_node, *r_root_, 0,
		   empty_delta, &global_result_);

    DivideReferences_(q_node_i, &queue);
  } else {
    for (int k = 0; k < GNP::QNode::CARDINALITY; k++) {
      Divide_(q_node.child(k));
    }
  }
}
#endif

#if 0
template<typename GNP>
void SingleTreeBreadth<GNP>::Divide_(
    index_t q_node_i, Queue* parent_queue) {
  const typename GNP::QNode q_node(*q_nodes_.StartRead(q_node_i));
  q_nodes_.StopRead(q_node_i);

  if (q_node.is_leaf()) {
    DivideReferences_(q_node_i, parent_queue);
    return;
  }

  if (!BeginExploringQueue_(q_node, parent_queue)) {
    return;
  }

  Queue child_queues[GNP::QNode::CARDINALITY];
  const typename GNP::QNode *q_children[GNP::QNode::CARDINALITY];

  parent_queue->summary_result.Seed(param_, q_node);
  parent_queue->summary_result.ApplyPostponed(param_, parent_queue->postponed, q_node);

  for (int k = 0; k < GNP::QNode::CARDINALITY; k++) {
    q_children[k] = q_nodes_.StartRead(q_node.child(k));
    child_queues[k].Init(param_);
  }

  DEBUG_ONLY(stats_.node_node_considered += parent_queue->q.size());

  for (index_t i = 0; i < parent_queue->q.size(); i++) {
    const QueueItem *item = &parent_queue->q[i];
    CacheRead<typename GNP::RNode> r_node(&r_nodes_, item->r_index);

    if (likely(GNP::Algorithm::ConsiderPairExtrinsic(
        param_, q_node, *r_node, item->delta, parent_queue->summary_result,
        global_result_, &parent_queue->postponed))) {
      if (likely(!r_node->is_leaf()) && likely(r_node->count() > 2 * q_node.count())) {
        // Only divide reference node if it is more than twice the size of the query.

        for (int k_r = 0; k_r < GNP::RNode::CARDINALITY; k_r++) {
          index_t r_child_i = r_node->child(k_r);
          CacheRead<typename GNP::RNode> r_child(&r_nodes_, r_child_i);

          for (int k_q = 0; k_q < GNP::QNode::CARDINALITY; k_q++) {
            // Loop for both query children

            if (unlikely(r_child->count() > q_children[k_q]->count()) && likely(!r_child->is_leaf())) {
              // Divide reference set an extra time if the reference node is large.
              for (int k_r2 = 0; k_r2 < GNP::RNode::CARDINALITY; k_r2++) {
                index_t r_child2_i = r_child->child(k_r2);
                CacheRead<typename GNP::RNode> r_child2(&r_nodes_, r_child2_i);
                child_queues[k_q].Consider(param_, *q_children[k_q], *r_child2,
                    r_child2_i, parent_queue->q[i].delta, &global_result_);
              }
            } else {
              child_queues[k_q].Consider(param_, *q_children[k_q], *r_child,
                  r_child_i, parent_queue->q[i].delta, &global_result_);
            }
          }
        }
      } else {
        for (int k_q = 0; k_q < GNP::QNode::CARDINALITY; k_q++) {
          child_queues[k_q].Consider(param_, *q_children[k_q], *r_node,
              item->r_index, parent_queue->q[i].delta, &global_result_);
          //child_queues[k_q].Reconsider(param_, *item);
        }
      }
    }
  }

  // Release the locks on the children to ease cache pressure in the FIFO
  for (int k = 0; k < GNP::QNode::CARDINALITY; k++) {
    //child_queues[k].Done(param_, parent_queue->postponed, *q_children[k]);
    child_queues[k].postponed.ApplyPostponed(param_, parent_queue->postponed);
    q_nodes_.StopRead(q_node.child(k));
  }

  for (int k = 0; k < GNP::QNode::CARDINALITY; k++) {
    Divide_(q_node.child(k), &child_queues[k]);
  }
}
#endif

template<typename GNP>
void SingleTreeBreadth<GNP>::BaseCase_(
    const typename GNP::QNode& q_node,
    const typename GNP::RNode& r_node,
    const typename GNP::Delta& delta,
    const typename GNP::QSummaryResult& unvisited) {
  DEBUG_ONLY(stats_.node_point_considered += q_node.count());

  typename GNP::PairVisitor visitor;
  visitor.Init(param_);

  CacheRead<typename GNP::QPoint> first_q_point(&q_points_, q_node.begin());
  CacheWrite<typename GNP::QResult> first_q_result(&q_results_, q_node.begin());
  CacheRead<typename GNP::RPoint> first_r_point(&r_points_, r_node.begin());
  size_t q_point_stride = q_points_.n_elem_bytes();
  size_t q_result_stride = q_results_.n_elem_bytes();
  size_t r_point_stride = r_points_.n_elem_bytes();
  index_t q_end = q_node.end();
  const typename GNP::QPoint *q_point = first_q_point;
  typename GNP::QResult *q_result = first_q_result;

  for (index_t q_i = q_node.begin(); q_i < q_end; ++q_i) {
    if (visitor.StartVisitingQueryPoint(param_, *q_point, q_i, r_node,
          delta, unvisited, q_result, &global_result_)) {
      const typename GNP::RPoint *r_point = first_r_point;
      index_t r_i = r_node.begin();
      index_t r_left = r_node.count();

      for (;;) {
        visitor.VisitPair(param_, *q_point, q_i, *r_point, r_i);
        if (unlikely(--r_left == 0)) {
          break;
        }
        r_i++;
        r_point = mem::PtrAddBytes(r_point, r_point_stride);
      }

      visitor.FinishVisitingQueryPoint(param_, *q_point, q_i, r_node,
          unvisited, q_result, &global_result_);

      DEBUG_ONLY(stats_.point_point_considered += r_node.count());
    }

    q_point = mem::PtrAddBytes(q_point, q_point_stride);
    q_result = mem::PtrAddBytes(q_result, q_result_stride);
  }
}
