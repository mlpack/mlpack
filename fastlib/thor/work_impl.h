/* Templated Implementations for GNP scheduling. */

template<typename Node>
void CentroidWorkQueue<Node>::Init(CacheArray<Node> *tree,
    const DecompNode *decomp_root, int n_threads, datanode *module) {
  granularity_ = fx_param_double(module, "granularity", 12);
  no_overflow_ = fx_param_bool(module, "no_overflow", false);
  tree_ = tree;
  n_threads_ = n_threads;
  root_ = new InternalNode(NONE, tree,
      decomp_root->index(), decomp_root->end_index());
  rankes_.Init(rpc::n_peers());
  DEBUG_ASSERT_MSG(decomp_root->info().begin_rank == 0
      && decomp_root->info().end_rank == rpc::n_peers(),
      "Can't handle incomplete decompositions (yet)");
  n_grains_ = 0;
  n_overflows_ = 0;
  n_overflow_points_ = 0;
  n_assigned_points_ = 0;
  n_preferred_ = 0;

  DistributeInitialWork_(decomp_root, root_);

  // Ensure we don't access the tree after we start the algorithm.
  tree_ = NULL;
}

template<typename Node>
void CentroidWorkQueue<Node>::DistributeInitialWork_(
    const DecompNode *decomp_node, InternalNode *node) {
  int begin_rank = decomp_node->info().begin_rank;
  int end_rank = decomp_node->info().end_rank;
  int n_ranks = end_rank - begin_rank;

  if (n_ranks == 1 || node->is_leaf()) {
    // Prime each rankor's centroid with the centroid of this block.
    // Note there will probably only be 1 rankor unless the pathological
    // case where we're trying to subdivide a leaf between rankors.
    // If we're subdividing a leaf between rankors, then some rankors
    // won't have any work to do...
    Vector center;
    node->node().bound().CalculateMidpoint(&center);
    index_t max_grain_size = math::RoundInt(
        node->count() / granularity_ / n_threads_);

    for (int i = begin_rank; i < end_rank; i++) {
      ProcessWorkQueue *queue = &rankes_[i];
      queue->max_grain_size = max_grain_size;
      queue->n_centers = 1;
      queue->sum_centers.Copy(center);
      queue->work_items.Init();
    }

    ProcessWorkQueue *queue = &rankes_[begin_rank];
    ArrayList<InternalNode*> node_stack;
    node_stack.Init();
    *node_stack.AddBack() = node;

    // Subdivide the node further if possible.
    while (node_stack.size() != 0) {
      InternalNode *cur = *node_stack.PopBackPtr();
      //double distance = cur->node().bound().MinDistanceSq(center);

      if (cur->is_leaf() || cur->count() <= max_grain_size) {
        // Put in work-queue in tree order.  This helps ensure that both
        // threads are working on relatively nearby data to maximize global
        // cache use.
        queue->work_items.Put(cur->node().begin(), cur);
        n_grains_++;
      } else {
        for (index_t k = 0; k < Node::CARDINALITY; k++) {
          *node_stack.AddBack() = cur->GetChild(tree_, k);
        }
      }
    }
  } else {
    for (index_t k = 0; k < Node::CARDINALITY; k++) {
      DistributeInitialWork_(decomp_node->child(k), node->GetChild(tree_, k));
    }
  }
}

template<typename Node>
void CentroidWorkQueue<Node>::GetWork(int rank_num, ArrayList<Grain> *work) {
  InternalNode *found_node;
  ProcessWorkQueue *queue = &rankes_[rank_num];

  found_node = NULL;

  while (found_node == NULL && !queue->work_items.is_empty()) {
    InternalNode *node = queue->work_items.Pop();
    if (node->info() == NONE) {
      found_node = node;
    }
  }

  if (!found_node) {
    if (!no_overflow_) {
      Vector center;
      center.Copy(queue->sum_centers);
      la::Scale(1.0 / queue->n_centers, &center);

      MinHeap<double, InternalNode*> prio;

      prio.Init();
      prio.Put(0, root_);

      // Single-tree nearest-node search
      while (!prio.is_empty()) {
        InternalNode *node = prio.Pop();
        if (node->info() != ALL) {
          if (!node->is_complete()) {
            // We can't explore a node that is missing children or whose
            // count is too large.
            if (node->info() == NONE) {
              found_node = node;
              n_overflow_points_ += found_node->count();
              n_overflows_++;
              break;
            }
          } else {
            DEBUG_ASSERT(node->is_complete());
            for (int i = 0; i < Node::CARDINALITY; i++) {
              InternalNode *child = node->child(i);
              prio.Put(child->node().bound().MinDistanceSq(center), child);
            }
          }
        }
      }
    }
  } else {
    n_preferred_++;
  }

  if (found_node == NULL) {
    work->Init();
  } else {
    // Show user-friendly status messages every 5% increment
    index_t count = found_node->count();
    n_assigned_points_ += count;

    percent_indicator("scheduled", n_assigned_points_, root_->count());

    // Mark all children as complete (non-recursive version)
    ArrayList<InternalNode*> stack;
    stack.Init();
    *stack.AddBack() = found_node;
    while (stack.size() != 0) {
      InternalNode *c = *stack.PopBackPtr();
      c->info() = ALL;
      for (index_t k = 0; k < Node::CARDINALITY; k++) {
        InternalNode *c_child = c->child(k);
        if (c_child) {
          *stack.AddBack() = c_child;
        }
      }
    }

    // Mark my parents as partially or fully complete
    for (InternalNode *p = found_node->parent(); p != NULL; p = p->parent()) {
      DEBUG_ASSERT(p->is_complete());
      p->info() = ALL;
      for (int k = 0; k < Node::CARDINALITY; k++) {
        if (p->child(k)->info() != ALL) {
          p->info() = SOME;
          break;
        }
      }
    }

    work->Init(1);
    Grain *grain = &(*work)[0];
    grain->node_index = found_node->index();
    grain->node_end_index = found_node->end_index();
    grain->point_begin_index = found_node->node().begin();
    grain->point_end_index = found_node->node().end();

    Vector midpoint;
    found_node->node().bound().CalculateMidpoint(&midpoint);
    la::AddTo(midpoint, &queue->sum_centers);
    queue->n_centers++;
  }
}

template<typename Node>
void CentroidWorkQueue<Node>::Report(struct datanode *module) {
  fx_format_result(module, "n_grains", "%"LI"d",
      n_preferred_ + n_overflows_);
  fx_format_result(module, "overflow_grain_ratio", "%.4f",
      1.0 * n_overflows_ / (n_preferred_ + n_overflows_));
  fx_format_result(module, "n_overflows", "%"LI"d",
      n_overflows_);
  fx_format_result(module, "overflow_point_ratio", "%.4f",
      1.0 * n_overflow_points_ / root_->count());
}
